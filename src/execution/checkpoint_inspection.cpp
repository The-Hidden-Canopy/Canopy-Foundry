#include "canopy_foundry/execution/service.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <streambuf>
#include <system_error>
#include <nlohmann/json.hpp>

namespace canopy_foundry {
namespace {
// Expose exactly the declared JSON header, never the following tensor bytes.
// The parser streams up to the existing 100 MiB limit without first
// materializing the whole header or a DOM. Individual tokens may still be
// large. This is not a tensor or lineage validator.
class HeaderStreambuf final : public std::streambuf {
public:
    HeaderStreambuf(std::istream& source, std::uint64_t size) : source_(source), remaining_(size) {}
    bool complete() const { return !short_read_ && remaining_ == 0 && gptr() == egptr(); }
protected:
    int_type underflow() override {
        if (gptr() && gptr() < egptr()) return traits_type::to_int_type(*gptr());
        if (!remaining_) return traits_type::eof();
        const auto amount = static_cast<std::streamsize>(std::min<std::uint64_t>(remaining_, buffer_.size()));
        source_.read(buffer_.data(), amount);
        if (source_.gcount() != amount) { short_read_ = true; return traits_type::eof(); }
        remaining_ -= static_cast<std::uint64_t>(amount);
        setg(buffer_.data(), buffer_.data(), buffer_.data() + amount);
        return traits_type::to_int_type(*gptr());
    }
private:
    std::istream& source_;
    std::uint64_t remaining_;
    std::array<char, 4096> buffer_{};
    bool short_read_{false};
};

bool has_bounded_header(const std::filesystem::path& path) {
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(path, ec);
    if (ec || !std::filesystem::is_regular_file(status)) return false;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec || size < 10) return false;
    std::ifstream in(path, std::ios::binary);
    std::array<unsigned char, 8> bytes{};
    in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!in) return false;
    std::uint64_t header_size = 0;
    for (unsigned int i = 0; i < bytes.size(); ++i)
        header_size |= std::uint64_t{bytes[i]} << (8 * i);
    if (header_size < 2 || header_size > (100ull << 20) || header_size > size - 8)
        return false;
    if (in.peek() != '{') return false;
    HeaderStreambuf buffer(in, header_size);
    std::istream header(&buffer);
    return nlohmann::json::accept(header) && buffer.complete();
}
}
CheckpointInspection inspect_checkpoint(const std::filesystem::path& directory) {
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(directory, ec);
    if (ec || !std::filesystem::is_directory(status))
        return {false, "Checkpoint must be a local directory; other formats are unsupported."};
    try {
        if (!has_bounded_header(directory / "model.safetensors") ||
            !has_bounded_header(directory / "optimizer_state.safetensors"))
            return {false, "Checkpoint pair missing, truncated, or has malformed JSON headers."};
    } catch (const std::exception&) {
        return {false, "Checkpoint pair or headers could not be inspected."};
    }
    return {true, "Pair, header bounds and JSON syntax only; native tensor, format, lineage and generation validation required."};
}
}
