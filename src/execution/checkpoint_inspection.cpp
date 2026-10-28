#include "canopy_foundry/execution/service.hpp"
#include <array>
#include <cstdint>
#include <fstream>
#include <system_error>

namespace canopy_foundry {
namespace {
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
    char first = 0;
    in.get(first);
    return in && first == '{';
}
}
CheckpointInspection inspect_checkpoint(const std::filesystem::path& directory) {
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(directory, ec);
    if (ec || !std::filesystem::is_directory(status))
        return {false, "Checkpoint must be a local directory; other formats are unsupported."};
    if (!has_bounded_header(directory / "model.safetensors") ||
        !has_bounded_header(directory / "optimizer_state.safetensors"))
        return {false, "Checkpoint pair missing, unsupported or truncated."};
    return {true, "Pair and header bounds only; native format, lineage and generation validation required."};
}
}
