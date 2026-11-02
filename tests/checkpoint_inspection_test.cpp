#include "canopy_foundry/execution/service.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

void write_fixture(const std::filesystem::path& path, unsigned char length, const std::string& body) {
    std::ofstream out(path, std::ios::binary);
    const char header[8] = {static_cast<char>(length), 0, 0, 0, 0, 0, 0, 0};
    out.write(header, sizeof(header));
    out << body;
}
void write_long_fixture(const std::filesystem::path& path, const std::string& header, const std::string& payload) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const auto length = static_cast<std::uint64_t>(header.size());
    for (unsigned int i = 0; i < 8; ++i) out.put(static_cast<char>((length >> (8 * i)) & 0xff));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    assert(out);
}
int main(int argc, char** argv) {
    assert(argc == 2);
    const auto root = std::filesystem::path(argv[1]);
    if (!std::filesystem::create_directory(root)) return 2; // refuse an existing directory
    const auto model = root / "model.safetensors";
    const auto optimizer = root / "optimizer_state.safetensors";
    if (canopy_foundry::inspect_checkpoint(root).requires_native_validation) return 1;
    write_fixture(model, 2, "{}");
    if (canopy_foundry::inspect_checkpoint(root).requires_native_validation) return 1;
    write_fixture(optimizer, 25, "{}");
    if (canopy_foundry::inspect_checkpoint(root).requires_native_validation) return 1;
    write_fixture(optimizer, 2, "[]");
    if (canopy_foundry::inspect_checkpoint(root).requires_native_validation) return 1;
    write_fixture(optimizer, 2, "{}");
    const auto result = canopy_foundry::inspect_checkpoint(root);
    if (!result.requires_native_validation) return 1;
    if (result.message.find("native") == std::string::npos) return 1;
    write_fixture(model, 10, "{not json}");
    if (canopy_foundry::inspect_checkpoint(root).requires_native_validation) return 1;
    write_fixture(model, 6, "{\"a\":1");
    if (canopy_foundry::inspect_checkpoint(root).requires_native_validation) return 1;
    // The JSON stream must stop at the declared boundary, before binary data,
    // including when a padded header spans multiple input buffers.
    write_long_fixture(model, "{}" + std::string(8192, ' '), std::string("\xff\x00", 2));
    if (!canopy_foundry::inspect_checkpoint(root).requires_native_validation) return 1;
    if (canopy_foundry::inspect_checkpoint(model).requires_native_validation) return 1;
    std::filesystem::remove(model);
    std::filesystem::remove(optimizer);
    if (!std::filesystem::remove(root)) return 1;
}
