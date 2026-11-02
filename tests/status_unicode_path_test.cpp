#include "ida_native/status.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <nlohmann/json.hpp>

int main() {
    namespace fs = std::filesystem;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = fs::temp_directory_path() / ("foundry_status_unicode_" + std::to_string(stamp));
    bool created = false;
    try {
        if (!fs::create_directory(root)) throw std::runtime_error("cannot create private fixture directory");
        created = true;
        const auto directory = root / fs::u8path("asset-\xCE\xA9");
        const auto status = directory / "status.json";
        ida_native::NativeRequest request{};
        ida_native::write_status(status, request, "fixture_first");
        ida_native::write_status(status, request, "fixture_second");
        std::ifstream input(status, std::ios::binary);
        const auto value = nlohmann::json::parse(input);
        if (value.value("phase", "") != "fixture_second") throw std::runtime_error("status phase was not replaced");
        auto temporary = status;
        temporary += ".tmp";
        if (fs::exists(temporary)) throw std::runtime_error("temporary status file remains after success");
        fs::remove_all(root);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        std::error_code ignored;
        if (created) fs::remove_all(root, ignored);
        return 1;
    }
}
