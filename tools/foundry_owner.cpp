#include "canopy_foundry/execution/local_backend.hpp"
#include "canopy_foundry/execution/owner_pipe.hpp"
#include <atomic>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {
std::atomic<bool> stop_requested{false};
bool should_stop(void*) noexcept { return stop_requested.load(); }
#ifdef _WIN32
BOOL WINAPI console_control(DWORD event) {
    if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT) { stop_requested.store(true); return TRUE; }
    return FALSE; // forced OS shutdown is not a verified process outcome
}
#endif
int owner_main(const std::vector<std::wstring>& arguments) {
    using namespace canopy_foundry;
    try {
        if (arguments.size() == 2 && (arguments[1] == L"--help" || arguments[1] == L"-h")) {
            std::cout << "Usage: foundry_owner --endpoint PRIVATE_FILE --binary ida_native_train.exe\n"
                         "       --binary-sha256 HEX --run-root EXISTING_DIRECTORY --device N [--max-seconds N]\n"
                         "Foreground local owner; no training starts until a Run request.\n"
                         "Ctrl+C requests exit only after verified quiescence. Use foundry cancel to stop a run.\n";
            return 0;
        }
        LocalBackendConfig config;
        std::filesystem::path endpoint;
        bool binary = false, digest = false, root = false, device = false;
        std::vector<std::wstring> seen;
        for (std::size_t i = 1; i < arguments.size(); i += 2) {
            const auto& key = arguments[i];
            if (i + 1 == arguments.size() || std::find(seen.begin(), seen.end(), key) != seen.end())
                throw std::runtime_error("incomplete or duplicate owner setting");
            seen.push_back(key);
            const auto& value = arguments[i + 1];
            const auto integer = [&] {
                if (value.empty() || value.find_first_not_of(L"0123456789") != std::wstring::npos)
                    throw std::runtime_error("owner setting requires a nonnegative integer");
                return std::stoi(value);
            };
            if (key == L"--endpoint") endpoint = value;
            else if (key == L"--binary") { config.binary = value; binary = true; }
            else if (key == L"--binary-sha256") { config.binary_sha256.assign(value.begin(), value.end()); digest = true; }
            else if (key == L"--run-root") { config.run_root = value; root = true; }
            else if (key == L"--device") { config.device = integer(); device = true; }
            else if (key == L"--max-seconds") config.maximum_runtime = std::chrono::seconds(integer());
            else throw std::runtime_error("unknown owner setting");
        }
        if (!binary || !digest || !root || !device || !endpoint.is_absolute() ||
            !config.binary.is_absolute() || !config.run_root.is_absolute())
            throw std::runtime_error("owner requires explicit absolute paths, binary pin and device");
#ifdef _WIN32
        if (!SetConsoleCtrlHandler(console_control, TRUE)) throw std::runtime_error("owner shutdown handler unavailable");
#endif
        // The operator pins the public CUDA executable; its actual runtime is
        // still unverified until the contained per-run native self-check.
        OwnerHost host(make_local_native_backend(std::move(config)), RuntimeCapabilities::from_build(true, false, false));
        std::string error;
        if (!host.open(endpoint, error)) throw std::runtime_error(error);
        std::cout << "Local owner ready; runtime unverified. Training, checkpoint and evaluation outcomes are separate.\n" << std::flush;
        const auto result = host.serve_until_quiescent(should_stop, nullptr, std::chrono::milliseconds(100), error);
        if (result == OwnerServeResult::error) { std::cerr << error << '\n'; return 4; }
        return host.may_shutdown() ? 0 : 4;
    } catch (const std::exception& error) { std::cerr << "Local owner unavailable: " << error.what() << '\n'; return 1; }
}
}
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) { return owner_main(std::vector<std::wstring>(argv, argv + argc)); }
#else
int main(int argc, char** argv) {
    std::vector<std::wstring> arguments;
    for (int i = 0; i < argc; ++i) arguments.emplace_back(argv[i], argv[i] + std::char_traits<char>::length(argv[i]));
    return owner_main(arguments);
}
#endif
