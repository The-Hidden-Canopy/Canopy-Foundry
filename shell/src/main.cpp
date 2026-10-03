#include "canopy_foundry/shell/model.hpp"
#include "canopy_foundry/shell/platform.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

namespace {

using canopy_foundry::shell::ShellState;

void print_usage() {
    std::cout
        << "Usage: foundry_shell [--frontend-root DIR] [--document FILE] [--simulate]\n"
        << "       foundry_shell --check [--frontend-root DIR]\n"
        << "       foundry_shell --headless [--frontend-root DIR] [--document FILE] [--simulate]\n";
}

std::filesystem::path default_frontend_root() {
    std::vector<std::filesystem::path> candidates;
    const auto current = std::filesystem::current_path();
    candidates.push_back(current / "frontend");
    candidates.push_back(current.parent_path() / "frontend");
    candidates.push_back(current.parent_path() / "share" / "canopy-foundry" / "frontend");

#ifdef _WIN32
    wchar_t module_path[32768] = {};
    const DWORD length = GetModuleFileNameW(
        nullptr, module_path, static_cast<DWORD>(std::size(module_path)));
    if (length > 0 && length < std::size(module_path)) {
        const auto executable = std::filesystem::path(module_path);
        candidates.push_back(executable.parent_path() / "frontend");
        candidates.push_back(executable.parent_path().parent_path() / "frontend");
        candidates.push_back(
            executable.parent_path().parent_path() /
            "share" / "canopy-foundry" / "frontend");
    }
#else
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (!error) {
        candidates.push_back(executable.parent_path() / "frontend");
        candidates.push_back(executable.parent_path().parent_path() / "frontend");
        candidates.push_back(
            executable.parent_path().parent_path() /
            "share" / "canopy-foundry" / "frontend");
    }
#endif

    for (const auto& candidate : candidates) {
        if (canopy_foundry::shell::frontend_available(candidate)) {
            return candidate;
        }
    }
    return current / "frontend";
}

int run_command(const std::vector<std::string>& arguments) {
    ShellState state;
    state.frontend_root = default_frontend_root();
    std::filesystem::path document_path;
    bool check_only = false;
    bool headless = false;
    bool simulate = false;

    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const std::string_view argument(arguments[index]);
        if (argument == "--help" || argument == "-h") {
            print_usage();
            return 0;
        }
        if (argument == "--check") {
            check_only = true;
            continue;
        }
        if (argument == "--headless") {
            headless = true;
            continue;
        }
        if (argument == "--simulate") {
            simulate = true;
            continue;
        }
        if (argument == "--frontend-root" || argument == "--document") {
            if (index + 1 >= arguments.size()) {
                std::cerr << "foundry_shell: missing value for " << argument << '\n';
                return 2;
            }
            const std::filesystem::path value(arguments[++index]);
            if (argument == "--frontend-root") {
                state.frontend_root = value;
            } else {
                document_path = value;
            }
            continue;
        }
        std::cerr << "foundry_shell: unknown option " << argument << '\n';
        print_usage();
        return 2;
    }

    state.frontend_present = canopy_foundry::shell::frontend_available(
        state.frontend_root);
    std::string sprite_error;
    const bool sprite_loaded = canopy_foundry::shell::load_ida_sprite_sheet(
        state.frontend_root / "assets" / "ida-cli-sprite.json",
        state.ida_sprite,
        sprite_error);
    (void)sprite_loaded;
    if (!document_path.empty()) {
        std::string error;
        if (!canopy_foundry::shell::load_document(state, document_path, error)) {
            std::cerr << "foundry_shell: " << error << ": "
                      << document_path.string() << '\n';
            return 2;
        }
    }
    if (simulate) {
        canopy_foundry::shell::start_simulation(state.simulation);
        state.view = canopy_foundry::shell::View::Simulation;
    }

    if (check_only) {
        std::cout << "frontend_root=" << state.frontend_root.string() << '\n'
                  << "frontend_present=" << (state.frontend_present ? "true" : "false")
                  << '\n'
                  << "ida_sprite_present=" << (state.ida_sprite.loaded ? "true" : "false")
                  << '\n'
                  << "ida_sprite_frames=" << state.ida_sprite.frames.size() << '\n'
                  << "simulation_available=true\n"
                  << "document_loaded=" << (state.document.loaded ? "true" : "false")
                  << '\n';
        return state.frontend_present ? 0 : 3;
    }

    if (headless) {
        std::cout << "view=" << canopy_foundry::shell::view_name(state.view) << '\n'
                  << "status=" << state.status << '\n'
                  << "frontend_present=" << (state.frontend_present ? "true" : "false")
                  << '\n'
                  << "ida_sprite_present=" << (state.ida_sprite.loaded ? "true" : "false")
                  << '\n'
                  << "simulation_stage="
                  << canopy_foundry::shell::simulation_stage_name(state.simulation.stage) << '\n'
                  << "simulation_progress=" << state.simulation.progress_percent << '\n'
                  << '\n';
        if (state.document.loaded) {
            std::cout << "document=" << state.document.name << '\n'
                      << "document_type=" << state.document.type << '\n'
                      << "posture=" << state.document.posture << '\n';
        }
        return 0;
    }

    return canopy_foundry::shell::run_native_shell(state);
}

}  // namespace

#ifdef _WIN32

std::string utf8_from_wide(const wchar_t* value) {
    if (value == nullptr || *value == L'\0') return {};
    const int required = WideCharToMultiByte(
        CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (required <= 1) return {};
    std::string result(static_cast<std::size_t>(required), '\0');
    WideCharToMultiByte(
        CP_UTF8, 0, value, -1, result.data(), required, nullptr, nullptr);
    result.pop_back();
    return result;
}

int WINAPI WinMain(
    HINSTANCE,
    HINSTANCE,
    LPSTR,
    int
) {
    int count = 0;
    LPWSTR* wide_arguments = CommandLineToArgvW(
        GetCommandLineW(), &count);
    if (wide_arguments == nullptr) return 2;

    std::vector<std::string> arguments;
    arguments.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        arguments.push_back(utf8_from_wide(wide_arguments[index]));
    }
    LocalFree(wide_arguments);
    return run_command(arguments);
}

#else

int main(int argc, char** argv) {
    std::vector<std::string> arguments;
    arguments.reserve(static_cast<std::size_t>(argc));
    for (int index = 0; index < argc; ++index) {
        arguments.emplace_back(argv[index] == nullptr ? "" : argv[index]);
    }
    return run_command(arguments);
}

#endif
