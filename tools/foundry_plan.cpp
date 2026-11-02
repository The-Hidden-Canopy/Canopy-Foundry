#include "canopy_foundry/execution/service.hpp"
#include "canopy_foundry/execution/owner_pipe.hpp"
#include "ida_native/utf8_arguments.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <nlohmann/json.hpp>

namespace {

void usage() {
    std::cout
        << "Usage: foundry plan REQUEST.json [--pretty]\n"
        << "       foundry validate REQUEST.json [--pretty]\n"
        << "       foundry capabilities [--pretty]\n"
        << "       foundry checkpoint DIRECTORY\n"
        << "       foundry inspect-local-run RUN_DIRECTORY\n"
        << "       foundry run REQUEST.json | foundry status RUN-ID | foundry cancel RUN-ID\n"
        << "       foundry current\n"
        << "       Add --owner-endpoint PRIVATE_FILE to these local control commands.\n\n"
        << "Start: foundry capabilities --pretty\n"
        << "       nvidia-smi; nvcc --version  (check the device and toolkit)\n"
        << "       foundry plan configs/examples/planning_only_request.json --pretty\n"
        << "The example is a one-step plan fixture, not runnable input data.\n"
        << "Plan and validate run the same planner; neither verifies the runtime.\n"
        << "A plan requires runtime and hardware verification. CPU/OpenCL training\n"
        << "profiles are disabled; CUDA requires a supported GPU and toolkit.\n"
        << "Checkpoint inspection checks a pair and JSON header syntax only. Exit 3\n"
        << "means rejected; exit 4 means native validation remains required.\n"
        << "Run checks the request and reports whether the local service is available.\n"
        << "The optional foreground foundry_owner pins and contains one local public CUDA run.\n"
        << "Without a verified owner endpoint, run and cancel remain unavailable.\n"
        << "If planning fails, check the request and retry; no run was started.\n"
        << "Guides: STARTER_GUIDE.md and docs/neural-foundry-guide.md\n";
}

// Informational only: no handle adoption, deletion, restart or success claim.
int inspect_local_run(const std::filesystem::path& directory) {
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(directory, ec);
    if (ec || !std::filesystem::is_directory(status)) {
        std::cerr << "Local run evidence directory unavailable.\n";
        return 4;
    }
    std::cout << "Unresolved local evidence; current process ownership and training outcome are not verified.\n";
    std::cout << "Inspect the original owner and retained handles; after owner loss, do not adopt a PID or infer success.\n";
    auto identity_path = directory.lexically_normal();
    if (identity_path.filename().empty()) identity_path = identity_path.parent_path();
    const auto expected_id = identity_path.filename().string();
    for (const char* name : {"launch-provenance.pending", "launch-provenance.json", "probe-process.json",
                             "probe-process-recovery-1.json", "probe-process-recovery-2.json", "trainer-process.json",
                             "trainer-process-recovery-1.json", "trainer-process-recovery-2.json", "output/status.json"}) {
        const auto path = directory / name;
        ec.clear();
        const auto file_status = std::filesystem::symlink_status(path, ec);
        if (ec || !std::filesystem::exists(file_status)) { std::cout << name << ": missing or unavailable\n"; continue; }
        if (!std::filesystem::is_regular_file(file_status)) { std::cout << name << ": not a regular file\n"; continue; }
        const auto size = std::filesystem::file_size(path, ec);
        if (ec || size > (2u << 20)) { std::cout << name << ": unavailable or over inspection limit\n"; continue; }
        std::ifstream in(path, std::ios::binary);
        std::string data(static_cast<std::size_t>(size), '\0');
        if (!in.read(data.data(), static_cast<std::streamsize>(data.size()))) { std::cout << name << ": incomplete read\n"; continue; }
        const auto value = nlohmann::json::parse(data, nullptr, false);
        if (!value.is_object()) { std::cout << name << ": malformed or partial\n"; continue; }
        if (std::string_view(name).find("process") != std::string_view::npos || std::string_view(name) == "launch-provenance.json") {
            if (!value.contains("run_id") || !value["run_id"].is_string() || value["run_id"].get<std::string>() != expected_id) {
                std::cout << name << ": run identity missing or mismatched\n"; continue;
            }
        }
        std::cout << name << ": parseable local data; contents require independent review\n";
    }
    return 4;
}

}  // namespace

int foundry_main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage();
            return 2;
        }
        const std::string_view command(argv[1]);
        if (command == "inspect-local-run") {
            if (argc != 3) throw std::runtime_error("inspect-local-run requires one run directory");
            return inspect_local_run(std::filesystem::u8path(argv[2]));
        }
        if (command == "help" || command == "--help" || command == "-h") {
            if (argc != 2) throw std::runtime_error("help takes no arguments");
            usage();
            return 0;
        }
        if (command == "run" || command == "status" || command == "cancel" || command == "current") {
            const bool current = command == "current";
            const int base = current ? 2 : 3;
            if (argc < base) throw std::runtime_error("run, status and cancel each require one argument");
            if (argc != base && argc != base + 2) throw std::runtime_error("invalid local control arguments");
            const std::string id = current ? "" : argv[2];
            canopy_foundry::LocalOperationResult result;
            if (argc == base + 2) {
                if (std::string_view(argv[base]) != "--owner-endpoint") throw std::runtime_error("unknown local control option");
                if (!canopy_foundry::local_owner_transport_enabled()) {
                    std::cerr << "Local owner transport is disabled in this build.\n";
                    return 4;
                }
                const auto endpoint = std::filesystem::u8path(argv[base + 1]);
                result = command == "run"
                    ? canopy_foundry::owner_file_start(endpoint, std::filesystem::absolute(std::filesystem::u8path(id)), std::chrono::seconds(5))
                    : canopy_foundry::owner_file_call(endpoint, current ? canopy_foundry::OwnerCommand::current : command == "status"
                        ? canopy_foundry::OwnerCommand::status : canopy_foundry::OwnerCommand::cancel, id, std::chrono::seconds(5));
            } else result = command == "run" ? canopy_foundry::start_local_run(std::filesystem::u8path(id))
                : command == "cancel" ? canopy_foundry::cancel_local_run(id) : canopy_foundry::status_local_run(id);
            std::cerr << result.message << '\n';
            return result.exit_code;
        }
        bool pretty = false;
        for (int index = 2; index < argc; ++index) {
            if (std::string_view(argv[index]) == "--pretty") {
                pretty = true;
            } else if ((command == "plan" || command == "validate" || command == "checkpoint") && index == 2) {
                // The request path is consumed below; this branch only keeps
                // the option parser explicit and rejects positional extras.
                continue;
            } else {
                throw std::runtime_error("unknown or incomplete option");
            }
        }

        const auto capabilities = canopy_foundry::build_capabilities();
        if (command == "capabilities") {
            if (argc > 3 || (argc == 3 && !pretty)) {
                throw std::runtime_error("capabilities accepts only --pretty");
            }
            std::cout << canopy_foundry::capabilities_json(capabilities, pretty) << '\n';
            return 0;
        }
        if (command == "checkpoint") {
            if (argc != 3 || pretty) throw std::runtime_error("checkpoint accepts a directory only");
            const auto inspection = canopy_foundry::inspect_checkpoint(std::filesystem::u8path(argv[2]));
            std::cout << inspection.message << '\n';
            // A partial inspection is never an authorization to resume.
            return inspection.requires_native_validation ? 4 : 3;
        }
        if ((command != "plan" && command != "validate") || argc < 3) {
            usage();
            return 2;
        }
        if (argc > 4 || (argc == 4 && !pretty)) {
            throw std::runtime_error("plan/validate accepts REQUEST.json and optional --pretty");
        }

        const auto resolution = canopy_foundry::plan_request(
            std::filesystem::u8path(argv[2]), capabilities);
        std::cout << canopy_foundry::plan_json(resolution.plan, pretty) << '\n';
        return resolution.plan.accepted ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "foundry: " << error.what() << '\n';
        return 1;
    }
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    try { ida_native::Utf8Arguments arguments(argc, argv); return foundry_main(argc, arguments.data()); }
    catch (const std::exception& error) { std::cerr << "foundry: " << error.what() << '\n'; return 1; }
}
#else
int main(int argc, char** argv) { return foundry_main(argc, argv); }
#endif
