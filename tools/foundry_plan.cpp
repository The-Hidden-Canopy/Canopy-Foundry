#include "canopy_foundry/execution/service.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

void usage() {
    std::cout
        << "Usage: foundry plan REQUEST.json [--pretty]\n"
        << "       foundry validate REQUEST.json [--pretty]\n"
        << "       foundry capabilities [--pretty]\n"
        << "       foundry checkpoint DIRECTORY\n"
        << "       foundry run REQUEST.json | foundry status RUN-ID | foundry cancel RUN-ID\n\n"
        << "Start: foundry capabilities --pretty\n"
        << "       nvidia-smi; nvcc --version  (check the device and toolkit)\n"
        << "       foundry plan configs/examples/planning_only_request.json --pretty\n"
        << "The example is a one-step plan fixture, not runnable input data.\n"
        << "Plan and validate run the same planner; neither verifies the runtime.\n"
        << "A plan requires runtime and hardware verification. CPU/OpenCL training\n"
        << "profiles are disabled; CUDA requires a supported GPU and toolkit.\n"
        << "Checkpoint inspection checks a pair and header bounds only. Exit 3\n"
        << "means rejected; exit 4 means native validation remains required.\n"
        << "Run checks the request and reports whether the local service is available.\n"
        << "Run and cancel remain unavailable until the local service is connected.\n"
        << "If planning fails, check the request and retry; no run was started.\n"
        << "Guides: STARTER_GUIDE.md and docs/neural-foundry-guide.md\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage();
            return 2;
        }
        const std::string_view command(argv[1]);
        if (command == "help" || command == "--help" || command == "-h") {
            if (argc != 2) throw std::runtime_error("help takes no arguments");
            usage();
            return 0;
        }
        if (command == "run" || command == "status" || command == "cancel") {
            if (argc != 3) throw std::runtime_error("run, status and cancel each require one argument");
            const auto result = command == "run"
                ? canopy_foundry::start_local_run(argv[2])
                : command == "status" ? canopy_foundry::status_local_run(argv[2])
                : canopy_foundry::cancel_local_run(argv[2]);
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
            const auto inspection = canopy_foundry::inspect_checkpoint(argv[2]);
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
            std::filesystem::path(argv[2]), capabilities);
        std::cout << canopy_foundry::plan_json(resolution.plan, pretty) << '\n';
        return resolution.plan.accepted ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "foundry: " << error.what() << '\n';
        return 1;
    }
}
