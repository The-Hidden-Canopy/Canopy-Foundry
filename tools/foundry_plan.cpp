#include "canopy_foundry/execution/plan.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

void usage() {
    std::cout
        << "Usage: foundry plan REQUEST.json [--pretty]\n"
        << "       foundry capabilities [--pretty]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage();
            return 2;
        }
        const std::string_view command(argv[1]);
        bool pretty = false;
        for (int index = 2; index < argc; ++index) {
            if (std::string_view(argv[index]) == "--pretty") {
                pretty = true;
            } else if (command == "plan" && index == 2) {
                // The request path is consumed below; this branch only keeps
                // the option parser explicit and rejects positional extras.
                continue;
            } else {
                throw std::runtime_error("unknown or incomplete option");
            }
        }

#if defined(CANOPY_FOUNDRY_BUILD_CUDA) && CANOPY_FOUNDRY_BUILD_CUDA
        constexpr bool cuda_enabled = true;
#else
        constexpr bool cuda_enabled = false;
#endif
#if defined(CANOPY_FOUNDRY_BUILD_OPENCL) && CANOPY_FOUNDRY_BUILD_OPENCL
        constexpr bool opencl_enabled = true;
#else
        constexpr bool opencl_enabled = false;
#endif
#if defined(CANOPY_FOUNDRY_BUILD_CPU) && CANOPY_FOUNDRY_BUILD_CPU
        constexpr bool cpu_enabled = true;
#else
        constexpr bool cpu_enabled = false;
#endif

        const auto capabilities = canopy_foundry::RuntimeCapabilities::from_build(
            cuda_enabled, opencl_enabled, cpu_enabled);
        if (command == "capabilities") {
            if (argc > 3 || (argc == 3 && !pretty)) {
                throw std::runtime_error("capabilities accepts only --pretty");
            }
            std::cout << canopy_foundry::capabilities_json(capabilities, pretty) << '\n';
            return 0;
        }
        if (command != "plan" || argc < 3) {
            usage();
            return 2;
        }
        if (argc > 4 || (argc == 4 && !pretty)) {
            throw std::runtime_error("plan accepts REQUEST.json and optional --pretty");
        }

        const auto request = ida_native::load_request(std::filesystem::path(argv[2]));
        const auto resolution = canopy_foundry::resolve_execution_plan(
            request, capabilities);
        std::cout << canopy_foundry::plan_json(resolution.plan, pretty) << '\n';
        return resolution.plan.accepted ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "foundry: " << error.what() << '\n';
        return 1;
    }
}
