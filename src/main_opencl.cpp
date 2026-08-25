#include "ida_native/opencl_runtime.hpp"
#include "ida_native/opencl_trainer.hpp"
#include "ida_native/cli_welcome.hpp"
#include "ida_native/request.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::json;

struct Arguments {
    std::filesystem::path request_json;
    std::filesystem::path kernel_source;
    int device{0};
};

void usage() {
    ida_native::print_cli_welcome(std::cout);
    std::cout << "Usage: ida_native_opencl_train --request-json REQUEST.json "
                 "[--device GPU] [--kernel-source FILE]\n";
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments args{};
    for (int i = 1; i < argc; ++i) {
        const std::string token = argv[i];
        if (token == "--help" || token == "-h") {
            usage();
            std::exit(0);
        }
        if (token == "--request-json" && i + 1 < argc) {
            args.request_json = argv[++i];
            continue;
        }
        if (token == "--device" && i + 1 < argc) {
            args.device = std::stoi(argv[++i]);
            if (args.device < 0) throw std::runtime_error("--device must be nonnegative");
            continue;
        }
        if (token == "--kernel-source" && i + 1 < argc) {
            args.kernel_source = argv[++i];
            continue;
        }
        throw std::runtime_error("unknown or incomplete option: " + token);
    }
    if (args.request_json.empty()) throw std::runtime_error("--request-json is required");
    return args;
}

std::filesystem::path locate_kernel_source(
    const std::filesystem::path& explicit_path,
    const char* executable
) {
    if (!explicit_path.empty()) return explicit_path;
    if (const char* env = std::getenv("IDA_NATIVE_OPENCL_KERNELS"); env && env[0]) {
        return env;
    }
    const auto executable_path = std::filesystem::absolute(executable).parent_path();
    const std::filesystem::path candidates[] = {
        executable_path / "opencl_smoke.cl",
        std::filesystem::current_path() / "opencl_smoke.cl",
        std::filesystem::current_path() / "kernels" / "opencl_smoke.cl",
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::is_regular_file(candidate)) return candidate;
    }
    return candidates[0];
}

void write_metrics(
    const std::filesystem::path& output,
    const ida_native::NativeRequest& request,
    const ida_native::OpenCLTrainResult& result
) {
    std::filesystem::create_directories(output);
    const json payload = {
        {"type", "complete"},
        {"status", "complete"},
        {"run_id", request.job_id},
        {"expected_terminal_phase", request.expected_terminal_phase},
        {"backend", "native_opencl"},
        {"device", result.device},
        {"steps", result.steps},
        {"loss", result.final_loss},
        {"tokens", result.tokens_processed},
        {"tokens_per_second", result.tokens_per_second},
        {"parameters_changed", result.parameters_changed},
        {"checkpoint_written", false},
    };
    std::ofstream file(output / "metrics.json", std::ios::trunc);
    if (!file) throw std::runtime_error("unable to write " + (output / "metrics.json").string());
    file << payload.dump(2) << '\n';
    std::cout << payload.dump() << '\n' << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
        ida_native::print_cli_welcome(std::cerr);
        const ida_native::NativeRequest request = ida_native::load_request(args.request_json);
        if (request.device.runtime != "opencl") {
            throw std::runtime_error(
                "this binary is the OpenCL backend; request.device.runtime must be opencl");
        }
        ida_native::OpenCLRuntime runtime = ida_native::create_opencl_runtime(args.device);
        try {
            std::cerr << "[ida_native_opencl_train] device: " << runtime.device_name << '\n';
            std::cerr << "[ida_native_opencl_train] OpenCL: " << runtime.device_version
                      << ", driver: " << runtime.driver_version
                      << ", compute_units: " << runtime.compute_units
                      << ", global_memory: " << (runtime.global_memory_bytes / (1024ull * 1024ull))
                      << " MiB\n";
            const auto kernel_source = locate_kernel_source(args.kernel_source, argv[0]);
            const auto on_step = [](const ida_native::OpenCLTrainMetrics& report) {
                std::cout << nlohmann::json{
                    {"type", "step"},
                    {"step", report.step},
                    {"loss", report.loss},
                    {"tokens", report.tokens},
                    {"tokens_per_second", report.tokens_per_second},
                }.dump() << '\n' << std::flush;
            };
            const auto result = ida_native::run_opencl_training(
                request, runtime, kernel_source, on_step);
            write_metrics(request.output_dir, request, result);
            ida_native::destroy_opencl_runtime(runtime);
            return 0;
        } catch (...) {
            ida_native::destroy_opencl_runtime(runtime);
            throw;
        }
    } catch (const std::exception& error) {
        std::cerr << "ida_native_opencl_train: " << error.what() << '\n';
        return 1;
    }
}
