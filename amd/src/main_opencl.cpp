#include "ida_native/opencl_runtime.hpp"
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
#include "ida_native/opencl_ontology.hpp"
#endif
#include "ida_native/opencl_cpp.hpp"
#include "ida_native/opencl_trainer.hpp"
#include "ida_native/cli_welcome.hpp"
#include "ida_native/request.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::json;

struct Arguments {
    std::filesystem::path request_json;
    std::filesystem::path kernel_source;
    int device{0};
    bool self_check{false};
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
    std::filesystem::path ontology_path;
    bool ontology{false};
#endif
};

void usage() {
    std::cout << "Usage: ida_native_opencl_train --request-json REQUEST.json "
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
                 "[--device GPU] [--kernel-source FILE] [--ontology] "
                 "[--ontology-path FILE]\n"
#else
                 "[--device GPU] [--kernel-source FILE]\n"
#endif
                 "       ida_native_opencl_train --self-check [--device GPU]\n";
}
Arguments parse_arguments(int argc, char** argv) {
    Arguments args{};
    for (int i = 1; i < argc; ++i) {
        const std::string token = argv[i];
        if (token == "--help" || token == "-h") {
            usage();
            std::exit(0);
        }
        if (token == "--self-check") {
            args.self_check = true;
            continue;
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
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
        if (token == "--ontology") {
            args.ontology = true;
            continue;
        }
        if (token == "--ontology-path" && i + 1 < argc) {
            args.ontology_path = argv[++i];
            args.ontology = true;
            continue;
        }
#endif
        throw std::runtime_error("unknown or incomplete option: " + token);
    }
    if (!args.self_check && args.request_json.empty()) {
        throw std::runtime_error("--request-json is required unless --self-check is used");
    }
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
        std::optional<ida_native::NativeRequest> request;
        if (!args.self_check) {
            request.emplace(ida_native::load_request(args.request_json));
            if (request->device.runtime != "opencl") {
                throw std::runtime_error(
                    "this binary is the OpenCL backend; request.device.runtime must be opencl");
            }
        }
        ida_native::OpenCLRuntime runtime = ida_native::create_opencl_runtime(args.device);
        try {
            std::cerr << "[ida_native_opencl_train] device: " << runtime.device_name << '\n';
            std::cerr << "[ida_native_opencl_train] OpenCL: " << runtime.device_version
                      << ", driver: " << runtime.driver_version
                      << ", compute_units: " << runtime.compute_units
                      << ", global_memory: " << (runtime.global_memory_bytes / (1024ull * 1024ull))
                      << " MiB\n";
            if (args.self_check) {
                const auto check = ida_native::opencl::run_vector_add_check(runtime);
                const nlohmann::json payload = {
                    {"type", "self_check"},
                    {"status", check.passed ? "passed" : "failed"},
                    {"backend", "native_opencl"},
                    {"device", runtime.device_name},
                    {"operation", "vector_add"},
                    {"elements", check.elements},
                    {"maximum_error", check.maximum_error},
                };
                std::cout << payload.dump() << '\n' << std::flush;
                ida_native::destroy_opencl_runtime(runtime);
                return check.passed ? 0 : 1;
            }
            const auto kernel_source = locate_kernel_source(args.kernel_source, argv[0]);
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
            std::optional<ida_native::OpenCLOntologyRecorder> ontology;
            ontology.emplace(*request, runtime, args.ontology_path, args.ontology);
            ontology->emit_smoke_launches(*request);
            const auto on_step = [&ontology](const ida_native::OpenCLTrainMetrics& report) {
                ontology->observe_step(report);
#else
            const auto on_step = [](const ida_native::OpenCLTrainMetrics& report) {
#endif
                std::cout << nlohmann::json{
                    {"type", "step"},
                    {"step", report.step},
                    {"loss", report.loss},
                    {"tokens", report.tokens},
                    {"tokens_per_second", report.tokens_per_second},
                }.dump() << '\n' << std::flush;
            };
            const auto result = ida_native::run_opencl_training(
                *request, runtime, kernel_source, on_step);
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
            ontology->complete(result);
#endif
            write_metrics(request->output_dir, *request, result);
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
