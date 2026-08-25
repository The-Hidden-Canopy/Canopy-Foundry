#include "ida_native/cpu_trainer.hpp"
#include "ida_native/cli_welcome.hpp"
#include "ida_native/request.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::json;

struct Arguments {
    std::filesystem::path request_json;
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments args{};
    for (int i = 1; i < argc; ++i) {
        const std::string token = argv[i];
        if (token == "--help" || token == "-h") {
            ida_native::print_cli_welcome(std::cout);
            std::cout << "Usage: ida_native_cpu_train --request-json REQUEST.json\n";
            std::exit(0);
        }
        if (token == "--request-json" && i + 1 < argc) {
            args.request_json = argv[++i];
            continue;
        }
        throw std::runtime_error("unknown or incomplete option: " + token);
    }
    if (args.request_json.empty()) throw std::runtime_error("--request-json is required");
    return args;
}

void write_metrics(
    const std::filesystem::path& output,
    const ida_native::NativeRequest& request,
    const ida_native::CPUTrainResult& result
) {
    std::filesystem::create_directories(output);
    const json payload = {
        {"type", "complete"},
        {"status", "complete"},
        {"run_id", request.job_id},
        {"expected_terminal_phase", request.expected_terminal_phase},
        {"backend", "native_cpu"},
        {"device", result.device},
        {"kernel_variant", result.kernel_variant},
        {"threads", result.threads},
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
        if (request.device.runtime != "cpu") {
            throw std::runtime_error(
                "this binary is the CPU backend; request.device.runtime must be cpu");
        }
        const auto on_step = [](const ida_native::CPUTrainMetrics& report) {
            std::cout << nlohmann::json{
                {"type", "step"},
                {"step", report.step},
                {"loss", report.loss},
                {"tokens", report.tokens},
                {"tokens_per_second", report.tokens_per_second},
            }.dump() << '\n' << std::flush;
        };
        const auto result = ida_native::run_cpu_training(request, on_step);
        write_metrics(request.output_dir, request, result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ida_native_cpu_train: " << error.what() << '\n';
        return 1;
    }
}
