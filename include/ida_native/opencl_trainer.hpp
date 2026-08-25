#pragma once

#include <filesystem>
#include <cstddef>
#include <functional>
#include <string>

#include "ida_native/opencl_runtime.hpp"
#include "ida_native/request.hpp"

namespace ida_native {

struct OpenCLTrainMetrics {
    int step{0};
    float loss{0.0f};
    double tokens_per_second{0.0};
    std::size_t tokens{0};
};

struct OpenCLTrainResult {
    int steps{0};
    float final_loss{0.0f};
    double tokens_per_second{0.0};
    std::size_t tokens_processed{0};
    std::string device;
    bool parameters_changed{false};
};

using OpenCLTrainCallback = std::function<void(const OpenCLTrainMetrics&)>;

OpenCLTrainResult run_opencl_training(
    const NativeRequest& request,
    OpenCLRuntime& runtime,
    const std::filesystem::path& kernel_source,
    OpenCLTrainCallback on_step = {}
);

}  // namespace ida_native
