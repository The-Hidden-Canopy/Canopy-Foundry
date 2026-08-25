#pragma once

#include <cstddef>
#include <functional>
#include <string>

#include "ida_native/request.hpp"

namespace ida_native {

struct CPUTrainMetrics {
    int step{0};
    float loss{0.0f};
    double tokens_per_second{0.0};
    std::size_t tokens{0};
};

struct CPUTrainResult {
    int steps{0};
    float final_loss{0.0f};
    double tokens_per_second{0.0};
    std::size_t tokens_processed{0};
    std::size_t threads{1};
    std::string device{"cpu"};
    std::string kernel_variant;
    bool parameters_changed{false};
};

using CPUTrainCallback = std::function<void(const CPUTrainMetrics&)>;

CPUTrainResult run_cpu_training(
    const NativeRequest& request,
    CPUTrainCallback on_step = {}
);

}  // namespace ida_native
