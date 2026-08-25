#pragma once

#include <cstddef>

namespace ida_native {

// AMD Zen-friendly host math. The implementation dispatches to AVX2/FMA
// when the current CPU exposes it and otherwise uses the scalar fallback.
const char* cpu_kernel_variant();
float cpu_dot(const float* left, const float* right, std::size_t count);

}  // namespace ida_native
