#include "ida_native/fp8_e4m3.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

__global__ void pack_unpack_kernel(
    const float* input, std::uint8_t* packed, float* unpacked, int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const auto code = ida_native::fp8_e4m3::pack(input[i]);
    packed[i] = code.bits;
    unpacked[i] = ida_native::fp8_e4m3::unpack(code.bits);
}

int main() {
    const float neg_zero = -0.0f;
    const std::vector<float> values = {
        0.0f, neg_zero, 1.0f, -1.0f, 0.015625f, 448.0f, -448.0f,
        447.0f, std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
    };
    const int n = static_cast<int>(values.size());
    float* d_input = nullptr;
    std::uint8_t* d_packed = nullptr;
    float* d_unpacked = nullptr;
    if (cudaMalloc(&d_input, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_packed, n * sizeof(std::uint8_t)) != cudaSuccess ||
        cudaMalloc(&d_unpacked, n * sizeof(float)) != cudaSuccess) return 2;
    cudaMemcpy(d_input, values.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    pack_unpack_kernel<<<1, 32>>>(d_input, d_packed, d_unpacked, n);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    std::vector<std::uint8_t> packed(n);
    std::vector<float> unpacked(n);
    cudaMemcpy(packed.data(), d_packed, n * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(unpacked.data(), d_unpacked, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        const auto reference = ida_native::fp8_e4m3::pack(values[i]);
        if (packed[i] != reference.bits) {
            std::fprintf(stderr, "pack mismatch at %d: got 0x%02x want 0x%02x\n",
                         i, packed[i], reference.bits);
            return 4;
        }
        if (reference.finite && std::isfinite(values[i]) &&
            std::fabs(unpacked[i] - ida_native::fp8_e4m3::unpack(reference.bits)) > 0.0f) {
            std::fprintf(stderr, "unpack mismatch at %d\n", i);
            return 5;
        }
    }
    if (packed[1] != 0x80u || packed[5] != 0x7eu || packed[6] != 0xfeu ||
        packed[8] != 0x7fu || packed[9] != 0x7fu) return 6;
    cudaFree(d_input); cudaFree(d_packed); cudaFree(d_unpacked);
    std::puts("fp8_e4m3_probe: PASS");
    return 0;
}
