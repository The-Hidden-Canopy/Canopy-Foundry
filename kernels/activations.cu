#include "ida_native/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace ida_native {
namespace {

__device__ __forceinline__ float stable_sigmoid(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

__device__ __forceinline__ float silu(float x) {
    return x * stable_sigmoid(x);
}

__device__ __forceinline__ float silu_derivative(float x) {
    const float sigmoid = stable_sigmoid(x);
    return sigmoid * (1.0f + x * (1.0f - sigmoid));
}

__global__ void k_swiglu_forward(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float gate_value = __bfloat162float(gate[i]);
    const float up_value = __bfloat162float(up[i]);
    out[i] = __float2bfloat16(silu(gate_value) * up_value);
}

__global__ void k_swiglu_backward(
    const __nv_bfloat16* __restrict__ grad_out,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ grad_gate,
    __nv_bfloat16* __restrict__ grad_up,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float gate_value = __bfloat162float(gate[i]);
    const float up_value = __bfloat162float(up[i]);
    const float grad = __bfloat162float(grad_out[i]);
    grad_gate[i] = __float2bfloat16(
        grad * up_value * silu_derivative(gate_value));
    grad_up[i] = __float2bfloat16(grad * silu(gate_value));
}

void validate_swiglu_args(
    const void* gate,
    const void* up,
    const void* out,
    int n
) {
    if (n < 0) throw std::invalid_argument("swiglu element count must be non-negative");
    if (n == 0) return;
    if (gate == nullptr || up == nullptr || out == nullptr) {
        throw std::invalid_argument("swiglu received a null tensor pointer");
    }
}

}  // namespace

void swiglu_forward(
    const __nv_bfloat16* d_gate,
    const __nv_bfloat16* d_up,
    __nv_bfloat16* d_out,
    int N,
    cudaStream_t stream
) {
    validate_swiglu_args(d_gate, d_up, d_out, N);
    if (N == 0) return;
    k_swiglu_forward<<<(N + 255) / 256, 256, 0, stream>>>(
        d_gate, d_up, d_out, N);
}

void swiglu_backward(
    const __nv_bfloat16* d_grad_out,
    const __nv_bfloat16* d_gate,
    const __nv_bfloat16* d_up,
    __nv_bfloat16* d_grad_gate,
    __nv_bfloat16* d_grad_up,
    int N,
    cudaStream_t stream
) {
    if (N < 0) throw std::invalid_argument("swiglu element count must be non-negative");
    if (N == 0) return;
    if (d_grad_out == nullptr || d_gate == nullptr || d_up == nullptr ||
        d_grad_gate == nullptr || d_grad_up == nullptr) {
        throw std::invalid_argument("swiglu backward received a null tensor pointer");
    }
    k_swiglu_backward<<<(N + 255) / 256, 256, 0, stream>>>(
        d_grad_out, d_gate, d_up, d_grad_gate, d_grad_up, N);
}

}  // namespace ida_native
