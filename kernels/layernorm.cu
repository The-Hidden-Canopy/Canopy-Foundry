#include "ida_native/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ida_native {
namespace {

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__device__ __forceinline__ float block_sum(float value, float* shared) {
    value = warp_sum(value);
    if ((threadIdx.x & 31) == 0)
        shared[threadIdx.x >> 5] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < (blockDim.x + 31) / 32
            ? shared[threadIdx.x] : 0.0f;
        value = warp_sum(value);
    }
    if (threadIdx.x == 0)
        shared[0] = value;
    __syncthreads();
    return shared[0];
}

__global__ void k_layernorm_forward(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ inv_std,
    int H,
    float eps
) {
    const int row = blockIdx.x;
    const auto* x_row = x + static_cast<std::size_t>(row) * H;
    auto* out_row = out + static_cast<std::size_t>(row) * H;
    __shared__ float sm[32];

    float sum = 0.0f;
    for (int h = threadIdx.x; h < H; h += blockDim.x)
        sum += __bfloat162float(x_row[h]);
    const float mean = block_sum(sum, sm) / static_cast<float>(H);

    float sq = 0.0f;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        const float centered = __bfloat162float(x_row[h]) - mean;
        sq += centered * centered;
    }
    const float variance = block_sum(sq, sm) / static_cast<float>(H);
    const float inv = rsqrtf(variance + eps);
    if (threadIdx.x == 0 && inv_std)
        inv_std[row] = inv;
    __syncthreads();

    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        const float normalized =
            (__bfloat162float(x_row[h]) - mean) * inv;
        out_row[h] = __float2bfloat16(
            normalized * __bfloat162float(scale[h])
            + __bfloat162float(bias[h]));
    }
}

__global__ void k_layernorm_backward(
    const __nv_bfloat16* __restrict__ d_out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const float* __restrict__ saved_inv_std,
    __nv_bfloat16* __restrict__ d_x,
    float* __restrict__ d_scale,
    float* __restrict__ d_bias,
    int H,
    float eps
) {
    const int row = blockIdx.x;
    const auto* d_out_row = d_out + static_cast<std::size_t>(row) * H;
    const auto* x_row = x + static_cast<std::size_t>(row) * H;
    auto* d_x_row = d_x + static_cast<std::size_t>(row) * H;
    __shared__ float sm1[32];
    __shared__ float sm2[32];

    float mean_sum = 0.0f;
    for (int h = threadIdx.x; h < H; h += blockDim.x)
        mean_sum += __bfloat162float(x_row[h]);
    mean_sum = warp_sum(mean_sum);
    if ((threadIdx.x & 31) == 0)
        sm1[threadIdx.x >> 5] = mean_sum;
    __syncthreads();
    float mean = threadIdx.x < 32
        ? (threadIdx.x < (blockDim.x + 31) / 32 ? sm1[threadIdx.x] : 0.0f)
        : 0.0f;
    if (threadIdx.x < 32)
        mean = warp_sum(mean);
    if (threadIdx.x == 0)
        sm1[0] = mean / static_cast<float>(H);
    __syncthreads();
    mean = sm1[0];

    const float inv = saved_inv_std[row];
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        const float dy = __bfloat162float(d_out_row[h]);
        const float gamma = __bfloat162float(scale[h]);
        const float centered = __bfloat162float(x_row[h]) - mean;
        sum1 += dy * gamma;
        sum2 += dy * gamma * centered;
        atomicAdd(&d_scale[h], dy * centered * inv);
        atomicAdd(&d_bias[h], dy);
    }
    sum1 = warp_sum(sum1);
    sum2 = warp_sum(sum2);
    if ((threadIdx.x & 31) == 0) {
        const int warp = threadIdx.x >> 5;
        sm1[warp] = sum1;
        sm2[warp] = sum2;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const int warps = (blockDim.x + 31) / 32;
        sum1 = threadIdx.x < warps ? sm1[threadIdx.x] : 0.0f;
        sum2 = threadIdx.x < warps ? sm2[threadIdx.x] : 0.0f;
        sum1 = warp_sum(sum1);
        sum2 = warp_sum(sum2);
    }
    if (threadIdx.x == 0) {
        sm1[0] = sum1;
        sm2[0] = sum2;
    }
    __syncthreads();
    sum1 = sm1[0];
    sum2 = sm2[0];
    const float inv2 = inv * inv;
    const float inv_h = 1.0f / static_cast<float>(H);
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        const float dy = __bfloat162float(d_out_row[h]);
        const float gamma = __bfloat162float(scale[h]);
        const float centered = __bfloat162float(x_row[h]) - mean;
        const float dx = inv * inv_h *
            (static_cast<float>(H) * dy * gamma - sum1
             - centered * inv2 * sum2);
        d_x_row[h] = __float2bfloat16(dx);
    }
    (void)eps;
}

}  // namespace

void layernorm_forward(
    const __nv_bfloat16* d_x,
    const __nv_bfloat16* d_scale,
    const __nv_bfloat16* d_bias,
    __nv_bfloat16* d_out,
    float* d_inv_std,
    int rows, int H, float eps,
    cudaStream_t stream
) {
    k_layernorm_forward<<<rows, 256, 0, stream>>>(
        d_x, d_scale, d_bias, d_out, d_inv_std, H, eps);
}

void layernorm_backward(
    const __nv_bfloat16* d_grad_out,
    const __nv_bfloat16* d_x,
    const __nv_bfloat16* d_scale,
    const float* d_inv_std,
    __nv_bfloat16* d_grad_x,
    float* d_grad_scale,
    float* d_grad_bias,
    int rows, int H, float eps,
    cudaStream_t stream
) {
    k_layernorm_backward<<<rows, 256, 0, stream>>>(
        d_grad_out, d_x, d_scale, d_inv_std, d_grad_x,
        d_grad_scale, d_grad_bias, H, eps);
}

}  // namespace ida_native
