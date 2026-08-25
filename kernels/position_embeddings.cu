#include "ida_native/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace ida_native {
namespace {

__device__ __forceinline__ int packed_position(
    const std::uint16_t* segs,
    int b,
    int s,
    int sequence_length
) {
    if (segs == nullptr) return s;
    const int start = static_cast<int>(
        segs[static_cast<std::size_t>(b) * sequence_length + s]);
    // Malformed segment metadata is handled fail-closed at the element
    // boundary. Request validation normally rejects it before launch; the
    // guard prevents a bad local buffer from turning into an OOB read.
    return start <= s ? s - start : -1;
}

__global__ void k_position_embedding_forward(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ position,
    const std::uint16_t* __restrict__ segs,
    int B,
    int S,
    int H,
    int max_positions
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(B) * S * H;
    if (i >= total) return;

    const int token = static_cast<int>(i / H);
    const int b = token / S;
    const int s = token % S;
    const int pos = packed_position(segs, b, s, S);
    if (pos < 0 || pos >= max_positions) return;

    hidden[i] = __float2bfloat16(
        __bfloat162float(hidden[i]) +
        __bfloat162float(position[static_cast<std::size_t>(pos) * H + (i % H)]));
}

__global__ void k_position_embedding_backward(
    const __nv_bfloat16* __restrict__ grad_hidden,
    const std::uint16_t* __restrict__ segs,
    float* __restrict__ grad_position,
    int B,
    int S,
    int H,
    int max_positions
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(B) * S * H;
    if (i >= total) return;

    const int token = static_cast<int>(i / H);
    const int b = token / S;
    const int s = token % S;
    const int pos = packed_position(segs, b, s, S);
    if (pos < 0 || pos >= max_positions) return;

    atomicAdd(
        &grad_position[static_cast<std::size_t>(pos) * H + (i % H)],
        __bfloat162float(grad_hidden[i]));
}

void validate_position_args(
    const void* hidden,
    const void* position,
    int B,
    int S,
    int H,
    int max_positions
) {
    if (B < 0 || S < 0 || H < 0) {
        throw std::invalid_argument("position embedding dimensions must be non-negative");
    }
    if (max_positions <= 0) {
        throw std::invalid_argument("position embedding table must have a positive size");
    }
    if (B == 0 || S == 0 || H == 0) return;
    if (hidden == nullptr || position == nullptr) {
        throw std::invalid_argument("position embedding received a null tensor pointer");
    }
}

}  // namespace

void position_embedding_forward(
    __nv_bfloat16* d_hidden,
    const __nv_bfloat16* d_position,
    const std::uint16_t* d_segs,
    int B,
    int S,
    int H,
    int max_positions,
    cudaStream_t stream
) {
    validate_position_args(d_hidden, d_position, B, S, H, max_positions);
    if (B == 0 || S == 0 || H == 0) return;
    const std::size_t total = static_cast<std::size_t>(B) * S * H;
    k_position_embedding_forward<<<
        static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        d_hidden, d_position, d_segs, B, S, H, max_positions);
}

void position_embedding_backward(
    const __nv_bfloat16* d_grad_hidden,
    const std::uint16_t* d_segs,
    float* d_grad_position,
    int B,
    int S,
    int H,
    int max_positions,
    cudaStream_t stream
) {
    if (B < 0 || S < 0 || H < 0) {
        throw std::invalid_argument("position embedding dimensions must be non-negative");
    }
    if (max_positions <= 0) {
        throw std::invalid_argument("position embedding table must have a positive size");
    }
    if (B == 0 || S == 0 || H == 0) return;
    if (d_grad_hidden == nullptr || d_grad_position == nullptr) {
        throw std::invalid_argument("position embedding backward received a null tensor pointer");
    }
    const std::size_t total = static_cast<std::size_t>(B) * S * H;
    k_position_embedding_backward<<<
        static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        d_grad_hidden, d_segs, d_grad_position, B, S, H, max_positions);
}

}  // namespace ida_native
