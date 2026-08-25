#include "ida_native/kernels.hpp"

#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ida_native {

// Warp-level reduce (intra-warp only, lane 0 holds result).
__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// Block-level reduce using shared memory scratch (size = nwarps floats).
// On return, thread 0 holds the result; smem contents are undefined.
__device__ float block_reduce_max(float val, float* smem) {
    const int lane   = threadIdx.x & 31;
    const int warpid = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) smem[warpid] = val;
    __syncthreads();
    // Let first warp reduce the warp-results stored in smem.
    const int nwarps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < nwarps) ? smem[threadIdx.x] : -FLT_MAX;
    if (warpid == 0) val = warp_reduce_max(val);
    return val;
}
__device__ float block_reduce_sum(float val, float* smem) {
    const int lane   = threadIdx.x & 31;
    const int warpid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[warpid] = val;
    __syncthreads();
    const int nwarps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0f;
    if (warpid == 0) val = warp_reduce_sum(val);
    return val;
}

// Combined fwd+bwd: computes per-token cross-entropy and the softmax gradient.
// One block per token position.  Each block handles one row of [V] logits.
// logits:    [BS, V]  bf16
// labels:    [BS]     int32 (-100 = ignore)
// loss_out:  [1]      float (mean loss, atomicAdd accumulation)
// grad:      [BS, V]  bf16  (softmax − one_hot, divided by n_valid)
// n_valid:   [1]      int   (count of non-ignored tokens)
// Shared memory: (blockDim.x/32 + 1) * sizeof(float) — caller provides.
__global__ void k_ce_fwd_bwd(
    const __nv_bfloat16* logits,
    const int32_t*       labels,
    float*               loss_out,
    __nv_bfloat16*       grad,
    int*                 n_valid,
    int V
) {
    extern __shared__ float smem[];  // nwarps floats

    const int row   = blockIdx.x;
    const int label = labels[row];

    // Ignored tokens: zero gradient, no loss contribution.
    if (label < 0) {
        for (int v = threadIdx.x; v < V; v += blockDim.x)
            grad[row * V + v] = __float2bfloat16(0.0f);
        return;
    }

    const __nv_bfloat16* logit_row = logits + row * V;

    // 1. Numerically stable softmax: find global max.
    float local_max = -FLT_MAX;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        float lv = __bfloat162float(logit_row[v]);
        if (lv > local_max) local_max = lv;
    }
    float g_max = block_reduce_max(local_max, smem);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = g_max;
    __syncthreads();

    // 2. Compute exp sum.
    float local_sum = 0.0f;
    for (int v = threadIdx.x; v < V; v += blockDim.x)
        local_sum += expf(__bfloat162float(logit_row[v]) - s_max);
    float g_sum = block_reduce_sum(local_sum, smem);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = g_sum;
    __syncthreads();

    // 3. Write gradient and collect label log-prob.
    // Only one thread will encounter v == label; reduce label_log_prob across block.
    const float inv_sum = 1.0f / s_sum;
    float label_log_prob = 0.0f;

    __nv_bfloat16* grad_row = grad + row * V;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        float prob = expf(__bfloat162float(logit_row[v]) - s_max) * inv_sum;
        float g    = prob - (v == label ? 1.0f : 0.0f);
        grad_row[v] = __float2bfloat16(g);
        if (v == label) label_log_prob = logf(prob + 1e-9f);
    }
    float g_label_log_prob = block_reduce_sum(label_log_prob, smem);

    if (threadIdx.x == 0) {
        atomicAdd(loss_out, -g_label_log_prob);
        atomicAdd(n_valid,  1);
    }
}

// Scale gradient by 1/n_valid after all rows have accumulated.
__global__ void k_scale_grad(
    __nv_bfloat16* grad, int total, int n_valid
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total || n_valid == 0) return;
    float g = __bfloat162float(grad[i]) / static_cast<float>(n_valid);
    grad[i] = __float2bfloat16(g);
}

void cross_entropy_fwd_bwd(
    const __nv_bfloat16* d_logits,
    const int32_t*       d_labels,
    float*               d_loss_out,
    __nv_bfloat16*       d_grad,
    int BS, int V,
    cudaStream_t stream
) {
    // Reset outputs
    cudaMemsetAsync(d_loss_out, 0, sizeof(float), stream);
    int* d_n_valid = nullptr;
    cudaMallocAsync(&d_n_valid, sizeof(int), stream);
    cudaMemsetAsync(d_n_valid, 0, sizeof(int), stream);

    // One block per token, 128 threads each reducing over V=32000.
    // Shared memory: nwarps = 128/32 = 4 floats.
    const int T      = 128;
    const int nwarps = (T + 31) / 32;
    k_ce_fwd_bwd<<<BS, T, nwarps * sizeof(float), stream>>>(
        d_logits, d_labels, d_loss_out, d_grad, d_n_valid, V);

    // Copy n_valid to host to scale loss, then scale grad on device.
    int h_n_valid = 0;
    cudaMemcpyAsync(&h_n_valid, d_n_valid, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (h_n_valid > 0) {
        float h_loss = 0.0f;
        cudaMemcpy(&h_loss, d_loss_out, sizeof(float), cudaMemcpyDeviceToHost);
        h_loss /= static_cast<float>(h_n_valid);
        cudaMemcpy(d_loss_out, &h_loss, sizeof(float), cudaMemcpyHostToDevice);

        const int total = BS * V;
        k_scale_grad<<<(total + 255) / 256, 256, 0, stream>>>(d_grad, total, h_n_valid);
    }

    cudaFreeAsync(d_n_valid, stream);
}

// Softmax over last dim (FP32, in-place).
__global__ void k_softmax_f32(float* x, int cols) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    float* r = x + row * cols;

    float local_max = -FLT_MAX;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        local_max = fmaxf(local_max, r[c]);
    float g_max = block_reduce_max(local_max, smem);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = g_max;
    __syncthreads();

    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        local_sum += expf(r[c] - s_max);
    float g_sum = block_reduce_sum(local_sum, smem);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = g_sum;
    __syncthreads();

    const float inv_sum = 1.0f / s_sum;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        r[c] = expf(r[c] - s_max) * inv_sum;
}

void softmax_inplace_f32(float* d_x, int rows, int cols, cudaStream_t stream) {
    const int T      = std::min(cols, 256);
    const int nwarps = (T + 31) / 32;
    k_softmax_f32<<<rows, T, nwarps * sizeof(float), stream>>>(d_x, cols);
}

// ─── chunked cross-entropy (fused-classifier path) ───────────────────────────
// Count non-ignored labels (accumulates into *d_n_valid; caller zeroes it).
__global__ void k_count_valid(const int32_t* labels, int n, int* n_valid) {
    int local = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        if (labels[i] >= 0) ++local;
    // warp + block reduce
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_xor_sync(0xffffffff, local, off);
    __shared__ int warp_sums[8];
    const int warpid = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) warp_sums[warpid] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        const int nwarps = (blockDim.x + 31) >> 5;
        for (int wq = 0; wq < nwarps; ++wq) total += warp_sums[wq];
        atomicAdd(n_valid, total);
    }
}

// Chunk-safe CE fwd+bwd: like k_ce_fwd_bwd but the gradient and loss are
// scaled by 1/n_valid (read from device) inside the kernel, so chunks can be
// processed independently with no rescaling pass and no host sync.
// grad may alias logits (each element is read before it is written).
__global__ void k_ce_fwd_bwd_scaled(
    const __nv_bfloat16* logits,
    const int32_t*       labels,
    float*               loss_out,
    __nv_bfloat16*       grad,
    const int*           n_valid,
    int V
) {
    extern __shared__ float smem[];

    const int row   = blockIdx.x;
    const int label = labels[row];

    if (label < 0) {
        for (int v = threadIdx.x; v < V; v += blockDim.x)
            grad[row * V + v] = __float2bfloat16(0.0f);
        return;
    }

    const float inv_n = 1.0f / static_cast<float>(max(*n_valid, 1));
    const __nv_bfloat16* logit_row = logits + static_cast<std::size_t>(row) * V;

    float local_max = -FLT_MAX;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        float lv = __bfloat162float(logit_row[v]);
        if (lv > local_max) local_max = lv;
    }
    float g_max = block_reduce_max(local_max, smem);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = g_max;
    __syncthreads();

    float local_sum = 0.0f;
    for (int v = threadIdx.x; v < V; v += blockDim.x)
        local_sum += expf(__bfloat162float(logit_row[v]) - s_max);
    float g_sum = block_reduce_sum(local_sum, smem);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = g_sum;
    __syncthreads();

    const float inv_sum = 1.0f / s_sum;
    float label_log_prob = 0.0f;

    __nv_bfloat16* grad_row = grad + static_cast<std::size_t>(row) * V;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        float prob = expf(__bfloat162float(logit_row[v]) - s_max) * inv_sum;
        float gv   = (prob - (v == label ? 1.0f : 0.0f)) * inv_n;
        if (v == label) label_log_prob = logf(prob + 1e-9f);
        grad_row[v] = __float2bfloat16(gv);
    }
    float g_label_log_prob = block_reduce_sum(label_log_prob, smem);

    if (threadIdx.x == 0)
        atomicAdd(loss_out, -g_label_log_prob * inv_n);
}

void cross_entropy_count_valid(
    const int32_t* d_labels, int n, int* d_n_valid, cudaStream_t stream
) {
    const int T = 256;
    const unsigned blocks = static_cast<unsigned>(std::min((n + T - 1) / T, 256));
    k_count_valid<<<blocks, T, 0, stream>>>(d_labels, n, d_n_valid);
}

void cross_entropy_fwd_bwd_chunk(
    const __nv_bfloat16* d_logits,
    const int32_t*       d_labels,
    float*               d_loss_out,
    __nv_bfloat16*       d_grad,
    int rows, int V,
    const int* d_n_valid,
    cudaStream_t stream
) {
    const int T      = 128;
    const int nwarps = (T + 31) / 32;
    k_ce_fwd_bwd_scaled<<<rows, T, nwarps * sizeof(float), stream>>>(
        d_logits, d_labels, d_loss_out, d_grad, d_n_valid, V);
}

// ─── attention softmax (causal) ──────────────────────────────────────────────
// One block per row.  Row layout: [B, nH, S, S] → row = b*nH*S + h*S + s,
// query position = row % S.  Columns > position are masked to exactly 0.
__global__ void k_causal_softmax_f32(float* x, int S) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const int pos = row % S;
    float* r = x + static_cast<std::size_t>(row) * S;

    float local_max = -FLT_MAX;
    for (int c = threadIdx.x; c <= pos; c += blockDim.x)
        local_max = fmaxf(local_max, r[c]);
    float g_max = block_reduce_max(local_max, smem);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = g_max;
    __syncthreads();

    float local_sum = 0.0f;
    for (int c = threadIdx.x; c <= pos; c += blockDim.x)
        local_sum += expf(r[c] - s_max);
    float g_sum = block_reduce_sum(local_sum, smem);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = g_sum;
    __syncthreads();

    const float inv_sum = 1.0f / s_sum;
    for (int c = threadIdx.x; c < S; c += blockDim.x)
        r[c] = (c <= pos) ? expf(r[c] - s_max) * inv_sum : 0.0f;
}

void attn_causal_softmax_f32(float* d_scores, int rows, int S, cudaStream_t stream) {
    const int T      = std::min(S, 256);
    const int nwarps = (T + 31) / 32;
    k_causal_softmax_f32<<<rows, T, nwarps * sizeof(float), stream>>>(d_scores, S);
}

// Softmax backward: dS = P ⊙ (dP − Σ_j dP_j·P_j) per row.
// P (probabilities, read-only) and dP (overwritten with dS in place).
// Masked columns have P == 0 so dS stays 0 there automatically.
__global__ void k_attn_softmax_bwd(const float* P, float* dP, int S) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const float* p = P  + static_cast<std::size_t>(row) * S;
    float*      dp = dP + static_cast<std::size_t>(row) * S;

    float local = 0.0f;
    for (int c = threadIdx.x; c < S; c += blockDim.x)
        local += p[c] * dp[c];
    float g_dot = block_reduce_sum(local, smem);
    __shared__ float s_dot;
    if (threadIdx.x == 0) s_dot = g_dot;
    __syncthreads();

    for (int c = threadIdx.x; c < S; c += blockDim.x)
        dp[c] = p[c] * (dp[c] - s_dot);
}

void attn_softmax_backward_f32(
    const float* d_probs, float* d_grad, int rows, int S, cudaStream_t stream
) {
    const int T      = std::min(S, 256);
    const int nwarps = (T + 31) / 32;
    k_attn_softmax_bwd<<<rows, T, nwarps * sizeof(float), stream>>>(d_probs, d_grad, S);
}

// ─── global gradient norm: Σ g² accumulated into a device scalar ─────────────
// Grid-stride thread accumulation → warp shuffle → shared memory → atomicAdd.
__global__ void k_sqsum_f32(const float* g, std::size_t n, float* acc) {
    extern __shared__ float smem[];
    float local = 0.0f;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float v = g[i];
        local += v * v;
    }
    float b = block_reduce_sum(local, smem);
    if (threadIdx.x == 0) atomicAdd(acc, b);
}

void sq_sum_acc_f32(const float* d_g, std::size_t n, float* d_acc, cudaStream_t stream) {
    const int T      = 256;
    const int nwarps = T / 32;
    const unsigned blocks = static_cast<unsigned>(
        std::min<std::size_t>((n + T - 1) / T, 1024));
    k_sqsum_f32<<<blocks, T, nwarps * sizeof(float), stream>>>(d_g, n, d_acc);
}

// Cast helpers
__global__ void k_bf16_to_f32(const __nv_bfloat16* src, float* dst, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}
__global__ void k_f32_to_bf16(const float* src, __nv_bfloat16* dst, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

void cast_bf16_to_f32(const __nv_bfloat16* d_src, float* d_dst, std::size_t n, cudaStream_t s) {
    k_bf16_to_f32<<<(n + 255) / 256, 256, 0, s>>>(d_src, d_dst, n);
}
void cast_f32_to_bf16(const float* d_src, __nv_bfloat16* d_dst, std::size_t n, cudaStream_t s) {
    k_f32_to_bf16<<<(n + 255) / 256, 256, 0, s>>>(d_src, d_dst, n);
}

}  // namespace ida_native
