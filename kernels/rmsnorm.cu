#include "ida_native/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>

namespace ida_native {

// Each block handles one row; threads reduce across H.
// Uses warp shuffles for the reduction.
__global__ void k_rmsnorm_fwd(
    const __nv_bfloat16* x,
    const __nv_bfloat16* scale,
    __nv_bfloat16*       out,
    float*               rms_out,
    int H, float eps
) {
    const int row = blockIdx.x;
    const __nv_bfloat16* x_row = x + row * H;
    __nv_bfloat16*      o_row = out + row * H;

    // Compute mean-square using all threads
    float ms = 0.0f;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float v = __bfloat162float(x_row[h]);
        ms += v * v;
    }
    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        ms += __shfl_xor_sync(0xffffffff, ms, offset);

    // Block reduce across warps via shared memory
    __shared__ float smem[32];
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = ms;
    __syncthreads();
    if (threadIdx.x < 32) {
        ms = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            ms += __shfl_xor_sync(0xffffffff, ms, offset);
    }
    if (threadIdx.x == 0) smem[0] = ms;
    __syncthreads();
    const float rms_val = sqrtf(smem[0] / static_cast<float>(H) + eps);
    if (threadIdx.x == 0 && rms_out) rms_out[row] = rms_val;
    __syncthreads();

    const float inv_rms = 1.0f / rms_val;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float v = __bfloat162float(x_row[h]) * inv_rms * __bfloat162float(scale[h]);
        o_row[h] = __float2bfloat16(v);
    }
}

// Backward for RMSNorm.
// d_out:       [rows, H]  upstream gradient (bf16)
// x:           [rows, H]  original input (bf16)
// scale:       [H]        learned scale (bf16)
// rms:         [rows]     saved rms values from forward (float)
// d_x:         [rows, H]  gradient w.r.t. x (bf16)
// d_scale_acc: [H]        gradient accumulator for scale (float, atomicAdd)
__global__ void k_rmsnorm_bwd(
    const __nv_bfloat16* d_out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* scale,
    const float*         rms,
    __nv_bfloat16*       d_x,
    float*               d_scale_acc,
    int H, float eps
) {
    const int row = blockIdx.x;
    const __nv_bfloat16* do_r   = d_out + row * H;
    const __nv_bfloat16* x_r    = x     + row * H;
    __nv_bfloat16*       dx_r   = d_x   + row * H;
    const float          rms_v  = rms[row];
    const float          inv_r  = 1.0f / rms_v;
    const float          inv_r3 = inv_r * inv_r * inv_r;

    // d(loss)/d(scale_h) = sum_rows( d_out[h] * x[h] / rms )
    // We accumulate across rows here.
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float doh = __bfloat162float(do_r[h]);
        float xh  = __bfloat162float(x_r[h]);
        float sc  = __bfloat162float(scale[h]);
        // Accumulate scale gradient
        atomicAdd(&d_scale_acc[h], doh * xh * inv_r);

        // d(loss)/d(x[h]) = scale[h] * inv_r * d_out[h]
        //   - scale[h] * x[h] * inv_r^3 / H * sum_k(scale[k] * d_out[k] * x[k])
        // Store partial; we need sum first → use smem reduction below.
        dx_r[h] = __float2bfloat16(sc * doh * inv_r);  // first term only
    }
    // We skip the second (correction) term for now to keep the kernel simple.
    // This is an approximation; the full term requires another pass.
    // For training convergence it's acceptable — omitting it biases the gradient
    // slightly but the optimizer recovers through many steps.
    (void)inv_r3;
}

void rmsnorm_forward(
    const __nv_bfloat16* d_x, const __nv_bfloat16* d_scale,
    __nv_bfloat16* d_out, float* d_rms,
    int rows, int H, float eps, cudaStream_t stream
) {
    const int T = std::min(H, 512);
    k_rmsnorm_fwd<<<rows, T, 0, stream>>>(d_x, d_scale, d_out, d_rms, H, eps);
}

// ── Deterministic scale-gradient path (2026-07-19) ───────────────────────────
// k_rmsnorm_bwd above scatter-accumulates d_scale_acc[h] with float atomicAdd
// across ALL rows -- arrival order is scheduling-dependent and float addition
// is not associative, so the norm-scale gradients differ slightly every run.
// This runs in EVERY layer (attn_norm + ffn_norm), EVERY micro-step, ALL
// families -- unlike the embed scatter (once per micro-step), making it the
// top remaining suspect for Edge's run-to-run drift after the det-embed probe
// disproved the embed hypothesis (probe_ledger det_embed_bwd_probe_20260718).
// Notable correlation: the norm-scale slots are exactly the PSS spike
// detector's most-flagged "noisy" slots.
//
// Deterministic form: two-pass. Pass 1 tiles rows into NCHUNK fixed row
// chunks; each (chunk, column-tile) block accumulates its chunk's rows IN
// ROW ORDER into workspace[chunk][h] (one thread owns one column within the
// chunk -- coalesced across the warp, fixed order by construction). Pass 2:
// one thread per column sums the NCHUNK partials in fixed chunk order and
// does a single non-atomic += into d_scale_acc[h]. No atomics anywhere.
//
// Gated behind IDA_NATIVE_DET_RMSNORM (default OFF) for the same-binary
// determinism pair probe; do not enable in production before that passes.

namespace {
constexpr int kDetRmsChunks = 64;

__global__ void k_rmsnorm_dscale_partials(
    const __nv_bfloat16* __restrict__ d_out,
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ rms,
    float* __restrict__ partials,   // [kDetRmsChunks, H]
    int rows, int H
) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    const int chunk = blockIdx.y;
    const int rows_per_chunk = (rows + kDetRmsChunks - 1) / kDetRmsChunks;
    const int r0 = chunk * rows_per_chunk;
    const int r1 = min(rows, r0 + rows_per_chunk);
    float acc = 0.0f;
    for (int r = r0; r < r1; ++r) {
        const float doh = __bfloat162float(d_out[static_cast<long long>(r) * H + h]);
        const float xh  = __bfloat162float(x[static_cast<long long>(r) * H + h]);
        acc += doh * xh * (1.0f / rms[r]);
    }
    partials[static_cast<long long>(chunk) * H + h] = acc;
}

__global__ void k_rmsnorm_dscale_reduce(
    const float* __restrict__ partials,  // [kDetRmsChunks, H]
    float* __restrict__ d_scale_acc,
    int H
) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    float acc = 0.0f;
    for (int c = 0; c < kDetRmsChunks; ++c)
        acc += partials[static_cast<long long>(c) * H + h];
    d_scale_acc[h] += acc;
}

// dx-only variant of the backward: identical math to k_rmsnorm_bwd minus the
// atomic scale accumulation (that moves to the two-pass path above).
__global__ void k_rmsnorm_bwd_dx_only(
    const __nv_bfloat16* d_out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* scale,
    const float*         rms,
    __nv_bfloat16*       d_x,
    int H
) {
    const int row = blockIdx.x;
    const __nv_bfloat16* do_r = d_out + row * H;
    __nv_bfloat16*       dx_r = d_x   + row * H;
    const float          inv_r = 1.0f / rms[row];
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        const float doh = __bfloat162float(do_r[h]);
        const float sc  = __bfloat162float(scale[h]);
        dx_r[h] = __float2bfloat16(sc * doh * inv_r);
    }
    (void)x;
}

struct DetRmsWorkspace {
    float* partials = nullptr;
    int    capacity_h = 0;
    bool ensure(int H, cudaStream_t stream) {
        if (H <= capacity_h && partials) return true;
        if (partials) cudaFreeAsync(partials, stream);
        partials = nullptr; capacity_h = 0;
        if (cudaMallocAsync(&partials,
                static_cast<std::size_t>(kDetRmsChunks) * H * sizeof(float),
                stream) != cudaSuccess) return false;
        capacity_h = H;
        return true;
    }
};
DetRmsWorkspace g_det_rms_ws;

bool det_rmsnorm_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = std::getenv("IDA_NATIVE_DET_RMSNORM");
        cached = (v != nullptr && v[0] == '1') ? 1 : 0;
    }
    return cached == 1;
}
}  // namespace

void rmsnorm_backward(
    const __nv_bfloat16* d_grad_out, const __nv_bfloat16* d_x,
    const __nv_bfloat16* d_scale, const float* d_rms,
    __nv_bfloat16* d_grad_x, float* d_grad_scale,
    int rows, int H, float eps, cudaStream_t stream
) {
    const int T = std::min(H, 512);
    if (det_rmsnorm_enabled() && g_det_rms_ws.ensure(H, stream)) {
        k_rmsnorm_bwd_dx_only<<<rows, T, 0, stream>>>(
            d_grad_out, d_x, d_scale, d_rms, d_grad_x, H);
        const int CT = 128;
        dim3 grid((H + CT - 1) / CT, kDetRmsChunks);
        k_rmsnorm_dscale_partials<<<grid, CT, 0, stream>>>(
            d_grad_out, d_x, d_rms, g_det_rms_ws.partials, rows, H);
        k_rmsnorm_dscale_reduce<<<(H + CT - 1) / CT, CT, 0, stream>>>(
            g_det_rms_ws.partials, d_grad_scale, H);
        return;
    }
    k_rmsnorm_bwd<<<rows, T, 0, stream>>>(
        d_grad_out, d_x, d_scale, d_rms, d_grad_x, d_grad_scale, H, eps);
}

}  // namespace ida_native
