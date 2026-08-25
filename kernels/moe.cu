// Cognitive-architecture sparse MoE port (2026-07-21/22). Forward-only
// (Step 3 of the port plan): PressureField -> ConstitutionalRouter ->
// LateralInhibition. See docs/native cognitive-architecture port plan.
//
// Packing-aware pooling: the PyTorch reference pools hidden.mean(dim=1),
// assuming one sample per batch row. Native packs multiple samples per row
// (sb.segs, block-diagonal causal masking, ~7x density) -- pooling per ROW
// instead of per PACKED SAMPLE would blend ~7 unrelated sequences' content
// into one router decision, a severe correctness bug, not an approximation.
// Every kernel below is keyed by segs[pos] (that position's sample START),
// so the router decision is computed once per packed sample and broadcast
// to every position within that sample's span -- matching PyTorch's
// per-sequence semantics exactly, without needing a variable-length
// "sample list" structure: everything stays in dense [B,S,...] shapes,
// using segs[pos] as a scatter/gather key instead of an explicit index.
#include <cstdlib>
#include "ida_native/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ida_native {

// Scalar BF16 atomics are not available on Pascal (sm_61). Keep the BF16
// accumulation contract by updating the containing 32-bit word with CAS; two
// adjacent BF16 values share that word, so the compare/exchange also protects
// the neighboring lane from a torn write. Newer architectures may use a
// native BF16 atomic in a future specialized path, but this fallback is the
// portable CUDA baseline for the P4 validation target.
namespace {
__device__ __forceinline__ void atomic_add_bf16(
    __nv_bfloat16* address, float value
) {
    const std::uintptr_t address_bits = reinterpret_cast<std::uintptr_t>(address);
    auto* word = reinterpret_cast<unsigned int*>(address_bits & ~std::uintptr_t(0x3));
    const unsigned int shift = (address_bits & 0x2) != 0 ? 16u : 0u;
    unsigned int old = *word;
    while (true) {
        const unsigned int assumed = old;
        __nv_bfloat16_raw current_raw{};
        current_raw.x = static_cast<unsigned short>((assumed >> shift) & 0xffffu);
        const float current = __bfloat162float(__nv_bfloat16(current_raw));
        const unsigned short updated_bits =
            __bfloat16_as_ushort(__float2bfloat16(current + value));
        const unsigned int next =
            (assumed & ~(0xffffu << shift)) |
            (static_cast<unsigned int>(updated_bits) << shift);
        old = atomicCAS(word, assumed, next);
        if (old == assumed) return;
    }
}
}  // namespace

// ── Packing-aware mean pooling ───────────────────────────────────────────────

namespace {

__global__ void k_moe_pool_zero(float* accum, float* count, std::size_t n_accum, std::size_t n_count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_accum) accum[i] = 0.0f;
    if (i < n_count) count[i] = 0.0f;
}

// ncu profiling (2026-07-28) found this kernel + its backward counterpart
// consuming ~20%% of GPU time combined, on the same selected-row AI path
// already fixed once this session (k_moe_small_proj_backward_weight). Same
// bug class: grid was only B blocks (32 for mb=32, far under the GPU's 132
// SMs) with each block SERIALLY walking all S positions in a for-loop. The
// serial walk was never actually required for correctness -- every write
// already goes through atomicAdd (both the H-wide accum and the count),
// which tolerates concurrent, out-of-order execution across positions by
// construction. Fixed: one block per (b,s) position (grid = (B,S)), same
// thread-parallel-over-H + atomicAdd body, no in-kernel loop over s at all.
__global__ void k_moe_pool_scatter(
    const __nv_bfloat16* __restrict__ hidden,   // [B,S,H]
    const std::uint16_t* __restrict__ segs,     // [B,S] or nullptr
    float* __restrict__ accum,                  // [B,S,H], pre-zeroed
    float* __restrict__ count,                  // [B,S], pre-zeroed
    int S, int H
) {
    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int start = segs ? static_cast<int>(segs[static_cast<std::size_t>(b) * S + s]) : 0;
    const __nv_bfloat16* x = hidden + (static_cast<std::size_t>(b) * S + s) * H;
    float* acc = accum + (static_cast<std::size_t>(b) * S + start) * H;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        atomicAdd(&acc[h], __bfloat162float(x[h]));
    }
    if (threadIdx.x == 0) {
        atomicAdd(&count[static_cast<std::size_t>(b) * S + start], 1.0f);
    }
}

// One block per (row, position) pair, i.e. grid.x = B*S: divide the
// accumulated sum by its sample's count. Positions that are NOT a sample
// start have count==0 (nothing ever scattered there) and are left as zero
// -- never read, since every real consumer gathers via segs[pos], which
// always resolves to a start position.
__global__ void k_moe_pool_finalize(
    float* __restrict__ accum,        // [B*S, H] in place
    const float* __restrict__ count,  // [B*S]
    int H
) {
    const int row = blockIdx.x;
    const float c = count[row];
    if (c <= 0.0f) return;
    const float inv_c = 1.0f / c;
    float* acc = accum + static_cast<std::size_t>(row) * H;
    for (int h = threadIdx.x; h < H; h += blockDim.x) acc[h] *= inv_c;
}

}  // namespace

void moe_pool_by_sample(
    const __nv_bfloat16* d_hidden, const std::uint16_t* d_segs,
    float* d_pool_scratch, float* d_count_scratch,
    int B, int S, int H, cudaStream_t stream
) {
    const std::size_t n_accum = static_cast<std::size_t>(B) * S * H;
    const std::size_t n_count = static_cast<std::size_t>(B) * S;
    const std::size_t n_max = n_accum > n_count ? n_accum : n_count;
    k_moe_pool_zero<<<static_cast<unsigned>((n_max + 255) / 256), 256, 0, stream>>>(
        d_pool_scratch, d_count_scratch, n_accum, n_count);
    const int T = H < 256 ? H : 256;
    {
        const dim3 grid(static_cast<unsigned>(B), static_cast<unsigned>(S));
        k_moe_pool_scatter<<<grid, T, 0, stream>>>(d_hidden, d_segs, d_pool_scratch, d_count_scratch, S, H);
    }
    k_moe_pool_finalize<<<static_cast<unsigned>(B * S), T, 0, stream>>>(d_pool_scratch, d_count_scratch, H);
}

// ── Broadcast a per-sample-start value back to every position in that sample ─

namespace {
__global__ void k_moe_gather_by_sample(
    const float* __restrict__ start_values,  // [B,S,N], meaningful only at starts
    const std::uint16_t* __restrict__ segs,  // [B,S] or nullptr
    float* __restrict__ out,                 // [B,S,N]
    int S, int N
) {
    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int start = segs ? static_cast<int>(segs[static_cast<std::size_t>(b) * S + s]) : 0;
    const float* src = start_values + (static_cast<std::size_t>(b) * S + start) * N;
    float* dst = out + (static_cast<std::size_t>(b) * S + s) * N;
    for (int n = threadIdx.x; n < N; n += blockDim.x) dst[n] = src[n];
}
}  // namespace

void moe_gather_by_sample(
    const float* d_start_values, const std::uint16_t* d_segs, float* d_out,
    int B, int S, int N, cudaStream_t stream
) {
    dim3 grid(B, S);
    const int T = N < 32 ? N : 32;
    k_moe_gather_by_sample<<<grid, T, 0, stream>>>(d_start_values, d_segs, d_out, S, N);
}

// ── Small dense projection (router/pressure GEMVs; num_routes/num_experts ────
// is tiny, <=11 in every real config, so a plain per-row GEMV avoids the
// tensor-core-oriented bf16 GEMM infra's shape/dtype constraints for what's
// a cheap, small computation relative to the rest of a training step). ──────

namespace {
// One block per output row; each thread strides over output columns and
// reduces over in_dim. out_dim is tiny (<=11) for the routes/experts-facing
// projections, but the pressure->modulation projection projects back OUT to
// the full hidden size (out_dim == H, up to 4096) -- launching with
// blockDim.x == out_dim unconditionally exceeded CUDA's 1024-thread-per-block
// limit for that call and made every real MoE burn's first forward pass an
// illegal kernel launch (cudaErrorInvalidValue on cudaLaunchKernel, found via
// compute-sanitizer 2026-07-23). The stride loop below is correct for both
// the tiny and the full-hidden-size out_dim cases.
__global__ void k_moe_small_proj(
    const float* __restrict__ in,       // [rows, in_dim]
    const __nv_bfloat16* __restrict__ w,  // [out_dim, in_dim] (torch [out,in] convention)
    float* __restrict__ out,            // [rows, out_dim]
    int in_dim, int out_dim
) {
    const int row = blockIdx.x;
    const float* x = in + static_cast<std::size_t>(row) * in_dim;
    for (int n = threadIdx.x; n < out_dim; n += blockDim.x) {
        const __nv_bfloat16* wr = w + static_cast<std::size_t>(n) * in_dim;
        float acc = 0.0f;
        for (int h = 0; h < in_dim; ++h) acc += x[h] * __bfloat162float(wr[h]);
        out[static_cast<std::size_t>(row) * out_dim + n] = acc;
    }
}

// Narrow-output variant (2026-07-28 profiling fix): three of this kernel's
// four real call sites (pressure_proj, router_score, pressure_to_routes)
// have out_dim == num_routes/num_experts, tiny (<=11 in every real config)
// while in_dim is a full hidden size (up to 4096) -- the one-thread-per-
// output-column scheme above starves the warp to out_dim/32 active lanes
// and forces each of those few threads through a fully serial in_dim
// reduction. One warp per output column instead: all 32 lanes stride over
// in_dim together, then a shuffle-reduce combines the partial sums --
// full warp occupancy, in_dim/32 serial work per lane instead of in_dim.
// Only used when out_dim <= 32 (guarantees blockDim.x = 32*out_dim <=
// 1024); the wide case (pressure_mod, out_dim == H) already gets good
// parallelism from the original kernel and is left untouched.
__global__ void k_moe_small_proj_narrow(
    const float* __restrict__ in,
    const __nv_bfloat16* __restrict__ w,
    float* __restrict__ out,
    int in_dim, int out_dim
) {
    const int row = blockIdx.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= out_dim) return;
    const float* x = in + static_cast<std::size_t>(row) * in_dim;
    const __nv_bfloat16* wr = w + static_cast<std::size_t>(warp) * in_dim;
    float acc = 0.0f;
    for (int h = lane; h < in_dim; h += 32) acc += x[h] * __bfloat162float(wr[h]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[static_cast<std::size_t>(row) * out_dim + warp] = acc;
}
}  // namespace

void moe_small_proj_f32(
    const float* d_in, const __nv_bfloat16* d_w, float* d_out,
    int rows, int in_dim, int out_dim, cudaStream_t stream
) {
    if (out_dim <= 32) {
        k_moe_small_proj_narrow<<<rows, out_dim * 32, 0, stream>>>(d_in, d_w, d_out, in_dim, out_dim);
    } else {
        const int T = out_dim < 256 ? out_dim : 256;
        k_moe_small_proj<<<rows, T, 0, stream>>>(d_in, d_w, d_out, in_dim, out_dim);
    }
}

// ── Elementwise activations (float, small tensors) ──────────────────────────

namespace {
__global__ void k_tanh_f32(float* x, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i]);
}
__global__ void k_sigmoid_f32(float* x, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0f / (1.0f + expf(-x[i]));
}
}  // namespace

void tanh_inplace_f32(float* d_x, std::size_t n, cudaStream_t stream) {
    k_tanh_f32<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(d_x, n);
}
void sigmoid_inplace_f32(float* d_x, std::size_t n, cudaStream_t stream) {
    k_sigmoid_f32<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(d_x, n);
}

// ── GPT-2 GELU-new -----------------------------------------------------------
namespace {
constexpr float kGeluNewC = 0.7978845608028654f;
constexpr float kGeluNewA = 0.044715f;

__device__ __forceinline__ float gelu_new_value(float x) {
    const float x2 = x * x;
    const float inner = kGeluNewC * (x + kGeluNewA * x * x2);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float gelu_new_derivative(float x) {
    const float x2 = x * x;
    const float inner = kGeluNewC * (x + kGeluNewA * x * x2);
    const float t = tanhf(inner);
    const float d_inner = kGeluNewC * (1.0f + 3.0f * kGeluNewA * x2);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * d_inner;
}

__global__ void k_gelu_new_forward(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    std::size_t n
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(gelu_new_value(__bfloat162float(x[i])));
}

__global__ void k_gelu_new_backward(
    const __nv_bfloat16* __restrict__ d_out,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ d_x,
    std::size_t n
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float xv = __bfloat162float(x[i]);
        d_x[i] = __float2bfloat16(
            __bfloat162float(d_out[i]) * gelu_new_derivative(xv));
    }
}
}  // namespace

void gelu_new_forward(
    const __nv_bfloat16* d_x,
    __nv_bfloat16* d_out,
    std::size_t n,
    cudaStream_t stream
) {
    k_gelu_new_forward<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(d_x, d_out, n);
}

void gelu_new_backward(
    const __nv_bfloat16* d_grad_out,
    const __nv_bfloat16* d_x,
    __nv_bfloat16* d_grad_x,
    std::size_t n,
    cudaStream_t stream
) {
    k_gelu_new_backward<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        d_grad_out, d_x, d_grad_x, n);
}

// ── QKV bias (2026-08-23) ────────────────────────────────────────────────────
namespace {
__global__ void k_bias_add_strided_bf16(
    __nv_bfloat16* __restrict__ inout, const __nv_bfloat16* __restrict__ bias,
    int rows, int cols, int row_stride, int base
) {
    const int row = blockIdx.x;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        __nv_bfloat16* p = inout + static_cast<std::size_t>(row) * row_stride + base + c;
        *p = __float2bfloat16(__bfloat162float(*p) + __bfloat162float(bias[c]));
    }
}
__global__ void k_col_sum_strided_bf16(
    const __nv_bfloat16* __restrict__ in, float* __restrict__ out_accum,
    int rows, int cols, int row_stride, int base
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    float acc = 0.0f;
    for (int row = 0; row < rows; ++row) {
        acc += __bfloat162float(in[static_cast<std::size_t>(row) * row_stride + base + c]);
    }
    out_accum[c] += acc;
}
}  // namespace

void bias_add_strided_bf16(
    __nv_bfloat16* d_inout, const __nv_bfloat16* d_bias,
    int rows, int cols, int row_stride, int base, cudaStream_t stream
) {
    const int T_raw = ((cols + 31) / 32) * 32;
    const int T = T_raw < 256 ? T_raw : 256;
    k_bias_add_strided_bf16<<<rows, T, 0, stream>>>(d_inout, d_bias, rows, cols, row_stride, base);
}

void col_sum_strided_bf16(
    const __nv_bfloat16* d_in, float* d_out_accum,
    int rows, int cols, int row_stride, int base, cudaStream_t stream
) {
    k_col_sum_strided_bf16<<<static_cast<unsigned>((cols + 255) / 256), 256, 0, stream>>>(
        d_in, d_out_accum, rows, cols, row_stride, base);
}

// ── ConstitutionalRouter: softmax -> hard top-k -> scatter -> renormalize ────
// Mirrors constitutional_router.py's scatter_+renormalize semantics exactly:
// keep the top_k raw softmax probabilities, zero the rest, renormalize the
// kept values to sum to 1. num_experts is tiny (<=11 in every real config)
// so a single-block-per-row selection (no shared-memory sort needed) is
// sufficient -- O(top_k * num_experts) per row, negligible either way.

namespace {
__global__ void k_moe_topk_scatter_renormalize(
    float* __restrict__ scores,  // [rows, num_experts] in: softmax probs, out: sparse route weights
    float* __restrict__ row_sum_save,  // [rows], nullable -- T = sum of kept (pre-renormalize) probs, for backward
    int num_experts, int top_k, bool normalize_topk
) {
    // One thread per row (num_experts is tiny; no need to parallelize within a row).
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    float* s = scores + static_cast<std::size_t>(row) * num_experts;

    bool kept[32];  // num_experts <= 11 in every real config; 32 is a safe ceiling
    for (int e = 0; e < num_experts; ++e) kept[e] = false;

    const int k = top_k < num_experts ? top_k : num_experts;
    for (int pick = 0; pick < k; ++pick) {
        int best = -1;
        float best_v = -1.0f;
        for (int e = 0; e < num_experts; ++e) {
            if (!kept[e] && s[e] > best_v) { best_v = s[e]; best = e; }
        }
        if (best < 0) break;
        kept[best] = true;
    }

    float sum = 0.0f;
    for (int e = 0; e < num_experts; ++e) {
        if (kept[e]) sum += s[e];
    }
    if (row_sum_save) row_sum_save[row] = sum;
    const float inv_sum = normalize_topk && sum > 0.0f ? (1.0f / sum) : 1.0f;
    for (int e = 0; e < num_experts; ++e) {
        s[e] = kept[e] ? s[e] * inv_sum : 0.0f;
    }
}
}  // namespace

void moe_topk_route_f32(
    float* d_scores, int rows, int num_experts, int top_k, cudaStream_t stream,
    float* d_row_sum_save, float* d_scores_save, bool normalize_topk
) {
    softmax_inplace_f32(d_scores, rows, num_experts, stream);
    if (d_scores_save) {
        cudaMemcpyAsync(d_scores_save, d_scores,
            static_cast<std::size_t>(rows) * num_experts * sizeof(float),
            cudaMemcpyDeviceToDevice, stream);
    }
    const int T = 128;
    k_moe_topk_scatter_renormalize<<<(rows + T - 1) / T, T, 0, stream>>>(
        d_scores, d_row_sum_save, num_experts, top_k, normalize_topk);
}

// ── LateralInhibition: dominance-cap ceiling + minority-floor + renormalize ──
// No learnable parameters (matches the reference: lateral_inhibition.py has
// none). Clamps any weight above dominance_cap down to it, raises any
// nonzero weight below minority_floor up to it, then renormalizes to sum 1.

namespace {
__global__ void k_moe_lateral_inhibition(
    float* __restrict__ w,
    float* __restrict__ c1_save,      // [rows,num_experts], nullable -- clamp(p,max=cap), pre-floor
    float* __restrict__ row_sum_save, // [rows], nullable -- S = sum(bounded), pre-final-renormalize
    int num_experts, float dominance_cap, float minority_floor
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    float* s = w + static_cast<std::size_t>(row) * num_experts;
    float* c1r = c1_save ? c1_save + static_cast<std::size_t>(row) * num_experts : nullptr;
    float sum = 0.0f;
    for (int e = 0; e < num_experts; ++e) {
        float v = s[e];
        if (v > dominance_cap) v = dominance_cap;  // max-clamp always applied first
        if (c1r) c1r[e] = v;                       // saved BEFORE the min-floor-raise
        if (v > 0.0f && v < minority_floor) v = minority_floor;
        s[e] = v;
        sum += v;
    }
    if (row_sum_save) row_sum_save[row] = sum;
    const float inv_sum = sum > 0.0f ? (1.0f / sum) : 0.0f;
    for (int e = 0; e < num_experts; ++e) s[e] *= inv_sum;
}
}  // namespace

void moe_lateral_inhibition_f32(
    float* d_weights, int rows, int num_experts,
    float dominance_cap, float minority_floor, cudaStream_t stream,
    float* d_c1_save, float* d_row_sum_save
) {
    const int T = 128;
    k_moe_lateral_inhibition<<<(rows + T - 1) / T, T, 0, stream>>>(
        d_weights, d_c1_save, d_row_sum_save, num_experts, dominance_cap, minority_floor);
}

// ── PressureField's modulation gate: hidden_out = hidden * sigmoid(mod(pressure)) ─
// modulation is [B,S,H] (already gathered/broadcast per-sample by the
// caller, matching PyTorch's per-sequence modulation.unsqueeze(1)); this
// just does the elementwise bf16*float multiply, writing bf16 out.

namespace {
__global__ void k_moe_modulate(
    const __nv_bfloat16* __restrict__ hidden, const float* __restrict__ modulation,
    __nv_bfloat16* __restrict__ out, std::size_t n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__bfloat162float(hidden[i]) * modulation[i]);
}
}  // namespace

void moe_modulate_hidden_f32(
    const __nv_bfloat16* d_hidden, const float* d_modulation, __nv_bfloat16* d_out,
    std::size_t n, cudaStream_t stream
) {
    k_moe_modulate<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(d_hidden, d_modulation, d_out, n);
}

// ── Elementwise add: logits = router_score_out + routed_pressure ────────────
// Covers both ConstitutionalRouter branches: when pressure_to_routes is
// nn.Identity() (num_routes==num_experts), the caller passes `pressure`
// itself as d_b; when it's a real projection, the caller passes that
// projection's output. Either way this is just an elementwise add.

namespace {
__global__ void k_add_f32(const float* a, const float* b, float* out, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
}  // namespace

void add_inplace_f32(float* d_a, const float* d_b, std::size_t n, cudaStream_t stream) {
    k_add_f32<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(d_a, d_b, d_a, n);
}

// ── GELU (exact-erf variant, matching nn.GELU()'s default — NOT the tanh ────
// approximation; swiglu's SiLU gate is not a substitute, this is a real
// numerical divergence point if approximated). CognitiveCircuitMLP forward
// only for Step 4; backward (Step 5) needs its own kernel + saved pre-
// activation input.

namespace {
__global__ void k_gelu_fwd(const __nv_bfloat16* x, __nv_bfloat16* out, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __bfloat162float(x[i]);
    const float g = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));  // 1/sqrt(2)
    out[i] = __float2bfloat16(g);
}
}  // namespace

void gelu_forward(const __nv_bfloat16* d_x, __nv_bfloat16* d_out, std::size_t n, cudaStream_t stream) {
    k_gelu_fwd<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(d_x, d_out, n);
}

// ── Expert-output accumulation: mixed[row,:] += weight[row] * term[row,:] ───
// weight is one column of the [rows, num_experts] route-weight tensor
// (stride num_experts between rows) -- the per-row scalar gate for one
// expert's contribution to the FFN-residual mixture.

namespace {
__global__ void k_moe_scale_accumulate(
    __nv_bfloat16* __restrict__ mixed,        // [rows, H], accumulated in place
    const __nv_bfloat16* __restrict__ term,   // [rows, H]
    const float* __restrict__ route_weights,  // [rows, num_experts]
    int expert_idx, int num_experts, int H
) {
    const int row = blockIdx.x;
    const float w = route_weights[static_cast<std::size_t>(row) * num_experts + expert_idx];
    __nv_bfloat16* m = mixed + static_cast<std::size_t>(row) * H;
    const __nv_bfloat16* t = term + static_cast<std::size_t>(row) * H;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        m[h] = __float2bfloat16(__bfloat162float(m[h]) + w * __bfloat162float(t[h]));
    }
}
__global__ void k_bf16_zero(__nv_bfloat16* x, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2bfloat16(0.0f);
}
__global__ void k_bf16_copy(const __nv_bfloat16* src, __nv_bfloat16* dst, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
__global__ void k_moe_copy_expert_fcin_slice_bf16(
    const __nv_bfloat16* __restrict__ all_expert_fcin,
    __nv_bfloat16* __restrict__ expert_fcin,
    int num_experts, int expert_width, int expert_idx, std::size_t n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int col = static_cast<int>(i % expert_width);
    const int row = static_cast<int>(i / expert_width);
    const std::size_t src = static_cast<std::size_t>(row) * num_experts * expert_width
                          + static_cast<std::size_t>(expert_idx) * expert_width
                          + col;
    expert_fcin[i] = all_expert_fcin[src];
}
__global__ void k_moe_build_expert_row_lists(
    const float* __restrict__ route_weights,
    int* __restrict__ counts,
    int* __restrict__ row_indices,
    int rows, int num_experts
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* rw = route_weights + static_cast<std::size_t>(row) * num_experts;
    for (int e = 0; e < num_experts; ++e) {
        if (rw[e] > 0.0f) {
            const int slot = atomicAdd(counts + e, 1);
            row_indices[static_cast<std::size_t>(e) * rows + slot] = row;
        }
    }
}
__global__ void k_moe_gather_rows_bf16(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    const int* __restrict__ row_indices,
    int width, std::size_t n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int col = static_cast<int>(i % width);
    const int compact_row = static_cast<int>(i / width);
    const int src_row = row_indices[compact_row];
    dst[i] = src[static_cast<std::size_t>(src_row) * width + col];
}
__global__ void k_moe_gather_rows_u8(
    const std::uint8_t* __restrict__ src,
    std::uint8_t* __restrict__ dst,
    const int* __restrict__ row_indices,
    int width, std::size_t n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int col = static_cast<int>(i % width);
    const int compact_row = static_cast<int>(i / width);
    const int src_row = row_indices[compact_row];
    dst[i] = src[static_cast<std::size_t>(src_row) * width + col];
}
__global__ void k_moe_scatter_accumulate_rows_bf16(
    __nv_bfloat16* __restrict__ mixed,
    const __nv_bfloat16* __restrict__ compact,
    const float* __restrict__ route_weights,
    const int* __restrict__ row_indices,
    int expert_idx, int num_experts, int H, std::size_t n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int h = static_cast<int>(i % H);
    const int compact_row = static_cast<int>(i / H);
    const int row = row_indices[compact_row];
    const float w = route_weights[static_cast<std::size_t>(row) * num_experts + expert_idx];
    __nv_bfloat16* dst = mixed + static_cast<std::size_t>(row) * H + h;
    *dst = __float2bfloat16(__bfloat162float(*dst) + w * __bfloat162float(compact[i]));
}
__global__ void k_moe_row_dot_selected_bf16(
    const __nv_bfloat16* __restrict__ a_full,
    const __nv_bfloat16* __restrict__ b_compact,
    float* __restrict__ out,
    const int* __restrict__ row_indices,
    int H, int out_stride, int out_col
) {
    const int compact_row = blockIdx.x;
    const int row = row_indices[compact_row];
    float sum = 0.0f;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        sum += __bfloat162float(a_full[static_cast<std::size_t>(row) * H + h]) *
               __bfloat162float(b_compact[static_cast<std::size_t>(compact_row) * H + h]);
    }
    __shared__ float sh[256];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[static_cast<std::size_t>(row) * out_stride + out_col] = sh[0];
}
__global__ void k_moe_gather_scale_rows_bf16(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    const float* __restrict__ route_weights,
    const int* __restrict__ row_indices,
    int expert_idx, int num_experts, int H, std::size_t n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int h = static_cast<int>(i % H);
    const int compact_row = static_cast<int>(i / H);
    const int row = row_indices[compact_row];
    const float w = route_weights[static_cast<std::size_t>(row) * num_experts + expert_idx];
    dst[i] = __float2bfloat16(w * __bfloat162float(src[static_cast<std::size_t>(row) * H + h]));
}
__global__ void k_moe_scatter_add_rows_bf16(
    __nv_bfloat16* __restrict__ dst_full,
    const __nv_bfloat16* __restrict__ src_compact,
    const int* __restrict__ row_indices,
    int H, std::size_t n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int h = static_cast<int>(i % H);
    const int compact_row = static_cast<int>(i / H);
    const int row = row_indices[compact_row];
    __nv_bfloat16* dst = dst_full + static_cast<std::size_t>(row) * H + h;
    *dst = __float2bfloat16(__bfloat162float(*dst) + __bfloat162float(src_compact[i]));
}
}  // namespace

void moe_scale_accumulate_bf16(
    __nv_bfloat16* d_mixed, const __nv_bfloat16* d_term, const float* d_route_weights,
    int rows, int expert_idx, int num_experts, int H, cudaStream_t stream
) {
    const int T = H < 256 ? H : 256;
    k_moe_scale_accumulate<<<rows, T, 0, stream>>>(d_mixed, d_term, d_route_weights, expert_idx, num_experts, H);
}

void bf16_zero(__nv_bfloat16* d_x, std::size_t n, cudaStream_t stream) {
    k_bf16_zero<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(d_x, n);
}
void bf16_copy(const __nv_bfloat16* d_src, __nv_bfloat16* d_dst, std::size_t n, cudaStream_t stream) {
    k_bf16_copy<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(d_src, d_dst, n);
}

void moe_copy_expert_fcin_slice_bf16(
    const __nv_bfloat16* d_all_expert_fcin, __nv_bfloat16* d_expert_fcin,
    int rows, int num_experts, int expert_width, int expert_idx,
    cudaStream_t stream
) {
    const std::size_t n = static_cast<std::size_t>(rows) * expert_width;
    k_moe_copy_expert_fcin_slice_bf16<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        d_all_expert_fcin, d_expert_fcin, num_experts, expert_width, expert_idx, n);
}

void moe_build_expert_row_lists(
    const float* d_route_weights, int* d_counts, int* d_row_indices,
    int rows, int num_experts, cudaStream_t stream
) {
    cudaMemsetAsync(d_counts, 0, static_cast<std::size_t>(num_experts) * sizeof(int), stream);
    const int T = 128;
    k_moe_build_expert_row_lists<<<(rows + T - 1) / T, T, 0, stream>>>(
        d_route_weights, d_counts, d_row_indices, rows, num_experts);
}

void moe_gather_rows_bf16(
    const __nv_bfloat16* d_src, __nv_bfloat16* d_dst,
    const int* d_row_indices, int n_rows, int width, cudaStream_t stream
) {
    const std::size_t n = static_cast<std::size_t>(n_rows) * width;
    k_moe_gather_rows_bf16<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        d_src, d_dst, d_row_indices, width, n);
}

void moe_gather_rows_u8(
    const std::uint8_t* d_src, std::uint8_t* d_dst,
    const int* d_row_indices, int n_rows, int width, cudaStream_t stream
) {
    const std::size_t n = static_cast<std::size_t>(n_rows) * width;
    k_moe_gather_rows_u8<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        d_src, d_dst, d_row_indices, width, n);
}

void moe_scatter_accumulate_rows_bf16(
    __nv_bfloat16* d_mixed, const __nv_bfloat16* d_compact,
    const float* d_route_weights, const int* d_row_indices,
    int n_rows, int expert_idx, int num_experts, int H, cudaStream_t stream
) {
    const std::size_t n = static_cast<std::size_t>(n_rows) * H;
    k_moe_scatter_accumulate_rows_bf16<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        d_mixed, d_compact, d_route_weights, d_row_indices, expert_idx, num_experts, H, n);
}

void moe_row_dot_selected_bf16(
    const __nv_bfloat16* d_a_full, const __nv_bfloat16* d_b_compact,
    float* d_out, const int* d_row_indices,
    int n_rows, int H, int out_stride, int out_col, cudaStream_t stream
) {
    const int T = H < 256 ? H : 256;
    k_moe_row_dot_selected_bf16<<<n_rows, T, 0, stream>>>(
        d_a_full, d_b_compact, d_out, d_row_indices, H, out_stride, out_col);
}

void moe_gather_scale_rows_bf16(
    const __nv_bfloat16* d_src, __nv_bfloat16* d_dst,
    const float* d_route_weights, const int* d_row_indices,
    int n_rows, int expert_idx, int num_experts, int H, cudaStream_t stream
) {
    const std::size_t n = static_cast<std::size_t>(n_rows) * H;
    k_moe_gather_scale_rows_bf16<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        d_src, d_dst, d_route_weights, d_row_indices, expert_idx, num_experts, H, n);
}

void moe_scatter_add_rows_bf16(
    __nv_bfloat16* d_dst_full, const __nv_bfloat16* d_src_compact,
    const int* d_row_indices, int n_rows, int H, cudaStream_t stream
) {
    const std::size_t n = static_cast<std::size_t>(n_rows) * H;
    k_moe_scatter_add_rows_bf16<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        d_dst_full, d_src_compact, d_row_indices, H, n);
}

// ═══════════════════════════════════════════════════════════════════════════
// Grouped/device-side selected-expert scheduler.
//
// The selected-row ablation above still copies per-expert active row counts
// back to the host so the host can issue variable-M cuBLAS GEMMs. That D2H
// copy and the host loop are the remaining synchronization point. The
// scheduler below keeps counts and offsets on the device: a prefix sum over
// the device-side counts gives each expert's offset in the global selected-
// row ordering, and a single kernel launch over (top_k * rows) selected rows
// reads its expert/row from device memory and performs the fc_in -> GELU ->
// fc_out -> scale -> accumulate work without any host-visible count.
//
// This is currently a BF16 correctness path; an FP8/WGMMA variant is needed
// to match the FP8 cuBLAS throughput path.
// ═══════════════════════════════════════════════════════════════════════════
namespace {
__global__ void k_moe_expert_offsets_from_counts(
    const int* __restrict__ counts,
    int* __restrict__ offsets,
    int num_experts
) {
    int sum = 0;
    for (int e = 0; e < num_experts; ++e) {
        offsets[e] = sum;
        sum += counts[e];
    }
}

// One block per selected row. Each block finds its expert and local row index
// from the device-side offsets/counts, loads the source row, computes the
// expert fc_in * GELU * fc_out, scales by the route weight, and atomically
// accumulates into mixed[row, h].
__global__ void k_moe_grouped_expert_forward_bf16(
    const __nv_bfloat16* __restrict__ hidden,
    const float* __restrict__ route_weights,
    const int* __restrict__ counts,
    const int* __restrict__ offsets,
    const int* __restrict__ row_indices,
    const __nv_bfloat16* __restrict__ fc_in_w,
    const __nv_bfloat16* __restrict__ fc_out_w,
    __nv_bfloat16* __restrict__ mixed,
    int rows, int H, int Ie, int num_experts
) {
    const int gid = blockIdx.x;
    int e = 0;
    for (; e < num_experts; ++e) {
        const int off = offsets[e];
        if (gid >= off && gid < off + counts[e]) break;
    }
    if (e >= num_experts) return;

    const int lid = gid - offsets[e];
    const int row = row_indices[static_cast<std::size_t>(e) * rows + lid];
    const float w = route_weights[static_cast<std::size_t>(row) * num_experts + e];

    extern __shared__ __nv_bfloat16 sh[];
    __nv_bfloat16* const sh_hidden = sh;
    __nv_bfloat16* const sh_fcin = sh + H;

    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        sh_hidden[h] = hidden[static_cast<std::size_t>(row) * H + h];
    }
    __syncthreads();

    const __nv_bfloat16* w_in = fc_in_w + static_cast<std::size_t>(e) * H * Ie;
    for (int j = threadIdx.x; j < Ie; j += blockDim.x) {
        float sum = 0.0f;
        const std::size_t w_col_off = static_cast<std::size_t>(j) * H;
        for (int h = 0; h < H; ++h) {
            sum += __bfloat162float(sh_hidden[h]) * __bfloat162float(w_in[w_col_off + h]);
        }
        const float v = sum;
        const float g = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
        sh_fcin[j] = __float2bfloat16(g);
    }
    __syncthreads();

    const __nv_bfloat16* w_out = fc_out_w + static_cast<std::size_t>(e) * Ie * H;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float sum = 0.0f;
        const std::size_t w_col_off = static_cast<std::size_t>(h) * Ie;
        for (int j = 0; j < Ie; ++j) {
            sum += __bfloat162float(sh_fcin[j]) * __bfloat162float(w_out[w_col_off + j]);
        }
        const std::size_t dst_idx = static_cast<std::size_t>(row) * H + h;
        atomic_add_bf16(&mixed[dst_idx], w * sum);
    }
}
}  // namespace

void moe_expert_offsets_from_counts(
    const int* d_counts, int* d_offsets, int num_experts, cudaStream_t stream
) {
    k_moe_expert_offsets_from_counts<<<1, 1, 0, stream>>>(d_counts, d_offsets, num_experts);
}

void moe_grouped_expert_forward_bf16(
    const __nv_bfloat16* d_hidden, const float* d_route_weights,
    const int* d_counts, const int* d_offsets, const int* d_row_indices,
    const __nv_bfloat16* d_fc_in_w, const __nv_bfloat16* d_fc_out_w,
    __nv_bfloat16* d_mixed,
    int rows, int H, int Ie, int num_experts, int total_selected_rows,
    cudaStream_t stream
) {
    const int threads = 256;
    const int smem_bytes = static_cast<int>(sizeof(__nv_bfloat16)) * (H + Ie);
    k_moe_grouped_expert_forward_bf16<<<total_selected_rows, threads, smem_bytes, stream>>>(
        d_hidden, d_route_weights, d_counts, d_offsets, d_row_indices,
        d_fc_in_w, d_fc_out_w, d_mixed, rows, H, Ie, num_experts);
}

// ═══════════════════════════════════════════════════════════════════════════
// FP8 grouped/device-side selected-expert forward. Same device-side offset/
// count scheduling as the BF16 kernel above (moe_expert_offsets_from_counts,
// one block per selected row, host-known total_selected_rows = BS*top_k, no
// host sync). Reads the SAME per-layer FP8 weight slots lt_gemm_fp8_nt
// already uses for the dense/variable-M path (w_in.fwd8/w_out.fwd8 -- both
// fixed E4M3, weights never calibrate to E5M2, only activations do) and the
// SAME fcin activation buffer/type/descale moe_expert_bank_forward already
// resolves upstream for both the fp4n-decoded and plain-fp8-quantized cases
// -- this kernel does not branch on fp4n at all, it just reads whatever
// uniform E4M3/E5M2 pointer it's handed.
//
// Scoped precision decision: the fc_in leg reads the real quantized (E4M3 or
// E5M2, per fcin_is_e5m2) activation input, matching lt_gemm_fp8_nt's actual
// production input exactly. The GELU intermediate between fc_in and fc_out
// stays float in shared memory and is NOT re-quantized to FP8 before the
// fc_out leg -- the existing per-expert host loop's fcout leg uses a
// persistent per-layer ActSlot delayed-scaling calibration (fp8_quant_act)
// with stateful side effects (advances amax history every call); doing that
// correctly inside a single fused per-row kernel is real added scope,
// deliberately deferred. This is a documented, one-directional
// simplification -- it never degrades the input precision below the FP8
// baseline's, only raises the intermediate's -- and must be accounted for
// with the same tolerance the BF16 correctness slice used, not a tighter one.
// Stage G3, tiled replacement for the Stage G1 one-block-per-row kernel
// above: the per-row design re-reads the ENTIRE expert weight matrix from
// global memory independently for every selected row, even when many rows
// share the same expert -- that redundant weight traffic, not scalar-vs-
// tensor-core math, is why the Stage G1 kernel measured 5-15x slower than
// the existing variable-M cuBLASLt path. Fix: each block now handles up to
// rows_per_block selected rows belonging to ONE expert (never spans an
// expert boundary) and streams the weight matrix through shared memory in
// H-chunks, reusing every weight element across all rows in the tile before
// moving to the next chunk -- weight global-memory traffic drops roughly
// rows_per_block-fold versus Stage G1, independent of tensor-core usage.
//
// Grid size is num_experts * ceil(rows / rows_per_block), and rows_per_block
// is derived purely from Ie's shared-memory footprint (computed host-side,
// same formula every call for a given layer shape) -- NOT from any
// per-step dispatch count, so this keeps the same no-host-sync property
// Stage G1 established. Blocks whose tile index falls past an expert's
// actual selected-row count for this step exit immediately (cheap).
//
// fc_in's per-row accumulators live in registers across the whole H
// K-chunk loop (kMoeTiledJCap output columns/thread x kMoeTiledRowsCap
// rows), matching the pattern lt_gemm_fp8_nt's own dW GEMM uses for
// accumulation. fc_out's K dimension (Ie) needs no chunked reload at all --
// its input is fc_in's already-fully-resident GELU output in sh_fcin, so
// fc_out just indexes it directly. Same scoped precision decision as Stage
// G1 (GELU intermediate stays float, never re-quantized -- see the Stage G1
// note below, still accurate) and same weight source (w_in.fwd8/w_out.fwd8,
// the real E4M3 buffers lt_gemm_fp8_nt already reads for the dense path).
// Register budget note (found empirically after the first tiled build):
// per-thread accum register count is kMoeTiledJCap * kRowsCap (compile-time,
// independent of the actual runtime row count in a tile), and thread count
// scales with max(Ie,H)/kMoeTiledJCap -- so a WIDE layer (e.g. MoE's
// Ie=H=4096) launches MORE threads, each ALSO carrying the same fixed
// per-thread accum count, and total block register footprint (threads x
// accum) scaled with kRowsCap=16 landed at ~65536 registers/block for that
// shape -- the entire SM's register file for ONE block, forcing spills that
// ate most of the tiling win the first measurement showed (only 2-3x over
// the naive kernel despite a nominal 16x weight-reuse factor). Fix: kRowsCap
// is now a template parameter with two instantiations, chosen host-side by
// how wide the layer is, so a wide layer gets a smaller row-tile (less
// reuse, but fits in registers) and a narrow layer (AI's Ie=1024) still
// gets the larger, more profitable tile.
namespace {
constexpr int kMoeTiledJCap = 8;
constexpr int kMoeTiledKChunk = 128;

template <int kRowsCap>
__global__ void k_moe_grouped_expert_forward_fp8_tiled(
    const void* __restrict__ hidden, const float* __restrict__ fcin_descale, int fcin_is_e5m2,
    const float* __restrict__ route_weights,
    const int* __restrict__ counts,
    const int* __restrict__ row_indices,
    const void* __restrict__ fc_in_w, const float* __restrict__ fc_in_descale,
    const void* __restrict__ fc_out_w, const float* __restrict__ fc_out_descale,
    __nv_bfloat16* __restrict__ mixed,
    int rows, int H, int Ie, int num_experts,
    int rows_per_block, int tiles_per_expert
) {
    const int e = blockIdx.x / tiles_per_expert;
    const int tile_idx = blockIdx.x % tiles_per_expert;
    const int local_start = tile_idx * rows_per_block;
    const int n_active = counts[e];
    if (local_start >= n_active) return;
    const int n_rows = min(rows_per_block, n_active - local_start);

    const int* my_row_indices = row_indices + static_cast<std::size_t>(e) * rows + local_start;
    const float in_descale = *fcin_descale;
    const float win_descale = *fc_in_descale;
    const float wout_descale = *fc_out_descale;

    extern __shared__ float sh_tiled[];
    float* const sh_chunk = sh_tiled;                                     // [rows_per_block][kMoeTiledKChunk]
    float* const sh_fcin = sh_tiled + rows_per_block * kMoeTiledKChunk;    // [rows_per_block][Ie]

    const auto* hidden_e4m3 = static_cast<const __nv_fp8_e4m3*>(hidden);
    const auto* hidden_e5m2 = static_cast<const __nv_fp8_e5m2*>(hidden);
    const __nv_fp8_e4m3* w_in = static_cast<const __nv_fp8_e4m3*>(fc_in_w) + static_cast<std::size_t>(e) * H * Ie;
    const __nv_fp8_e4m3* w_out = static_cast<const __nv_fp8_e4m3*>(fc_out_w) + static_cast<std::size_t>(e) * Ie * H;

    // ── fc_in: H -> Ie, K-chunked over H. Each weight element w_in[j][k] is
    // read once per chunk and reused against all n_rows rows already staged
    // in shared memory, instead of once per row as Stage G1 did. ──
    {
        float accum[kMoeTiledJCap][kRowsCap];
        for (int jj = 0; jj < kMoeTiledJCap; ++jj)
            for (int r = 0; r < kRowsCap; ++r) accum[jj][r] = 0.0f;

        for (int k0 = 0; k0 < H; k0 += kMoeTiledKChunk) {
            const int kt = min(kMoeTiledKChunk, H - k0);
            for (int idx = threadIdx.x; idx < n_rows * kt; idx += blockDim.x) {
                const int r = idx / kt;
                const int k = idx % kt;
                const int row = my_row_indices[r];
                const std::size_t gidx = static_cast<std::size_t>(row) * H + k0 + k;
                const float raw = fcin_is_e5m2 ? float(hidden_e5m2[gidx]) : float(hidden_e4m3[gidx]);
                sh_chunk[r * kMoeTiledKChunk + k] = raw * in_descale;
            }
            __syncthreads();

            for (int jj = 0; jj < kMoeTiledJCap; ++jj) {
                const int j = threadIdx.x + jj * blockDim.x;
                if (j >= Ie) break;
                const std::size_t w_row_off = static_cast<std::size_t>(j) * H + k0;
                for (int k = 0; k < kt; ++k) {
                    const float wv = float(w_in[w_row_off + k]) * win_descale;
                    for (int r = 0; r < n_rows; ++r) {
                        accum[jj][r] += sh_chunk[r * kMoeTiledKChunk + k] * wv;
                    }
                }
            }
            __syncthreads();
        }

        for (int jj = 0; jj < kMoeTiledJCap; ++jj) {
            const int j = threadIdx.x + jj * blockDim.x;
            if (j >= Ie) break;
            for (int r = 0; r < n_rows; ++r) {
                const float v = accum[jj][r];
                const float g = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
                sh_fcin[r * Ie + j] = g;
            }
        }
        __syncthreads();
    }

    // ── fc_out: Ie -> H. sh_fcin already holds this tile's full GELU output
    // for every row (no chunked reload -- it's fully resident), so this leg
    // reads it directly while streaming w_out the same reused-across-rows way. ──
    {
        float accum[kMoeTiledJCap][kRowsCap];
        for (int jj = 0; jj < kMoeTiledJCap; ++jj)
            for (int r = 0; r < kRowsCap; ++r) accum[jj][r] = 0.0f;

        for (int jj = 0; jj < kMoeTiledJCap; ++jj) {
            const int h = threadIdx.x + jj * blockDim.x;
            if (h >= H) break;
            const std::size_t w_row_off = static_cast<std::size_t>(h) * Ie;
            for (int k = 0; k < Ie; ++k) {
                const float wv = float(w_out[w_row_off + k]) * wout_descale;
                for (int r = 0; r < n_rows; ++r) {
                    accum[jj][r] += sh_fcin[r * Ie + k] * wv;
                }
            }
        }

        for (int jj = 0; jj < kMoeTiledJCap; ++jj) {
            const int h = threadIdx.x + jj * blockDim.x;
            if (h >= H) break;
            for (int r = 0; r < n_rows; ++r) {
                const int row = my_row_indices[r];
                const float rw = route_weights[static_cast<std::size_t>(row) * num_experts + e];
                const std::size_t dst_idx = static_cast<std::size_t>(row) * H + h;
                atomic_add_bf16(&mixed[dst_idx], rw * accum[jj][r]);
            }
        }
    }
}
}  // namespace

void moe_grouped_expert_forward_fp8(
    const void* d_hidden, const float* d_fcin_descale, int fcin_is_e5m2,
    const float* d_route_weights,
    const int* d_counts, const int* /*d_offsets, unused by the tiled kernel*/, const int* d_row_indices,
    const void* d_fc_in_w, const float* d_fc_in_descale,
    const void* d_fc_out_w, const float* d_fc_out_descale,
    __nv_bfloat16* d_mixed,
    int rows, int H, int Ie, int num_experts, int /*total_selected_rows, unused by the tiled kernel*/,
    cudaStream_t stream
) {
    // Wide-layer instantiation gets a smaller row-tile (register-safe);
    // narrow layers get the bigger, more profitable one. Threshold chosen
    // so kRowsCap=4's block footprint (threads * kMoeTiledJCap * 4) and
    // kRowsCap=16's (threads * kMoeTiledJCap * 16) both stay well under the
    // 65536-register/SM budget for the two production shapes this was
    // measured against (MoE Ie=H=4096, AI Ie=1024/H=2048).
    constexpr int kRowsCapWide = 8;
    constexpr int kRowsCapNarrow = 16;
    constexpr int kSharedBudgetBytes = 160 * 1024;
    const int widest = Ie > H ? Ie : H;
    const bool use_wide = widest > 2048;
    const int rows_cap = use_wide ? kRowsCapWide : kRowsCapNarrow;

    int rows_per_block = kSharedBudgetBytes / (static_cast<int>(sizeof(float)) * (kMoeTiledKChunk + Ie));
    if (rows_per_block > rows_cap) rows_per_block = rows_cap;
    if (rows_per_block < 1) rows_per_block = 1;
    int threads = ((widest + kMoeTiledJCap - 1) / kMoeTiledJCap + 31) / 32 * 32;
    if (threads < 32) threads = 32;
    if (threads > 1024) threads = 1024;
    const int tiles_per_expert = (rows + rows_per_block - 1) / rows_per_block;
    const int smem_bytes = static_cast<int>(sizeof(float)) *
        rows_per_block * (kMoeTiledKChunk + Ie);

    static bool attr_configured_wide = false;
    static bool attr_configured_narrow = false;
    if (use_wide) {
        if (!attr_configured_wide) {
            cudaFuncSetAttribute(
                k_moe_grouped_expert_forward_fp8_tiled<kRowsCapWide>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kSharedBudgetBytes);
            attr_configured_wide = true;
        }
        k_moe_grouped_expert_forward_fp8_tiled<kRowsCapWide>
            <<<num_experts * tiles_per_expert, threads, smem_bytes, stream>>>(
                d_hidden, d_fcin_descale, fcin_is_e5m2, d_route_weights,
                d_counts, d_row_indices,
                d_fc_in_w, d_fc_in_descale, d_fc_out_w, d_fc_out_descale,
                d_mixed, rows, H, Ie, num_experts, rows_per_block, tiles_per_expert);
    } else {
        if (!attr_configured_narrow) {
            cudaFuncSetAttribute(
                k_moe_grouped_expert_forward_fp8_tiled<kRowsCapNarrow>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kSharedBudgetBytes);
            attr_configured_narrow = true;
        }
        k_moe_grouped_expert_forward_fp8_tiled<kRowsCapNarrow>
            <<<num_experts * tiles_per_expert, threads, smem_bytes, stream>>>(
                d_hidden, d_fcin_descale, fcin_is_e5m2, d_route_weights,
                d_counts, d_row_indices,
                d_fc_in_w, d_fc_in_descale, d_fc_out_w, d_fc_out_descale,
                d_mixed, rows, H, Ie, num_experts, rows_per_block, tiles_per_expert);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 5: hand-written backward for everything above. Same file, same
// packing-aware philosophy -- every reduction below is keyed by segs[pos]
// exactly like the forward pooling, and the two genuine "N positions share
// one value" points (the pooled/pooled2 mean-pool+gather pair) are the only
// places needing an explicit sum-by-sample reduction; everything else
// (pressure/modulation/logits/route_weights) is per-position redundant
// computation whose backward is just as redundant and correct without any
// special-casing, since weight-gradient GEMMs already sum over every row
// (redundant or not) and input-gradients only need reducing once they reach
// an actual shared variable -- which is exactly the pool+gather boundary.
// ═══════════════════════════════════════════════════════════════════════════

// GELU backward: d(pre) = d(post) * gelu'(pre), gelu'(x) = Phi(x) + x*phi(x)
// where Phi = 0.5*(1+erf(x/sqrt2)) (the forward's own gate value) and phi is
// the standard-normal pdf. Needs the PRE-activation value (not recoverable
// from the post-activation alone, unlike e.g. sigmoid/tanh) -- callers must
// keep pre_gelu and post_gelu in separate buffers, not overwrite in place.
namespace {
__global__ void k_gelu_bwd(
    const __nv_bfloat16* __restrict__ pre, const __nv_bfloat16* __restrict__ d_post,
    __nv_bfloat16* __restrict__ d_pre, std::size_t n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = __bfloat162float(pre[i]);
    const float cdf = 0.5f * (1.0f + erff(x * 0.7071067811865476f));
    const float pdf = 0.3989422804014327f * expf(-0.5f * x * x);  // 1/sqrt(2*pi)
    const float deriv = cdf + x * pdf;
    d_pre[i] = __float2bfloat16(__bfloat162float(d_post[i]) * deriv);
}
}  // namespace

void gelu_backward(
    const __nv_bfloat16* d_pre, const __nv_bfloat16* d_grad_post, __nv_bfloat16* d_grad_pre,
    std::size_t n, cudaStream_t stream
) {
    k_gelu_bwd<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(d_pre, d_grad_post, d_grad_pre, n);
}

// Row dot product: out[row] = sum_h(a[row,h]*b[row,h]). Used for
// d(route_weight[:,e]) = dot(d_mixed, expert_out_e) per row.
namespace {
__global__ void k_moe_row_dot(
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
    float* __restrict__ out, int H, int out_stride, int out_col
) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const __nv_bfloat16* ar = a + static_cast<std::size_t>(row) * H;
    const __nv_bfloat16* br = b + static_cast<std::size_t>(row) * H;
    float acc = 0.0f;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        acc += __bfloat162float(ar[h]) * __bfloat162float(br[h]);
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[static_cast<std::size_t>(row) * out_stride + out_col] = sdata[0];
}
}  // namespace

// Writes to out[row*out_stride + out_col] -- lets callers fill one column of
// a [rows, num_experts]-shaped buffer directly (out_stride=num_experts,
// out_col=e), e.g. for d(route_weight[:,e]).
void moe_row_dot_bf16(
    const __nv_bfloat16* d_a, const __nv_bfloat16* d_b, float* d_out,
    int rows, int H, int out_stride, int out_col, cudaStream_t stream
) {
    const int T = 256;
    k_moe_row_dot<<<rows, T, T * sizeof(float), stream>>>(d_a, d_b, d_out, H, out_stride, out_col);
}

// ── Small dense projection backward (mirrors moe_small_proj_f32) ───────────
// Forward: out[row,:] = in[row,:] @ W^T, W stored [out_dim,in_dim].

namespace {
// One block per row, one thread per in_dim column (in_dim can be H, up to a
// few thousand -- loop if in_dim > blockDim).
__global__ void k_moe_small_proj_backward_input(
    const float* __restrict__ d_out,      // [rows, out_dim]
    const __nv_bfloat16* __restrict__ w,  // [out_dim, in_dim]
    float* __restrict__ d_in,             // [rows, in_dim]
    int in_dim, int out_dim
) {
    const int row = blockIdx.x;
    const float* dy = d_out + static_cast<std::size_t>(row) * out_dim;
    float* dx = d_in + static_cast<std::size_t>(row) * in_dim;
    for (int h = threadIdx.x; h < in_dim; h += blockDim.x) {
        float acc = 0.0f;
        for (int o = 0; o < out_dim; ++o) {
            acc += dy[o] * __bfloat162float(w[static_cast<std::size_t>(o) * in_dim + h]);
        }
        dx[h] = acc;
    }
}

// Narrow-in_dim variant (2026-07-29, same bug class as Fix 1/Fix 4's
// forward/backward-weight kernels, found while closing out the "worth a
// quick look" item claude-handoff.md flagged for this kernel's sibling).
// Two of this kernel's four real call sites (pressure_to_routes,
// pressure_mod -- trainer.cu ~5628/5666) have in_dim == num_routes, tiny
// (<=11 in every real config) -- the original kernel's one-thread-per-h
// scheme launches as few as num_routes total threads (a SUB-warp launch,
// not just a starved warp), each looping the full out_dim serially. The
// pressure_mod call site is the worse of the two: out_dim == H there (up
// to 4096), so as few as ~9 threads each serially walk up to 4096
// iterations alone. One warp per in_dim column instead: all 32 lanes
// stride over out_dim together, then a shuffle-reduce combines the
// partial sums. Note W's [out_dim, in_dim] row-major layout means this
// access (lanes varying the SLOW-varying out_dim index) is not coalesced
// the way k_moe_small_proj_narrow's forward equivalent is -- accepted
// here because going from ~num_routes-way to 32-way parallelism still
// dominates the coalescing loss, same tradeoff logic as Fix 1/2's
// atomicAdd-based row-chunking. Only used when in_dim <= 32 (guarantees
// blockDim.x = 32*in_dim <= 1024); router_score/pressure_proj (in_dim ==
// H) already get good parallelism from the original kernel and are left
// untouched.
__global__ void k_moe_small_proj_backward_input_narrow(
    const float* __restrict__ d_out,
    const __nv_bfloat16* __restrict__ w,
    float* __restrict__ d_in,
    int in_dim, int out_dim
) {
    const int row = blockIdx.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= in_dim) return;
    const float* dy = d_out + static_cast<std::size_t>(row) * out_dim;
    float acc = 0.0f;
    for (int o = lane; o < out_dim; o += 32)
        acc += dy[o] * __bfloat162float(w[static_cast<std::size_t>(o) * in_dim + warp]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) d_in[static_cast<std::size_t>(row) * in_dim + warp] = acc;
}

// ncu profiling (2026-07-28, real Nsight Compute data, not guesswork) found
// this kernel consuming 31%+ of ALL GPU time in the selected-row expert
// path that already reaches ~18.2k tok/s -- more than every expert FFN
// GEMM combined. Root cause: grid was `in_dim` blocks (2048 for the router
// projection's H=2048, but only ~num_routes for the pressure_mod
// projection -- as few as ~4 blocks total on a 132-SM GPU) with NO
// parallelism across the `rows` reduction dimension at all -- every thread
// serially scanned up to tens of thousands of tokens alone. Fixed: row-
// chunked grid (blockIdx.y over row chunks) with atomicAdd combining each
// chunk's partial sum -- mathematically identical accumulation (atomicAdd
// IS `+=`, just computed by many concurrent partial sums instead of one
// serial scan per (o,i) pair), but now scales block-level parallelism with
// `rows` instead of ignoring it entirely.
__global__ void k_moe_small_proj_backward_weight(
    const float* __restrict__ d_out,   // [rows, out_dim]
    const float* __restrict__ x,       // [rows, in_dim]
    float* __restrict__ dW,            // [out_dim, in_dim], atomicAdd accumulate
    int rows, int in_dim, int out_dim, int rows_per_chunk
) {
    const int i = blockIdx.x;
    if (i >= in_dim) return;
    const int row_start = blockIdx.y * rows_per_chunk;
    if (row_start >= rows) return;
    const int row_end = min(row_start + rows_per_chunk, rows);
    for (int o = threadIdx.x; o < out_dim; o += blockDim.x) {
        float acc = 0.0f;
        for (int row = row_start; row < row_end; ++row) {
            acc += d_out[static_cast<std::size_t>(row) * out_dim + o] * x[static_cast<std::size_t>(row) * in_dim + i];
        }
        if (acc != 0.0f) {
            atomicAdd(&dW[static_cast<std::size_t>(o) * in_dim + i], acc);
        }
    }
}

__global__ void k_moe_small_proj_backward_weight_bf16(
    const float* __restrict__ d_out,
    const float* __restrict__ x,
    __nv_bfloat16* __restrict__ dW,
    int rows, int in_dim, int out_dim, int rows_per_chunk
) {
    const int i = blockIdx.x;
    if (i >= in_dim) return;
    const int row_start = blockIdx.y * rows_per_chunk;
    if (row_start >= rows) return;
    const int row_end = min(row_start + rows_per_chunk, rows);
    for (int o = threadIdx.x; o < out_dim; o += blockDim.x) {
        float acc = 0.0f;
        for (int row = row_start; row < row_end; ++row) {
            acc += d_out[static_cast<std::size_t>(row) * out_dim + o] *
                   x[static_cast<std::size_t>(row) * in_dim + i];
        }
        if (acc != 0.0f) {
            atomicAdd(&dW[static_cast<std::size_t>(o) * in_dim + i],
                      __float2bfloat16(acc));
        }
    }
}
}  // namespace

void moe_small_proj_backward_input(
    const float* d_out, const __nv_bfloat16* d_w, float* d_in,
    int rows, int in_dim, int out_dim, cudaStream_t stream
) {
    if (in_dim <= 32) {
        k_moe_small_proj_backward_input_narrow<<<rows, in_dim * 32, 0, stream>>>(d_out, d_w, d_in, in_dim, out_dim);
    } else {
        const int T = in_dim < 256 ? in_dim : 256;
        k_moe_small_proj_backward_input<<<rows, T, 0, stream>>>(d_out, d_w, d_in, in_dim, out_dim);
    }
}

void moe_small_proj_backward_weight(
    const float* d_out, const float* d_x, __nv_bfloat16* d_w_grad,
    int rows, int in_dim, int out_dim, cudaStream_t stream
) {
    constexpr int kRowsPerChunk = 256;
    const int chunks = (rows + kRowsPerChunk - 1) / kRowsPerChunk;
    const unsigned nchunk = static_cast<unsigned>(chunks > 0 ? chunks : 1);
    const int T = out_dim < 256 ? out_dim : 256;
    const dim3 grid(static_cast<unsigned>(in_dim), nchunk);
    k_moe_small_proj_backward_weight_bf16<<<grid, T, 0, stream>>>(
        d_out, d_x, d_w_grad, rows, in_dim, out_dim, kRowsPerChunk);
}


// ── router-gradient weight kernel, thread-mapping fix (2026-07-31) ──────────
// dW[o,i] = sum_rows d_out[row,o] * x[row,i]
//
// The original mapped THREADS to `o` (out_dim) and BLOCKS to `i` (in_dim):
//     T = out_dim < 256 ? out_dim : 256;
//     grid(in_dim, chunks);
// For the router projections out_dim IS the route/expert count -- 9 for
// pressure_proj (num_cognitive_routes), 11 for router_score
// (num_personality_experts). So it launched blocks of NINE threads into
// 32-lane warps: 72% of every warp idle, across ~524k blocks at mb=16. And
// with `i` fixed per block, x[row*in_dim + i] walked a COLUMN at in_dim*4 =
// 16 KB stride -- every load its own cache line, zero coalescing.
//
// Measured cost before this fix (NCU, 2026-07-28, 2815 ms aggregate): 379.8 ms
// = 13.5% of ALL GPU time, the single largest kernel in the engine, and that
// is AFTER an earlier 2.3x row-chunking fix took it down from 31%.
//
// Fix is a remap, not a rewrite: put threads on the LARGE dimension so warps
// fill and the large-stride operand becomes the coalesced one. The small
// dimension collapses into a per-thread register accumulator, and its operand
// becomes a broadcast every lane in the block shares (L1 serves it once).
// Two mirrored variants because pressure_mod inverts the shape
// (in_dim=9, out_dim=4096). Accumulation order per (o,i) is unchanged, so the
// atomicAdd result is equivalent.
namespace {
constexpr int kSmallProjMaxAcc = 32;   // route/expert counts are 9-11

// in_dim >= out_dim  (pressure_proj, router_score: in=4096, out=9/11)
// threads -> i : x load is coalesced, d_out is a per-row broadcast.
__global__ void k_moe_small_proj_bw_wide(
    const float* __restrict__ d_out, const float* __restrict__ x,
    float* __restrict__ dW, int rows, int in_dim, int out_dim, int rows_per_chunk
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_start = blockIdx.y * rows_per_chunk;
    if (i >= in_dim || row_start >= rows) return;
    const int row_end = min(row_start + rows_per_chunk, rows);

    float acc[kSmallProjMaxAcc];
    for (int o = 0; o < out_dim; ++o) acc[o] = 0.0f;

    for (int row = row_start; row < row_end; ++row) {
        const float xv = x[static_cast<std::size_t>(row) * in_dim + i];   // coalesced
        const float* drow = d_out + static_cast<std::size_t>(row) * out_dim;
        for (int o = 0; o < out_dim; ++o) acc[o] += drow[o] * xv;         // broadcast
    }
    for (int o = 0; o < out_dim; ++o)
        if (acc[o] != 0.0f)
            atomicAdd(&dW[static_cast<std::size_t>(o) * in_dim + i], acc[o]);  // coalesced
}

// out_dim > in_dim  (pressure_mod: in=9, out=4096)
// threads -> o : d_out load is coalesced, x is the per-row broadcast.
__global__ void k_moe_small_proj_bw_tall(
    const float* __restrict__ d_out, const float* __restrict__ x,
    float* __restrict__ dW, int rows, int in_dim, int out_dim, int rows_per_chunk
) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_start = blockIdx.y * rows_per_chunk;
    if (o >= out_dim || row_start >= rows) return;
    const int row_end = min(row_start + rows_per_chunk, rows);

    float acc[kSmallProjMaxAcc];
    for (int i = 0; i < in_dim; ++i) acc[i] = 0.0f;

    for (int row = row_start; row < row_end; ++row) {
        const float dv = d_out[static_cast<std::size_t>(row) * out_dim + o];  // coalesced
        const float* xrow = x + static_cast<std::size_t>(row) * in_dim;
        for (int i = 0; i < in_dim; ++i) acc[i] += dv * xrow[i];              // broadcast
    }
    for (int i = 0; i < in_dim; ++i)
        if (acc[i] != 0.0f)
            atomicAdd(&dW[static_cast<std::size_t>(o) * in_dim + i], acc[i]);
}
}  // namespace


// Ablation gate for the 2026-07-31 small_proj thread remap. Default ON.
// IDA_NATIVE_SMALL_PROJ_REMAP=0 falls back to the ORIGINAL kernel. The remap
// preserves per-(o,i) accumulation order by design, so it SHOULD be nearer
// bit-identical than the LRSS reduction-order change -- this flag is what
// proves that rather than assuming it.
static bool small_proj_remap_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("IDA_NATIVE_SMALL_PROJ_REMAP");
        return !(e && e[0] == '0');
    }();
    return v;
}

void moe_small_proj_backward_weight(
    const float* d_out, const float* d_x, float* d_w_grad,
    int rows, int in_dim, int out_dim, cudaStream_t stream
) {
    constexpr int kRowsPerChunk = 256;
    const int chunks = (rows + kRowsPerChunk - 1) / kRowsPerChunk;
    const unsigned nchunk = static_cast<unsigned>(chunks > 0 ? chunks : 1);
    constexpr int T = 256;

    // Thread the LARGE dimension so warps fill and the strided operand becomes
    // the coalesced one; the small dimension folds into registers and its
    // operand is broadcast. Falls back to the original kernel if the small
    // dimension exceeds the register accumulator.
    const int small = in_dim < out_dim ? in_dim : out_dim;
    if (small_proj_remap_enabled() && small <= kSmallProjMaxAcc) {
        if (in_dim >= out_dim) {
            const dim3 grid(static_cast<unsigned>((in_dim + T - 1) / T), nchunk);
            k_moe_small_proj_bw_wide<<<grid, T, 0, stream>>>(
                d_out, d_x, d_w_grad, rows, in_dim, out_dim, kRowsPerChunk);
        } else {
            const dim3 grid(static_cast<unsigned>((out_dim + T - 1) / T), nchunk);
            k_moe_small_proj_bw_tall<<<grid, T, 0, stream>>>(
                d_out, d_x, d_w_grad, rows, in_dim, out_dim, kRowsPerChunk);
        }
        return;
    }
    const int Tf = out_dim < 256 ? out_dim : 256;
    const dim3 grid(static_cast<unsigned>(in_dim), nchunk);
    k_moe_small_proj_backward_weight<<<grid, Tf, 0, stream>>>(
        d_out, d_x, d_w_grad, rows, in_dim, out_dim, kRowsPerChunk);
}

// ── tanh / sigmoid backward (elementwise, given the POST-activation value) ──

namespace {
__global__ void k_tanh_bwd_f32(const float* post, const float* d_post, float* d_pre, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { const float p = post[i]; d_pre[i] = d_post[i] * (1.0f - p * p); }
}
__global__ void k_sigmoid_bwd_f32(const float* post, const float* d_post, float* d_pre, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { const float p = post[i]; d_pre[i] = d_post[i] * p * (1.0f - p); }
}
}  // namespace

void tanh_backward_f32(const float* d_post, const float* d_grad_post, float* d_grad_pre,
                        std::size_t n, cudaStream_t stream) {
    k_tanh_bwd_f32<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(d_post, d_grad_post, d_grad_pre, n);
}
void sigmoid_backward_f32(const float* d_post, const float* d_grad_post, float* d_grad_pre,
                           std::size_t n, cudaStream_t stream) {
    k_sigmoid_bwd_f32<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(d_post, d_grad_post, d_grad_pre, n);
}

// ── PressureField modulation-multiply backward ──────────────────────────────
// Forward: hidden_gated = hidden * modulation (both already per-position).
// d(hidden) += d(hidden_gated)*modulation;  d(modulation) = d(hidden_gated)*hidden.

namespace {
__global__ void k_moe_modulate_backward(
    const __nv_bfloat16* __restrict__ d_hidden_gated,
    const __nv_bfloat16* __restrict__ hidden,
    const float* __restrict__ modulation,
    __nv_bfloat16* __restrict__ d_hidden_accum,  // +=
    float* __restrict__ d_modulation_out,        // overwrite (fresh per call)
    std::size_t n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float dhg = __bfloat162float(d_hidden_gated[i]);
    const float h = __bfloat162float(hidden[i]);
    d_hidden_accum[i] = __float2bfloat16(__bfloat162float(d_hidden_accum[i]) + dhg * modulation[i]);
    d_modulation_out[i] = dhg * h;
}
}  // namespace

void moe_modulate_hidden_backward(
    const __nv_bfloat16* d_hidden_gated, const __nv_bfloat16* d_hidden,
    const float* d_modulation, __nv_bfloat16* d_hidden_accum, float* d_modulation_grad_out,
    std::size_t n, cudaStream_t stream
) {
    k_moe_modulate_backward<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        d_hidden_gated, d_hidden, d_modulation, d_hidden_accum, d_modulation_grad_out, n);
}

// ── Generic "normalize by row-sum" backward: y = x/S, S = sum_row(x) ────────
// d(x_i) = (d(y_i) - dot_row(d(y),y)) / S. Reused for both ConstitutionalRouter's
// top-k renormalize (sparse->p) and LateralInhibition's final renormalize
// (bounded->route_weights) -- same formula either way, S/row_sum supplied
// by the caller from what the forward kernel already computed.

namespace {
__global__ void k_moe_norm_by_sum_backward(
    const float* __restrict__ y, const float* __restrict__ d_y,
    const float* __restrict__ row_sum, float* __restrict__ d_x, int num_experts
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const float* yr = y + static_cast<std::size_t>(row) * num_experts;
    const float* dyr = d_y + static_cast<std::size_t>(row) * num_experts;
    float* dxr = d_x + static_cast<std::size_t>(row) * num_experts;
    const float S = row_sum[row];
    if (S <= 1e-6f) {
        for (int e = 0; e < num_experts; ++e) dxr[e] = 0.0f;
        return;
    }
    float dot = 0.0f;
    for (int e = 0; e < num_experts; ++e) dot += dyr[e] * yr[e];
    const float inv_s = 1.0f / S;
    for (int e = 0; e < num_experts; ++e) dxr[e] = (dyr[e] - dot) * inv_s;
}
}  // namespace

void moe_norm_by_sum_backward_f32(
    const float* d_y, const float* d_dy, const float* d_row_sum, float* d_dx,
    int rows, int num_experts, cudaStream_t stream
) {
    const int T = 128;
    k_moe_norm_by_sum_backward<<<(rows + T - 1) / T, T, 0, stream>>>(d_y, d_dy, d_row_sum, d_dx, num_experts);
}

// Zeros d_x wherever `reference` is exactly zero -- the topk scatter's
// unselected positions are hard graph constants (zero-filled outputs, never
// touched by the scatter), so their gradient must be discarded regardless
// of what the renormalize backward formula above would otherwise produce
// there (that formula alone doesn't know about the scatter's masking).
namespace {
__global__ void k_moe_zero_where_zero(const float* __restrict__ reference, float* __restrict__ d_x, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && reference[i] == 0.0f) d_x[i] = 0.0f;
}
}  // namespace

void moe_zero_where_zero_f32(const float* d_reference, float* d_x, std::size_t n, cudaStream_t stream) {
    k_moe_zero_where_zero<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(d_reference, d_x, n);
}

// ── LateralInhibition clamp backward ────────────────────────────────────────
// Forward (see moe_lateral_inhibition_f32): c1 = clamp(p, max=dominance_cap);
// bounded = (c1>0 && c1<minority_floor) ? minority_floor : c1. Gradient is
// blocked (zeroed) wherever either clamp actually fired -- saturated at the
// cap, or raised to the floor constant -- and passes through unchanged
// everywhere else (including the "never selected, c1==0" case, which is a
// real data-dependent zero, not a graph constant like topk's scatter).
namespace {
__global__ void k_moe_inhib_clamp_backward(
    const float* __restrict__ c1, float dominance_cap, float minority_floor,
    float* __restrict__ d_bounded, std::size_t n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = c1[i];
    const bool cap_saturated = v >= dominance_cap;
    const bool floor_raised = v > 0.0f && v < minority_floor;
    if (cap_saturated || floor_raised) d_bounded[i] = 0.0f;
}
}  // namespace

void moe_inhib_clamp_backward_f32(
    const float* d_c1, float dominance_cap, float minority_floor, float* d_bounded_inout,
    std::size_t n, cudaStream_t stream
) {
    k_moe_inhib_clamp_backward<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        d_c1, dominance_cap, minority_floor, d_bounded_inout, n);
}

// ── Sample-count + pool/gather backward ─────────────────────────────────────

namespace {
__global__ void k_moe_compute_counts(const std::uint16_t* __restrict__ segs, float* __restrict__ count, int S) {
    const int b = blockIdx.x;
    if (threadIdx.x != 0) return;
    float* count_row = count + static_cast<std::size_t>(b) * S;
    for (int s = 0; s < S; ++s) count_row[s] = 0.0f;
    for (int s = 0; s < S; ++s) {
        const int start = segs ? static_cast<int>(segs[static_cast<std::size_t>(b) * S + s]) : 0;
        count_row[start] += 1.0f;
    }
}

// Scatter-add d_gathered (per-position gradient) into per-sample-start slots
// -- the backward of moe_gather_by_sample's broadcast-read (gather backward
// = scatter-add of every reader's gradient back to its source). Same fix as
// k_moe_pool_scatter above: one block per (b,s) position instead of one
// block per row serially walking all S positions -- the atomicAdd already
// makes per-position order irrelevant, so the serial walk bought nothing.
__global__ void k_moe_pool_backward_scatter(
    const float* __restrict__ d_gathered,  // [B,S,H]
    const std::uint16_t* __restrict__ segs,
    float* __restrict__ d_start_accum,     // [B,S,H], pre-zeroed
    int S, int H
) {
    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int start = segs ? static_cast<int>(segs[static_cast<std::size_t>(b) * S + s]) : 0;
    const float* g = d_gathered + (static_cast<std::size_t>(b) * S + s) * H;
    float* acc = d_start_accum + (static_cast<std::size_t>(b) * S + start) * H;
    for (int h = threadIdx.x; h < H; h += blockDim.x) atomicAdd(&acc[h], g[h]);
}

__global__ void k_moe_pool_backward_divide(float* __restrict__ accum, const float* __restrict__ count, int H) {
    const int row = blockIdx.x;
    const float c = count[row];
    if (c <= 0.0f) return;
    const float inv_c = 1.0f / c;
    float* acc = accum + static_cast<std::size_t>(row) * H;
    for (int h = threadIdx.x; h < H; h += blockDim.x) acc[h] *= inv_c;
}

// Broadcasts the (already count-divided) per-sample-start gradient back to
// every position in that sample, ADDING into a bf16 gradient accumulator --
// the backward of the mean-pool's own sum-over-positions step.
__global__ void k_moe_broadcast_add_bf16(
    const float* __restrict__ start_grad,  // [B,S,H], meaningful only at starts
    const std::uint16_t* __restrict__ segs,
    __nv_bfloat16* __restrict__ d_out,     // [B,S,H], +=
    int S, int H
) {
    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int start = segs ? static_cast<int>(segs[static_cast<std::size_t>(b) * S + s]) : 0;
    const float* src = start_grad + (static_cast<std::size_t>(b) * S + start) * H;
    __nv_bfloat16* dst = d_out + (static_cast<std::size_t>(b) * S + s) * H;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        dst[h] = __float2bfloat16(__bfloat162float(dst[h]) + src[h]);
    }
}
}  // namespace

void moe_compute_sample_counts(const std::uint16_t* d_segs, float* d_count, int B, int S, cudaStream_t stream) {
    k_moe_compute_counts<<<B, 1, 0, stream>>>(d_segs, d_count, S);
}

// Combines: zero scratch -> scatter-add d_gathered by segs -> divide by
// count -> broadcast-add into d_hidden_accum. d_start_grad_scratch is
// [B,S,H] scratch (caller-owned, reused across calls); d_count must already
// hold this segs layout's per-sample-start counts (moe_compute_sample_counts,
// called once and reused -- both pooling passes in one layer share the same
// segs, hence the same counts).
void moe_pool_by_sample_backward(
    const float* d_grad_gathered, const std::uint16_t* d_segs, const float* d_count,
    float* d_start_grad_scratch, __nv_bfloat16* d_hidden_grad_accum,
    int B, int S, int H, cudaStream_t stream
) {
    const std::size_t n = static_cast<std::size_t>(B) * S * H;
    const int T = H < 256 ? H : 256;
    cudaMemsetAsync(d_start_grad_scratch, 0, n * sizeof(float), stream);
    {
        const dim3 grid(static_cast<unsigned>(B), static_cast<unsigned>(S));
        k_moe_pool_backward_scatter<<<grid, T, 0, stream>>>(d_grad_gathered, d_segs, d_start_grad_scratch, S, H);
    }
    k_moe_pool_backward_divide<<<static_cast<unsigned>(B * S), T, 0, stream>>>(d_start_grad_scratch, d_count, H);
    dim3 grid(B, S);
    k_moe_broadcast_add_bf16<<<grid, T, 0, stream>>>(d_start_grad_scratch, d_segs, d_hidden_grad_accum, S, H);
}

// ═══════════════════════════════════════════════════════════════════════════
// Private quantization paths are deployment-owned. The public trainer keeps
// the symbol as a fail-closed compatibility surface so an old local setting
// cannot silently execute an unreviewed precision path.
void moe_fake_quant_roundtrip(
    const __nv_bfloat16*, __nv_bfloat16*, float*, float, std::size_t,
    cudaStream_t
) {
    throw std::runtime_error(
        "private activation quantization is not included in the public binary");
}

void moe_pack_fp4_act(
    float*, float*, float*, float*, float*, const __nv_bfloat16*,
    std::uint8_t*, std::size_t, bool, cudaStream_t
) {
    throw std::runtime_error(
        "private packed-FP4 activation is not included in the public binary");
}

void moe_decode_packed_fp4_to_e4m3(
    const std::uint8_t*, __nv_fp8_e4m3*, const float*, std::size_t,
    cudaStream_t
) {
    throw std::runtime_error(
        "private packed-FP4 activation is not included in the public binary");
}

}  // namespace ida_native
