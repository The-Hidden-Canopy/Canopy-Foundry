// ─── LRSS native: TemporalTraceEmitter + MultiscaleMemoryBank ────────────────
// Port of src/ida_train/models/ida_lattice/{temporal_trace_emitter,
// multiscale_memory}.py per docs/lrss-native-port-contract.md.  The
// LowRankStateSupersampler is deliberately NOT here — it consumes the
// recurrent state cell the native architecture doesn't have.
//
// Everything runs in FP32 internally over BF16-stored weights (the tensors
// are tiny: [B,H], [A,H], [J,B,A] with J=8, A=32 — ~34 MFLOP per micro-step
// against the step's ~550 GFLOP).  Naive kernels beat plumbing cuBLAS for
// shapes this small.
//
// Recompute contract (fp8-doc bug class, designed in):
//   - lrss_record_anchor fires on the TRUE forward only (caller gates on
//     is_recompute) and records the PRE-UPDATE bank view's successor: the
//     bank consumed by micro-step t is frozen before t's record lands.
//   - The forward saves delta/gate/mix/attn/pooled; the backward consumes
//     the saved tensors — the bank never re-advances.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "ida_native/kernels.hpp"

namespace ida_native {

namespace {

constexpr int kTpb = 256;

__device__ inline float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }

// pooled[b,h] = mean over content positions of hidden[b,s,h].
// content = tokens[b,s] not flagged in common_mask (mask null → all tokens).
// n_content[b] written for the backward's scatter.
__global__ void k_lrss_pool(
    const __nv_bfloat16* __restrict__ hidden,   // [B,S,H]
    const std::uint32_t* __restrict__ tokens,   // [B,S]
    const std::uint8_t*  __restrict__ common_mask, // [V] or nullptr
    float* __restrict__ pooled,                 // [B,H]
    float* __restrict__ n_content,              // [B]
    int B, int S, int H
) {
    const int b = blockIdx.x;
    if (b >= B) return;
    // Count content positions once per block (thread 0), share via smem.
    __shared__ float s_n;
    if (threadIdx.x == 0) {
        int n = 0;
        for (int s = 0; s < S; ++s) {
            const std::uint32_t t = tokens[static_cast<std::size_t>(b) * S + s];
            if (!common_mask || !common_mask[t]) ++n;
        }
        s_n = static_cast<float>(n > 0 ? n : S);
        n_content[b] = s_n;
    }
    __syncthreads();
    const float inv_n = 1.0f / s_n;
    const bool all = (s_n == static_cast<float>(S));
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < S; ++s) {
            if (!all) {
                const std::uint32_t t = tokens[static_cast<std::size_t>(b) * S + s];
                if (common_mask && common_mask[t]) continue;
            }
            acc += bf(hidden[(static_cast<std::size_t>(b) * S + s) * H + h]);
        }
        pooled[static_cast<std::size_t>(b) * H + h] = acc * inv_n;
    }
}

// ring[slot][h] = mean over B of pooled[b,h]
__global__ void k_lrss_record(
    const float* __restrict__ pooled, float* __restrict__ ring_slot,
    int B, int H
) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) acc += pooled[static_cast<std::size_t>(b) * H + h];
    ring_slot[h] = acc / static_cast<float>(B);
}

// out[r,n] = sum_k W[n,k] * x[r,k]   (torch nn.Linear orientation, W [N,K] bf16)
__global__ void k_lrss_linear(
    const float* __restrict__ x, const __nv_bfloat16* __restrict__ W,
    float* __restrict__ out, int R, int K, int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * N) return;
    const int r = idx / N, n = idx % N;
    float acc = 0.0f;
    const __nv_bfloat16* wrow = W + static_cast<std::size_t>(n) * K;
    const float* xrow = x + static_cast<std::size_t>(r) * K;
    for (int k = 0; k < K; ++k) acc += bf(wrow[k]) * xrow[k];
    out[idx] = acc;
}

// rel[b,a] = (q[b,:]·k[a,:]) / sqrt(H)
__global__ void k_lrss_relevance(
    const float* __restrict__ q, const float* __restrict__ k,
    float* __restrict__ rel, int B, int A, int H
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * A) return;
    const int b = idx / A, a = idx % A;
    float acc = 0.0f;
    for (int h = 0; h < H; ++h)
        acc += q[static_cast<std::size_t>(b) * H + h] * k[static_cast<std::size_t>(a) * H + h];
    rel[idx] = acc * rsqrtf(static_cast<float>(H));
}

// attn[j,b,:] = softmax_a( rel[b,a] - elapsed[a]/tau[j] ),
// tau[j] = clamp(exp(log_tau[j]), tau_min, tau_max).
// One thread per (j,b): A <= 64 keeps this trivial.
__global__ void k_lrss_attn(
    const float* __restrict__ rel,             // [B,A]
    const __nv_bfloat16* __restrict__ log_tau, // [J]
    const float* __restrict__ elapsed,         // [A]
    float* __restrict__ attn,                  // [J,B,A]
    int J, int B, int A, float tau_min, float tau_max
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= J * B) return;
    const int j = idx / B, b = idx % B;
    const float tau = fminf(fmaxf(expf(bf(log_tau[j])), tau_min), tau_max);
    float m = -1e30f;
    float logits[64];
    for (int a = 0; a < A; ++a) {
        logits[a] = rel[static_cast<std::size_t>(b) * A + a] - elapsed[a] / tau;
        m = fmaxf(m, logits[a]);
    }
    float z = 0.0f;
    for (int a = 0; a < A; ++a) { logits[a] = expf(logits[a] - m); z += logits[a]; }
    const float inv = 1.0f / z;
    for (int a = 0; a < A; ++a)
        attn[(static_cast<std::size_t>(j) * B + b) * A + a] = logits[a] * inv;
}


// ─── Occupancy-fixed LRSS kernels (2026-07-31) ───────────────────────────────
// The originals were correct but launch-shaped badly: on a 132-SM H100 at
// mb=16 (B=16, H=4096, A=32, J=8) they ran at
//   k_lrss_pool       <<<B=16 blocks>>>        12%  of the GPU
//   k_lrss_relevance  <<<gd(B*A)=2 blocks>>>   1.5% of the GPU
//   k_lrss_attn       <<<gd(J*B)=1 block>>>    0.8% of the GPU
// serially, every micro-step, in the critical path. LRSS's FLOPs really are
// trivial (~34 MFLOP vs the step's ~550 GFLOP) -- the wall time came from
// running almost none of the GPU while everything waited, plus per-thread
// serial walks (k_lrss_pool: 32,768 iterations/thread).
//
// Same fix as k_act_row_normsq already uses elsewhere in this engine: one
// block per output element, shuffle+shared reduction instead of a serial walk.
// Math is unchanged, so results are bit-identical.

// Content count, parallel over S. Was thread 0 of the pool kernel walking all
// S alone; hoisted out so the pool kernel can be tiled over H.
__global__ void k_lrss_count(
    const std::uint32_t* __restrict__ tokens,
    const std::uint8_t*  __restrict__ common_mask,
    float* __restrict__ n_content, int B, int S
) {
    __shared__ float sm[kTpb / 32];
    const int b = blockIdx.x;
    float n = 0.0f;
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        const std::uint32_t t = tokens[static_cast<std::size_t>(b) * S + s];
        if (!common_mask || !common_mask[t]) n += 1.0f;
    }
    for (int off = 16; off > 0; off >>= 1) n += __shfl_xor_sync(0xffffffff, n, off);
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = n;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int w = 0; w < blockDim.x / 32; ++w) t += sm[w];
        n_content[b] = (t > 0.0f) ? t : static_cast<float>(S);
    }
}

// One thread per (b,h); grid tiles H so the launch is B*ceil(H/tpb) blocks
// instead of B. Each thread now walks S once for a single h.
__global__ void k_lrss_pool_tiled(
    const __nv_bfloat16* __restrict__ hidden,
    const std::uint32_t* __restrict__ tokens,
    const std::uint8_t*  __restrict__ common_mask,
    float* __restrict__ pooled,
    const float* __restrict__ n_content,
    int B, int S, int H
) {
    const int b = blockIdx.y;
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || h >= H) return;
    const float s_n = n_content[b];
    const float inv_n = 1.0f / s_n;
    const bool all = (s_n == static_cast<float>(S));
    float acc = 0.0f;
    for (int s = 0; s < S; ++s) {
        if (!all) {
            const std::uint32_t t = tokens[static_cast<std::size_t>(b) * S + s];
            if (common_mask && common_mask[t]) continue;
        }
        acc += bf(hidden[(static_cast<std::size_t>(b) * S + s) * H + h]);
    }
    pooled[static_cast<std::size_t>(b) * H + h] = acc * inv_n;
}

// One BLOCK per (b,a) with a shuffle reduction over H, instead of one THREAD
// per (b,a) walking H. B*A blocks (512 at mb=16) vs 2.
__global__ void k_lrss_relevance_par(
    const float* __restrict__ q, const float* __restrict__ k,
    float* __restrict__ rel, int B, int A, int H
) {
    __shared__ float sm[kTpb / 32];
    const int idx = blockIdx.x;
    if (idx >= B * A) return;
    const int b = idx / A, a = idx % A;
    float acc = 0.0f;
    for (int h = threadIdx.x; h < H; h += blockDim.x)
        acc += q[static_cast<std::size_t>(b) * H + h] * k[static_cast<std::size_t>(a) * H + h];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, off);
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int w = 0; w < blockDim.x / 32; ++w) t += sm[w];
        rel[idx] = t * rsqrtf(static_cast<float>(H));
    }
}

// One WARP per (j,b) doing the softmax over A in registers via shuffles.
// The original ran one THREAD per (j,b) with a float logits[64] local array --
// 64 registers or a local-memory spill per thread, on a 1-block launch.
__global__ void k_lrss_attn_warp(
    const float* __restrict__ rel,
    const __nv_bfloat16* __restrict__ log_tau,
    const float* __restrict__ elapsed,
    float* __restrict__ attn,
    int J, int B, int A, float tau_min, float tau_max
) {
    const int idx = blockIdx.x;
    if (idx >= J * B) return;
    const int j = idx / B, b = idx % B;
    const float tau = fminf(fmaxf(expf(bf(log_tau[j])), tau_min), tau_max);
    const int lane = threadIdx.x;
    float v = -1e30f;
    if (lane < A) v = rel[static_cast<std::size_t>(b) * A + lane] - elapsed[lane] / tau;
    float m = v;
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    float e = (lane < A) ? expf(v - m) : 0.0f;
    float z = e;
    for (int off = 16; off > 0; off >>= 1) z += __shfl_xor_sync(0xffffffff, z, off);
    if (lane < A)
        attn[(static_cast<std::size_t>(j) * B + b) * A + lane] = e / z;
}


// Ablation gate for the 2026-07-31 occupancy fix. Default ON (the fix is
// deployed and measured faster); set IDA_NATIVE_LRSS_OCCUPANCY=0 to launch the
// ORIGINAL serial-walk kernels instead. This exists because the fix changes
// REDUCTION ORDER (serial loop -> __shfl_xor_sync tree), and float addition is
// not associative -- so it can move the loss without anything being wrong. The
// 1600-micro numerics arm that returned delta 0.010732 vs reference 4.2857
// carried this change AND the moe small_proj remap at once and therefore could
// not attribute the delta to either. This flag is what makes them separable.
static bool lrss_occupancy_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("IDA_NATIVE_LRSS_OCCUPANCY");
        return !(e && e[0] == '0');
    }();
    return v;
}

// scale_w softmax → sw[J]; then wsum[b,a] = sum_j sw[j]·attn[j,b,a]
__global__ void k_lrss_scale_softmax(
    const __nv_bfloat16* __restrict__ scale_w, float* __restrict__ sw, int J
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float m = -1e30f;
    for (int j = 0; j < J; ++j) m = fmaxf(m, bf(scale_w[j]));
    float z = 0.0f;
    for (int j = 0; j < J; ++j) { sw[j] = expf(bf(scale_w[j]) - m); z += sw[j]; }
    for (int j = 0; j < J; ++j) sw[j] /= z;
}

__global__ void k_lrss_wsum(
    const float* __restrict__ attn, const float* __restrict__ sw,
    float* __restrict__ wsum, int J, int B, int A
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * A) return;
    float acc = 0.0f;
    for (int j = 0; j < J; ++j)
        acc += sw[j] * attn[static_cast<std::size_t>(j) * B * A + idx];
    wsum[idx] = acc;
}

// mix[b,h] = sum_a wsum[b,a]·bank[a,h]
__global__ void k_lrss_contract(
    const float* __restrict__ wsum, const float* __restrict__ bank,
    float* __restrict__ mix, int B, int A, int H
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    const int b = idx / H, h = idx % H;
    float acc = 0.0f;
    for (int a = 0; a < A; ++a)
        acc += wsum[static_cast<std::size_t>(b) * A + a] * bank[static_cast<std::size_t>(a) * H + h];
    mix[idx] = acc;
}

// gate pre-act computed by k_lrss_linear over concat — build concat first.
__global__ void k_lrss_concat(
    const float* __restrict__ pooled, const float* __restrict__ mix,
    float* __restrict__ cat, int B, int H
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * 2 * H) return;
    const int b = idx / (2 * H), c = idx % (2 * H);
    cat[idx] = (c < H) ? pooled[static_cast<std::size_t>(b) * H + c]
                       : mix[static_cast<std::size_t>(b) * H + (c - H)];
}

// g = sigmoid(pre + bias); delta = g∘mix; hidden[b,s,h] += delta[b,h]
__global__ void k_lrss_gate_delta(
    const float* __restrict__ pre, const __nv_bfloat16* __restrict__ bias,
    const float* __restrict__ mix,
    float* __restrict__ gate, float* __restrict__ delta, int B, int H
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    const int h = idx % H;
    const float g = 1.0f / (1.0f + expf(-(pre[idx] + bf(bias[h]))));
    gate[idx] = g;
    delta[idx] = g * mix[idx];
}

__global__ void k_lrss_broadcast_add(
    __nv_bfloat16* __restrict__ hidden, const float* __restrict__ delta,
    int B, int S, int H
) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t n = static_cast<std::size_t>(B) * S * H;
    if (idx >= n) return;
    const int h = idx % H;
    const int b = idx / (static_cast<std::size_t>(S) * H);
    hidden[idx] = __float2bfloat16(bf(hidden[idx]) + delta[static_cast<std::size_t>(b) * H + h]);
}

// ── backward kernels ─────────────────────────────────────────────────────────

// d_delta[b,h] = sum_s d_hidden[b,s,h]
__global__ void k_lrss_ddelta(
    const __nv_bfloat16* __restrict__ d_hidden, float* __restrict__ d_delta,
    int B, int S, int H
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    const int b = idx / H, h = idx % H;
    float acc = 0.0f;
    for (int s = 0; s < S; ++s)
        acc += bf(d_hidden[(static_cast<std::size_t>(b) * S + s) * H + h]);
    d_delta[idx] = acc;
}

// dgate_pre = d_delta∘mix∘g(1-g);  dmix_1 = d_delta∘g
// db[h] += sum_b dgate_pre  (grad for bias, fp32 accumulate via atomics)
__global__ void k_lrss_dgate(
    const float* __restrict__ d_delta, const float* __restrict__ mix,
    const float* __restrict__ gate,
    float* __restrict__ dpre, float* __restrict__ dmix,
    float* __restrict__ g_bias, int B, int H
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    const int h = idx % H;
    const float g = gate[idx];
    const float dp = d_delta[idx] * mix[idx] * g * (1.0f - g);
    dpre[idx] = dp;
    dmix[idx] = d_delta[idx] * g;
    atomicAdd(&g_bias[h], dp);
}

// dW[n,k] += sum_r dy[r,n]·x[r,k]   (fp32 grad accumulate)
__global__ void k_lrss_dweight(
    const float* __restrict__ dy, const float* __restrict__ x,
    float* __restrict__ dW, int R, int K, int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    const int n = idx / K, k = idx % K;
    float acc = 0.0f;
    for (int r = 0; r < R; ++r)
        acc += dy[static_cast<std::size_t>(r) * N + n] * x[static_cast<std::size_t>(r) * K + k];
    dW[idx] += acc;
}

// dx[r,k] = sum_n dy[r,n]·W[n,k]
__global__ void k_lrss_dinput(
    const float* __restrict__ dy, const __nv_bfloat16* __restrict__ W,
    float* __restrict__ dx, int R, int K, int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * K) return;
    const int r = idx / K, k = idx % K;
    float acc = 0.0f;
    for (int n = 0; n < N; ++n)
        acc += dy[static_cast<std::size_t>(r) * N + n] * bf(W[static_cast<std::size_t>(n) * K + k]);
    dx[idx] = acc;
}

// split d_cat → d_pooled(+=), d_mix(+=)
__global__ void k_lrss_split(
    const float* __restrict__ dcat, float* __restrict__ dpooled,
    float* __restrict__ dmix, int B, int H
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * 2 * H) return;
    const int b = idx / (2 * H), c = idx % (2 * H);
    if (c < H) dpooled[static_cast<std::size_t>(b) * H + c] += dcat[idx];
    else       dmix  [static_cast<std::size_t>(b) * H + (c - H)] += dcat[idx];
}

// dwsum[b,a] = sum_h dmix[b,h]·bank[a,h]   (bank detached: no d_bank)
__global__ void k_lrss_dwsum(
    const float* __restrict__ dmix, const float* __restrict__ bank,
    float* __restrict__ dwsum, int B, int A, int H
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * A) return;
    const int b = idx / A, a = idx % A;
    float acc = 0.0f;
    for (int h = 0; h < H; ++h)
        acc += dmix[static_cast<std::size_t>(b) * H + h] * bank[static_cast<std::size_t>(a) * H + h];
    dwsum[idx] = acc;
}

// d_scale_w[j] += sum_{b,a} dwsum[b,a]·attn[j,b,a] chained through the
// scale softmax: dsw[j] = sw[j]·(dot_j - sum_i sw[i]·dot_i).
// d_attn[j,b,a] = sw[j]·dwsum[b,a] → chained through per-(j,b) softmax:
// drel[b,a] += sw[j]·(dwsum[b,a] - sum_a' attn·dwsum)·attn[j,b,a]
// d_log_tau[j] += sum_{b,a} dattn_pre[j,b,a]·(elapsed[a]/tau_j)·(clamp-interior)
__global__ void k_lrss_dattn(
    const float* __restrict__ dwsum, const float* __restrict__ attn,
    const float* __restrict__ sw, const __nv_bfloat16* __restrict__ log_tau,
    const float* __restrict__ elapsed,
    float* __restrict__ drel,        // [B,A] accumulate
    float* __restrict__ g_log_tau,   // [J] fp32 accumulate
    float* __restrict__ dot_j,       // [J] scratch: sum dwsum·attn_j
    int J, int B, int A, float tau_min, float tau_max
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= J * B) return;
    const int j = idx / B, b = idx % B;
    const float tau_raw = expf(bf(log_tau[j]));
    const float tau = fminf(fmaxf(tau_raw, tau_min), tau_max);
    const bool interior = (tau_raw > tau_min && tau_raw < tau_max);
    const float swj = sw[j];
    const float* at = attn + (static_cast<std::size_t>(j) * B + b) * A;
    const float* dw = dwsum + static_cast<std::size_t>(b) * A;
    float inner = 0.0f, dot = 0.0f;
    for (int a = 0; a < A; ++a) { inner += at[a] * dw[a]; dot += at[a] * dw[a]; }
    float dtau_acc = 0.0f;
    for (int a = 0; a < A; ++a) {
        const float dlogit = swj * (dw[a] - inner) * at[a];
        atomicAdd(&drel[static_cast<std::size_t>(b) * A + a], dlogit);
        // logit = rel - elapsed/tau → d(logit)/d(log_tau) = elapsed/tau (interior)
        if (interior) dtau_acc += dlogit * (elapsed[a] / tau);
    }
    if (interior && dtau_acc != 0.0f) atomicAdd(&g_log_tau[j], dtau_acc);
    atomicAdd(&dot_j[j], dot);
}

__global__ void k_lrss_dscalew(
    const float* __restrict__ dot_j, const float* __restrict__ sw,
    float* __restrict__ g_scale_w, int J
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float sdot = 0.0f;
    for (int j = 0; j < J; ++j) sdot += sw[j] * dot_j[j];
    for (int j = 0; j < J; ++j) g_scale_w[j] += sw[j] * (dot_j[j] - sdot);
}

// dq[b,h] += drel[b,a]·k[a,h]/√H ;  dk[a,h] += drel[b,a]·q[b,h]/√H
__global__ void k_lrss_drel_qk(
    const float* __restrict__ drel, const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ dq, float* __restrict__ dk, int B, int A, int H
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float inv = rsqrtf(static_cast<float>(H));
    if (idx < B * H) {
        const int b = idx / H, h = idx % H;
        float acc = 0.0f;
        for (int a = 0; a < A; ++a)
            acc += drel[static_cast<std::size_t>(b) * A + a] * k[static_cast<std::size_t>(a) * H + h];
        dq[idx] = acc * inv;
    } else if (idx < B * H + A * H) {
        const int i2 = idx - B * H;
        const int a = i2 / H, h = i2 % H;
        float acc = 0.0f;
        for (int b = 0; b < B; ++b)
            acc += drel[static_cast<std::size_t>(b) * A + a] * q[static_cast<std::size_t>(b) * H + h];
        dk[i2] = acc * inv;
    }
}

// scatter d_pooled back to d_hidden over content positions
__global__ void k_lrss_pool_scatter(
    __nv_bfloat16* __restrict__ d_hidden, const float* __restrict__ d_pooled,
    const std::uint32_t* __restrict__ tokens,
    const std::uint8_t* __restrict__ common_mask,
    const float* __restrict__ n_content, int B, int S, int H
) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t n = static_cast<std::size_t>(B) * S * H;
    if (idx >= n) return;
    const int h = idx % H;
    const std::size_t bs = idx / H;
    const int b = bs / S, s = bs % S;
    if (common_mask) {
        const std::uint32_t t = tokens[static_cast<std::size_t>(b) * S + s];
        if (common_mask[t] && n_content[b] != static_cast<float>(S)) return;
    }
    d_hidden[idx] = __float2bfloat16(
        bf(d_hidden[idx]) + d_pooled[static_cast<std::size_t>(b) * H + h] / n_content[b]);
}

// ── LSS head kernels ─────────────────────────────────────────────────────────
// The int2 store operand: anchors quantize to 4 uniform levels
// {-1, -1/3, 1/3, 1}·scale (codes {-3,-1,1,3}, dequant = code/3·scale).

// Two-pass int2 anchor quantizer: pass 1 computes the batch-mean row's amax
// scale; pass 2 quantizes to the 4 uniform levels.
__global__ void k_lss_amax_meanrow(
    const float* __restrict__ mix, float* __restrict__ scale, int B, int H
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float m = 1e-12f;
    for (int h = 0; h < H; ++h) {
        float acc = 0.0f;
        for (int b = 0; b < B; ++b) acc += mix[static_cast<std::size_t>(b) * H + h];
        m = fmaxf(m, fabsf(acc / B));
    }
    *scale = m;
}
__global__ void k_lss_quant2(
    const float* __restrict__ mix, const float* __restrict__ scale,
    signed char* __restrict__ q, int B, int H
) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) acc += mix[static_cast<std::size_t>(b) * H + h];
    acc /= B;
    const float v = fminf(fmaxf(acc / *scale, -1.0f), 1.0f);
    // nearest of {-1,-1/3,1/3,1}: thresholds at ±2/3 and 0
    signed char c;
    if (v >= 2.0f / 3.0f) c = 3;
    else if (v >= 0.0f)   c = 1;
    else if (v >= -2.0f / 3.0f) c = -1;
    else c = -3;
    q[h] = c;
}
__global__ void k_lss_dequant(
    const signed char* __restrict__ q, const float* __restrict__ scale,
    float* __restrict__ anchor, int H
) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    anchor[h] = (static_cast<float>(q[h]) / 3.0f) * (*scale);
}

// PSS Stage 1b: joint widens from [pooled;anchor] to [pooled;anchor;spike]
// when spike_dim > 0 (IDA_NATIVE_PSS_SPIKE_JOINT=1). spike is a small
// [spike_dim] vector broadcast identically across the batch -- it is the
// LRSS per-slot spike-ratio bucket summary staged once per optimizer step
// (see the PSS Stage 1b block in trainer.cu), not a per-example quantity.
// spike_dim=0 (the default) reduces this exactly to the original
// [pooled;anchor] kernel -- the `else` branch below is unreachable in that
// case since idx never reaches c >= 2*H.
__global__ void k_lss_joint(
    const float* __restrict__ pooled, const float* __restrict__ anchor,
    const float* __restrict__ spike, int spike_dim,
    float* __restrict__ joint, int B, int H
) {
    const int joint_dim = 2 * H + spike_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * joint_dim) return;
    const int b = idx / joint_dim, c = idx % joint_dim;
    if (c < H) joint[idx] = pooled[static_cast<std::size_t>(b) * H + c];
    else if (c < 2 * H) joint[idx] = anchor[c - H];
    else joint[idx] = spike[c - 2 * H];
}

__global__ void k_lss_relu(float* x, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(x[i], 0.0f);
}

// recon = anchor + up_out ; aux partial = (recon − mix)² accumulated
__global__ void k_lss_recon_aux(
    const float* __restrict__ up_out, const float* __restrict__ anchor,
    const float* __restrict__ mix,
    float* __restrict__ recon, float* __restrict__ aux, int B, int H
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    const int h = idx % H;
    const float r = anchor[h] + up_out[idx];
    recon[idx] = r;
    const float e = r - mix[idx];
    atomicAdd(aux, e * e / (B * H));
}

// d_recon = gate-path grad + aux grad (2·w·(recon−mix)/(B·H));
// relu backward mask applied to d_hidden_lss.
__global__ void k_lss_drecon_aux(
    const float* __restrict__ recon, const float* __restrict__ mix,
    float* __restrict__ d_recon, float aux_w, int B, int H
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    d_recon[idx] += aux_w * 2.0f * (recon[idx] - mix[idx]) / (B * H);
}
__global__ void k_lss_relu_bwd(
    const float* __restrict__ act, float* __restrict__ grad, std::size_t n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && act[i] <= 0.0f) grad[i] = 0.0f;
}

inline int gd(std::size_t n) { return static_cast<int>((n + kTpb - 1) / kTpb); }

__global__ void k_f32_absmax(const float* __restrict__ x, std::size_t n, float* out) {
    float local = 0.0f;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        local = fmaxf(local, fabsf(x[i]));
    }
    atomicMax(reinterpret_cast<int*>(out), __float_as_int(local));
}

bool lrss_debug_enabled() {
    const char* e = std::getenv("IDA_NATIVE_LRSS_DEBUG");
    return e && e[0] == '1';
}

// The reconstruction is injected into the trunk by default
// (mix = lss_recon) so the gate/delta path uses the sharpened output instead
// of the raw bank retrieval. IDA_NATIVE_LSS_INJECT=0 retains shadow mode for
// controlled ablation without changing checkpoint shape.
bool lrss_inject_enabled() {
    const char* e = std::getenv("IDA_NATIVE_LSS_INJECT");
    return !(e && e[0] == '0');
}

bool lss_feedback_enabled() {
    const char* e = std::getenv("IDA_NATIVE_LSS_FEEDBACK_SKIP");
    return !(e && e[0] == '0');
}

void lrss_debug_print(const char* label, bool is_recompute, int call_id,
                       const float* d_x, std::size_t n, cudaStream_t st) {
    if (!lrss_debug_enabled()) return;
    float* d_out = nullptr;
    cudaMallocAsync(&d_out, sizeof(float), st);
    cudaMemsetAsync(d_out, 0, sizeof(float), st);
    const std::size_t b = (n + 255) / 256;
    const unsigned blocks = static_cast<unsigned>(b < 512 ? (b < 1 ? 1 : b) : 512);
    k_f32_absmax<<<blocks, 256, 0, st>>>(d_x, n, d_out);
    float h = 0.0f;
    cudaMemcpyAsync(&h, d_out, sizeof(float), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    cudaFreeAsync(d_out, st);
    std::fprintf(stderr, "[lrss-debug] call=%d %-8s %-12s absmax=%.10g\n",
                 call_id, is_recompute ? "(re)" : "(fw)", label, h);
}

}  // namespace

// ── host orchestration ───────────────────────────────────────────────────────

void lrss_forward(
    const LrssParams& p, LrssScratch& s,
    __nv_bfloat16* hidden, const std::uint32_t* tokens,
    const std::uint8_t* common_mask,
    int B, int S, int H, cudaStream_t st, bool is_recompute
) {
    static int g_lrss_debug_call = 0;
    const int call_id = lrss_debug_enabled() ? g_lrss_debug_call++ : 0;
    s.B_last = B;
    // The pool always runs — the anchor recorder needs it even on a cold
    // bank (first micro-step records the first anchor from this pool).
    if (lrss_occupancy_enabled()) {
        // Occupancy fix: count hoisted (parallel over S), pool tiled over H so
        // the launch is B*ceil(H/kTpb) blocks instead of B (16 -> 256 at mb=16).
        k_lrss_count<<<B, kTpb, 0, st>>>(tokens, common_mask, s.n_content, B, S);
        dim3 gpool(static_cast<unsigned>((H + kTpb - 1) / kTpb), static_cast<unsigned>(B));
        k_lrss_pool_tiled<<<gpool, kTpb, 0, st>>>(
            hidden, tokens, common_mask, s.pooled, s.n_content, B, S, H);
    } else {
        k_lrss_pool<<<B, kTpb, 0, st>>>(hidden, tokens, common_mask, s.pooled, s.n_content, B, S, H);
    }
    if (s.bank_count <= 0) {
        // Cold bank: no injection this step (Python behaves identically —
        // the multiscale block is skipped until anchors exist).
        cudaMemsetAsync(s.delta, 0, static_cast<std::size_t>(B) * H * sizeof(float), st);
        s.applied = false;
        return;
    }
    const int A = s.bank_count;
    const int J = p.num_scales;
    k_lrss_linear<<<gd(static_cast<std::size_t>(B) * H), kTpb, 0, st>>>(s.pooled, p.query_w, s.q, B, H, H);
    k_lrss_linear<<<gd(static_cast<std::size_t>(A) * H), kTpb, 0, st>>>(s.bank, p.key_w, s.k, A, H, H);
    if (lrss_occupancy_enabled()) {
        k_lrss_relevance_par<<<static_cast<unsigned>(B) * A, kTpb, 0, st>>>(s.q, s.k, s.rel, B, A, H);
    } else {
        k_lrss_relevance<<<gd(static_cast<std::size_t>(B) * A), kTpb, 0, st>>>(s.q, s.k, s.rel, B, A, H);
    }
    if (lrss_occupancy_enabled()) {
        k_lrss_attn_warp<<<static_cast<unsigned>(J) * B, 32, 0, st>>>(
            s.rel, p.log_tau, s.elapsed, s.attn, J, B, A, p.tau_min, p.tau_max);
    } else {
        k_lrss_attn<<<gd(static_cast<std::size_t>(J) * B), kTpb, 0, st>>>(
            s.rel, p.log_tau, s.elapsed, s.attn, J, B, A, p.tau_min, p.tau_max);
    }
    k_lrss_scale_softmax<<<1, 1, 0, st>>>(p.scale_w, s.sw, J);
    k_lrss_wsum<<<gd(static_cast<std::size_t>(B) * A), kTpb, 0, st>>>(s.attn, s.sw, s.wsum, J, B, A);
    k_lrss_contract<<<gd(static_cast<std::size_t>(B) * H), kTpb, 0, st>>>(s.wsum, s.bank, s.mix, B, A, H);
    lrss_debug_print("pooled", is_recompute, call_id, s.pooled, static_cast<std::size_t>(B) * H, st);
    lrss_debug_print("mix(bank)", is_recompute, call_id, s.mix, static_cast<std::size_t>(B) * H, st);

    // LSS head (IDA_NATIVE_LSS, SHADOW mode): reconstruct the mix from the
    // int2 anchor store + current pooled state, in parallel with the bank
    // (which keeps driving the injection and training via the main loss —
    // the Python module's runs-alongside contract).  The LSS trains purely
    // from the aux reconstruction loss; a falling aux curve IS the "does
    // LSS work" answer.  Cold anchor → head idle this step.
    s.lss_active_this_step = false;
    if (p.lss_rank > 0 && s.lss_anchor_valid) {
        const int R = p.lss_rank;
        // joint_dim must match the FIXED width lss_down/lss_joint were
        // allocated with (LatticeWeights::pss_spike_joint_dim) -- it cannot
        // vary at runtime or the down-proj GEMM below reads the weight
        // matrix at the wrong stride. Before the first optimizer-step
        // publish (pss_spike_valid==0), the spike columns are simply the
        // zeros the allocator memset them to -- a harmless "no signal yet",
        // not a dimension change.
        const int spike_dim = s.pss_spike ? p.pss_spike_dim : 0;
        const int joint_dim = 2 * H + spike_dim;
        k_lss_dequant<<<gd(H), kTpb, 0, st>>>(s.lss_anchor_q, s.lss_anchor_scale, s.lss_anchor, H);
        k_lss_joint<<<gd(static_cast<std::size_t>(B) * joint_dim), kTpb, 0, st>>>(
            s.pooled, s.lss_anchor, s.pss_spike, spike_dim, s.lss_joint, B, H);
        k_lrss_linear<<<gd(static_cast<std::size_t>(B) * R), kTpb, 0, st>>>(
            s.lss_joint, p.lss_down, s.lss_hidden, B, joint_dim, R);
        k_lss_relu<<<gd(static_cast<std::size_t>(B) * R), kTpb, 0, st>>>(
            s.lss_hidden, static_cast<std::size_t>(B) * R);
        k_lrss_linear<<<gd(static_cast<std::size_t>(B) * H), kTpb, 0, st>>>(
            s.lss_hidden, p.lss_up, s.lss_recon, B, R, H);
        // Intermittent-crash guard (2026-07-30 investigation): k_lss_recon_aux
        // was observed under compute-sanitizer memcheck to atomicAdd into a
        // NULL s.lss_aux -- 523 hazards, fully reproducible on ONE run, then
        // absent on an immediately-following identical rerun (same binary,
        // same request, same recipe). That intermittency rules out a static
        // logic bug (allocation-guard mismatch, stream-ordering, dangling
        // stack pointer, and OOM-retry masking in arena.cu's
        // ida_malloc_async were all directly ruled out by reading code) --
        // it is a genuine timing-dependent corruption whose true trigger is
        // not yet root-caused. Racecheck against the vendor FP8 GEMM kernels
        // that dominate this recipe is swamped with false-positive hazard
        // reports (Hopper cuBLASLt/cutlass warp-specialized kernels use
        // mbarrier synchronization racecheck's model doesn't understand),
        // so it may never surface the real hit. Until root-caused, fail
        // loud and diagnostic at the true origin instead of segfaulting
        // opaquely at some later, unrelated sync point -- same philosophy
        // as cuda_context_is_healthy()'s self-detection for a poisoned
        // context (native/include/ida_native/cuda_check.hpp): continuing
        // to launch into a corrupted pointer risks a silently wrong
        // reconstruction-aux accumulation, worse than a loud, fast,
        // diagnosable failure the existing supervisor/retry stack already
        // recovers from.
        if (!s.lss_aux || !s.lss_recon || !s.lss_anchor || !s.mix || !s.lss_hidden) {
            std::fprintf(stderr,
                "[ida_native_train] FATAL: LSS scratch pointer invalid "
                "before k_lss_recon_aux (call_id=%d is_recompute=%d B=%d "
                "H=%d R=%d lss_aux=%p lss_recon=%p lss_anchor=%p mix=%p "
                "lss_hidden=%p p.lss_down=%p p.lss_up=%p &s=%p) -- refusing "
                "to launch into a corrupted/null buffer; this is the "
                "intermittent LSS scratch race under investigation.\n",
                call_id, is_recompute, B, H, R,
                (void*)s.lss_aux, (void*)s.lss_recon, (void*)s.lss_anchor,
                (void*)s.mix, (void*)s.lss_hidden, (void*)p.lss_down,
                (void*)p.lss_up, (void*)&s);
            std::fflush(stderr);
            std::_Exit(1);
        }
        cudaMemsetAsync(s.lss_aux, 0, sizeof(float), st);
        k_lss_recon_aux<<<gd(static_cast<std::size_t>(B) * H), kTpb, 0, st>>>(
            s.lss_recon, s.lss_anchor, s.mix, s.lss_recon, s.lss_aux, B, H);
        // Injection replaces the bank mix with the reconstruction so the
        // gate/delta path uses the sharpened output. Shadow mode remains an
        // explicit ablation through IDA_NATIVE_LSS_INJECT=0.
        s.lss_injected = false;
        if (lrss_inject_enabled() && s.mix_bank_save) {
            // Save the bank mix before overwriting — lrss_record_anchor needs
            // the original bank output, not the injected reconstruction, so the
            // anchor tracks the BANK rather than the potentially-unstable recon.
            cudaMemcpyAsync(s.mix_bank_save, s.mix,
                            static_cast<std::size_t>(B) * H * sizeof(float),
                            cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(s.mix, s.lss_recon,
                            static_cast<std::size_t>(B) * H * sizeof(float),
                            cudaMemcpyDeviceToDevice, st);
            s.lss_injected = true;
        }
        s.lss_active_this_step = true;
        // Evidence and authority are separate. Always publish the measured
        // auxiliary loss; only feedback_ready may authorize it to alter the
        // next forward. Gating the copy on feedback made valid runs report a
        // fabricated zero whenever mutation was disabled for an ablation.
        cudaMemcpyAsync(&s.lss_last_aux, s.lss_aux, sizeof(float),
                        cudaMemcpyDeviceToHost, st);
        s.lss_aux_copy_pending = 1;
        s.lss_aux_valid = 0;
        s.lss_feedback_ready = 0;
        if (lss_feedback_enabled()) {
            // Feedback is a next-micro state transition and therefore needs
            // the measurement now. Evidence-only runs publish at the
            // trainer's existing heartbeat/window fence instead.
            cudaStreamSynchronize(st);
            s.lss_aux_copy_pending = 0;
            s.lss_aux_valid = 1;
            s.lss_feedback_ready = 1;
        }
        lrss_debug_print("anchor", is_recompute, call_id, s.lss_anchor, H, st);
        lrss_debug_print("joint", is_recompute, call_id, s.lss_joint, static_cast<std::size_t>(B) * 2 * H, st);
        lrss_debug_print("hidden", is_recompute, call_id, s.lss_hidden, static_cast<std::size_t>(B) * R, st);
        lrss_debug_print("recon", is_recompute, call_id, s.lss_recon, static_cast<std::size_t>(B) * H, st);
        lrss_debug_print("mix(post)", is_recompute, call_id, s.mix, static_cast<std::size_t>(B) * H, st);
    }

    k_lrss_concat<<<gd(static_cast<std::size_t>(B) * 2 * H), kTpb, 0, st>>>(s.pooled, s.mix, s.cat, B, H);
    k_lrss_linear<<<gd(static_cast<std::size_t>(B) * H), kTpb, 0, st>>>(s.cat, p.gate_w, s.gpre, B, 2 * H, H);
    k_lrss_gate_delta<<<gd(static_cast<std::size_t>(B) * H), kTpb, 0, st>>>(
        s.gpre, p.gate_b, s.mix, s.gate, s.delta, B, H);
    k_lrss_broadcast_add<<<gd(static_cast<std::size_t>(B) * S * H), kTpb, 0, st>>>(hidden, s.delta, B, S, H);
    lrss_debug_print("gate", is_recompute, call_id, s.gate, static_cast<std::size_t>(B) * H, st);
    lrss_debug_print("delta", is_recompute, call_id, s.delta, static_cast<std::size_t>(B) * H, st);
    s.applied = true;
}

void lrss_record_anchor(
    LrssScratch& s, int max_anchors, int H, int causal_time, cudaStream_t st
) {
    // Requires lrss_forward's pool of THIS step (s.pooled valid).  Ring is
    // [max_anchors][H]; cursor wraps; elapsed recomputed host-side at read.
    k_lrss_record<<<gd(H), kTpb, 0, st>>>(s.pooled, s.ring + static_cast<std::size_t>(s.cursor) * H, s.B_last, H);
    s.times[s.cursor] = causal_time;
    s.cursor = (s.cursor + 1) % max_anchors;
    if (s.count < max_anchors) ++s.count;
    // LSS anchor store (the int2 operand): this step's actual bank output,
    // batch-meaned and quantized to 4 levels — the coarse memory the NEXT
    // step's reconstruction starts from.  True-forward hook only (same
    // recompute rule as the ring).
    if (s.lss_anchor_q && s.applied) {
        // Always quantize the BANK mix as the anchor, never the injected recon.
        // When injection is active s.mix holds the reconstruction; mix_bank_save
        // holds the original bank retrieval that the anchor should track.
        const float* anchor_src = (s.lss_injected && s.mix_bank_save)
                                  ? s.mix_bank_save : s.mix;
        k_lss_amax_meanrow<<<1, 1, 0, st>>>(anchor_src, s.lss_anchor_scale, s.B_last, H);
        k_lss_quant2<<<gd(H), kTpb, 0, st>>>(anchor_src, s.lss_anchor_scale, s.lss_anchor_q, s.B_last, H);
        s.lss_anchor_valid = 1;
    }
}

void lrss_backward(
    const LrssParams& p, LrssScratch& s, LrssGrads& g,
    __nv_bfloat16* d_hidden, const std::uint32_t* tokens,
    const std::uint8_t* common_mask,
    int B, int S, int H, cudaStream_t st
) {
    if (!s.applied) return;
    const int A = s.bank_count;
    const int J = p.num_scales;
    const std::size_t bh = static_cast<std::size_t>(B) * H;
    cudaMemsetAsync(s.d_pooled, 0, bh * sizeof(float), st);
    cudaMemsetAsync(s.d_mix, 0, bh * sizeof(float), st);
    cudaMemsetAsync(s.drel, 0, static_cast<std::size_t>(B) * A * sizeof(float), st);
    cudaMemsetAsync(s.dot_j, 0, static_cast<std::size_t>(J) * sizeof(float), st);

    k_lrss_ddelta<<<gd(bh), kTpb, 0, st>>>(d_hidden, s.d_delta, B, S, H);
    k_lrss_dgate<<<gd(bh), kTpb, 0, st>>>(
        s.d_delta, s.mix, s.gate, s.dpre, s.d_mix, g.gate_b, B, H);
    // gate GEMM grads: dW_gate += dpreᵀ·cat ; d_cat = dpre·W_gate
    k_lrss_dweight<<<gd(static_cast<std::size_t>(H) * 2 * H), kTpb, 0, st>>>(
        s.dpre, s.cat, g.gate_w, B, 2 * H, H);
    k_lrss_dinput<<<gd(static_cast<std::size_t>(B) * 2 * H), kTpb, 0, st>>>(
        s.dpre, p.gate_w, s.dcat, B, 2 * H, H);
    k_lrss_split<<<gd(static_cast<std::size_t>(B) * 2 * H), kTpb, 0, st>>>(s.dcat, s.d_pooled, s.d_mix, B, H);
    // In shadow mode the gate consumed the bank mix, so its task gradient
    // follows the bank query/key path. Inject mode replaced that mix with the
    // LSS reconstruction; routing the same d_mix through the bank would add a
    // gradient edge that did not exist in the forward graph.
    if (!s.lss_injected) {
        k_lrss_dwsum<<<gd(static_cast<std::size_t>(B) * A), kTpb, 0, st>>>(s.d_mix, s.bank, s.dwsum, B, A, H);
        k_lrss_dattn<<<gd(static_cast<std::size_t>(J) * B), kTpb, 0, st>>>(
            s.dwsum, s.attn, s.sw, p.log_tau, s.elapsed,
            s.drel, g.log_tau, s.dot_j, J, B, A, p.tau_min, p.tau_max);
        k_lrss_dscalew<<<1, 1, 0, st>>>(s.dot_j, s.sw, g.scale_w, J);
        // relevance → q/k paths
        k_lrss_drel_qk<<<gd(bh + static_cast<std::size_t>(A) * H), kTpb, 0, st>>>(
            s.drel, s.q, s.k, s.dq, s.dk, B, A, H);
        k_lrss_dweight<<<gd(static_cast<std::size_t>(H) * H), kTpb, 0, st>>>(
            s.dq, s.pooled, g.query_w, B, H, H);
        k_lrss_dweight<<<gd(static_cast<std::size_t>(H) * H), kTpb, 0, st>>>(
            s.dk, s.bank, g.key_w, A, H, H);
        // d_pooled += dq·W_query (input-side of the query projection)
        k_lrss_dinput<<<gd(bh), kTpb, 0, st>>>(s.dq, p.query_w, s.dq_in, B, H, H);
        // fold: d_pooled aggregates gate-path (already in) + query-path
        k_lrss_split_add(s.d_pooled, s.dq_in, bh, st);
    }

    // LSS head backward.  d_recon starts from the main-loss gradient (d_mix
    // from the gate path, since mix = lss_recon after injection), then the
    // aux reconstruction term is added on top.  Grads flow to lss_up/lss_down
    // and (via the joint's pooled half) into d_pooled.  The anchor half is the
    // detached int2 store.
    if (p.lss_rank > 0 && s.lss_active_this_step) {
        const int R = p.lss_rank;
        lrss_debug_print("d_pooled(pre-lss)", false, -1, s.d_pooled, bh, st);
        // Seed d_lss_recon:
        //  - Inject mode: copy d_mix (∂L/∂mix == ∂L/∂lss_recon since mix=lss_recon)
        //    so the main-loss gradient trains lss_up/lss_down alongside the aux loss.
        //  - Shadow mode: zero — LSS does not affect the trunk so only the aux loss
        //    (MSE reconstruction fidelity) should shape the LSS weights.
        if (s.lss_injected) {
            cudaMemcpyAsync(s.d_lss_recon, s.d_mix, bh * sizeof(float),
                            cudaMemcpyDeviceToDevice, st);
        } else {
            cudaMemsetAsync(s.d_lss_recon, 0, bh * sizeof(float), st);
        }
        // Forward computes the auxiliary loss against the bank retrieval
        // before inject mode overwrites s.mix with s.lss_recon. Backward must
        // use that same frozen target; comparing against overwritten s.mix
        // makes recon - mix exactly zero and silently disables aux learning.
        const float* lss_target =
            (s.lss_injected && s.mix_bank_save) ? s.mix_bank_save : s.mix;
        k_lss_drecon_aux<<<gd(bh), kTpb, 0, st>>>(
            s.lss_recon, lss_target, s.d_lss_recon, p.lss_aux_weight, B, H);
        lrss_debug_print("d_recon", false, -1, s.d_lss_recon, bh, st);
        // dW_up += d_reconᵀ·hidden ; d_hidden_lss = d_recon·W_up
        k_lrss_dweight<<<gd(static_cast<std::size_t>(H) * R), kTpb, 0, st>>>(
            s.d_lss_recon, s.lss_hidden, g.lss_up, B, R, H);
        k_lrss_dinput<<<gd(static_cast<std::size_t>(B) * R), kTpb, 0, st>>>(
            s.d_lss_recon, p.lss_up, s.d_lss_hidden, B, R, H);
        k_lss_relu_bwd<<<gd(static_cast<std::size_t>(B) * R), kTpb, 0, st>>>(
            s.lss_hidden, s.d_lss_hidden, static_cast<std::size_t>(B) * R);
        // joint_dim mirrors the forward computation exactly (fixed per-run
        // width; see the forward comment above).
        const int joint_dim = 2 * H + (s.pss_spike ? p.pss_spike_dim : 0);
        // dW_down += d_hiddenᵀ·joint ; d_joint = d_hidden·W_down
        k_lrss_dweight<<<gd(static_cast<std::size_t>(R) * joint_dim), kTpb, 0, st>>>(
            s.d_lss_hidden, s.lss_joint, g.lss_down, B, joint_dim, R);
        // reuse dcat as d_joint scratch (allocated B*joint_dim -- see
        // alloc_lrss_scratch), then fold the pooled half.
        k_lrss_dinput<<<gd(static_cast<std::size_t>(B) * joint_dim), kTpb, 0, st>>>(
            s.d_lss_hidden, p.lss_down, s.dcat, B, joint_dim, R);
        // Deliberately still launched at the FIXED literal 2*H, not
        // joint_dim: this only ever splits the pooled/anchor halves out of
        // dcat's first 2*H columns. When spike_dim > 0, dcat is wider than
        // 2*H but the spike-gradient tail (columns [2H, joint_dim)) is never
        // read here -- an automatic dead end, the same "no kernel needed"
        // pattern the anchor half already relies on below.
        k_lrss_split<<<gd(static_cast<std::size_t>(B) * 2 * H), kTpb, 0, st>>>(
            s.dcat, s.d_pooled, s.d_mix, B, H);   // anchor half lands in d_mix scratch — dead end by design
        lrss_debug_print("d_pooled(post-lss)", false, -1, s.d_pooled, bh, st);
    }

    // scatter through the masked mean
    k_lrss_pool_scatter<<<gd(static_cast<std::size_t>(B) * S * H), kTpb, 0, st>>>(
        d_hidden, s.d_pooled, tokens, common_mask, s.n_content, B, S, H);

    // LSS telemetry: the aux reconstruction loss IS the "does LSS work"
    // curve — surface it every 16 micro-steps (one tiny sync, negligible
    // against the micro-step).
    if (p.lss_rank > 0 && s.lss_active_this_step) {
        static int lss_probe = 0;
        if ((lss_probe++ % 16) == 0) {
            float aux = 0.0f;
            cudaStreamSynchronize(st);
            cudaMemcpy(&aux, s.lss_aux, sizeof(float), cudaMemcpyDeviceToHost);
            std::fprintf(stderr, "[lss] micro=%d aux_recon_loss=%.6g\n",
                         lss_probe, aux);
        }
    }
}

// tiny helper: y += x
namespace {
__global__ void k_lrss_add(float* y, const float* x, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
}  // namespace
void k_lrss_split_add(float* y, const float* x, std::size_t n, cudaStream_t st) {
    k_lrss_add<<<gd(n), kTpb, 0, st>>>(y, x, n);
}

// Rebuild the device bank view (newest-first) + elapsed from the ring.
// Host-side ring bookkeeping; copies are tiny ([A,H] f32 D2D + [A] H2D).
void lrss_refresh_bank(LrssScratch& s, int max_anchors, int H, int now, cudaStream_t st) {
    const int A = s.count;
    s.bank_count = A;
    if (A <= 0) return;
    float elapsed_h[64];
    for (int i = 0; i < A; ++i) {
        // newest-first: slot (cursor-1-i) mod max
        const int slot = ((s.cursor - 1 - i) % max_anchors + max_anchors) % max_anchors;
        cudaMemcpyAsync(s.bank + static_cast<std::size_t>(i) * H,
                        s.ring + static_cast<std::size_t>(slot) * H,
                        static_cast<std::size_t>(H) * sizeof(float),
                        cudaMemcpyDeviceToDevice, st);
        const float e = static_cast<float>(now - s.times[slot]);
        elapsed_h[i] = e < 1.0f ? 1.0f : e;
    }
    cudaMemcpyAsync(s.elapsed, elapsed_h, static_cast<std::size_t>(A) * sizeof(float),
                    cudaMemcpyHostToDevice, st);
    cudaStreamSynchronize(st);  // elapsed_h is stack memory — must land before return
}

}  // namespace ida_native
