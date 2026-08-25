#include "ida_native/kernels.hpp"
#include "ida_native/backend_contract.hpp"

#include <stdexcept>
#include <string>
#include <string_view>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Flash-style causal attention: online-softmax forward with saved LSE,
// recompute backward.  Never materializes the [S, S] score matrix.
//
// Tiled implementation with DYNAMIC shared memory: each block owns BR rows
// (queries in fwd/dq, keys in dkv) and streams BC opposing rows per tile.
// BR/BC shrink as Hd grows so the tiles fit the H100's 227 KB shared budget:
//   Hd ≤  64 (swift/edge): BR=16, BC=32
//   Hd ≤ 128:              BR=16, BC=16
//   Hd ≤ 256 (ai):         BR= 8, BC=16
//   Hd ≤ 512 (moe):        BR= 8, BC= 8
//
// Layout for all tensors: [BH, S, Hd] FP32 row-major, BH = B * nH.
// Shared rows are padded to Hd+1 to kill bank conflicts.

namespace ida_native {

AttentionBackendKind parse_attention_backend(std::string_view backend) {
    if (backend.empty() || backend == "scalar_flash") {
        return AttentionBackendKind::ScalarFlash;
    }
    if (backend == "hopper_wgmma_packed_fp4") {
        return AttentionBackendKind::HopperWgmmaPackedFp4;
    }
    if (backend == "hopper_wgmma_fp8") {
        return AttentionBackendKind::HopperWgmmaFp8;
    }
    if (backend == kBlackwellMxf4Fp4Backend) {
        return AttentionBackendKind::BlackwellMxf4Fp4;
    }
    throw std::runtime_error(
        "unknown native attention backend: " + std::string(backend));
}

const char* attention_backend_name(AttentionBackendKind backend) {
    switch (backend) {
        case AttentionBackendKind::ScalarFlash:
            return "scalar_flash";
        case AttentionBackendKind::HopperWgmmaPackedFp4:
            return "hopper_wgmma_packed_fp4";
        case AttentionBackendKind::HopperWgmmaFp8:
            return "hopper_wgmma_fp8";
        case AttentionBackendKind::BlackwellMxf4Fp4:
            return kBlackwellMxf4Fp4Backend.data();
        default:
            return "unknown";
    }
}

static constexpr int kMaxHdSupported = 512;
static constexpr int kHopperPackedFp4SeqLen = 2048;
static constexpr int kHopperPackedFp4HeadDim = 64;
static constexpr int kHopperWgmmaTileM = 64;
static constexpr int kHopperWgmmaTileN = 64;
static constexpr int kHopperWgmmaTileK = 32;
static constexpr int kHopperTmaStageDepth = 2;

enum class PackedFp4BackwardReplayMode : int {
    Bf16Roundtrip = 0,
    PackedE4m3 = 1,
    SavedE4m3 = 2,
};

static PackedFp4BackwardReplayMode packed_fp4_backward_replay_mode() {
    // Default packed_e4m3 — best replay-symmetry result in the July 5 32/8
    // matrix (final GN 170.7 vs roundtrip's 314.0), verified default
    // 2026-07-08.  Set IDA_NATIVE_PACKED_FP4_BWD_REPLAY=roundtrip (or any
    // other value) for ablation.
    const char* e = std::getenv("IDA_NATIVE_PACKED_FP4_BWD_REPLAY");
    if (!e || !e[0]) {
        return PackedFp4BackwardReplayMode::PackedE4m3;
    }
    const std::string_view mode(e);
    if (mode == "packed_e4m3") {
        return PackedFp4BackwardReplayMode::PackedE4m3;
    }
    if (mode == "saved_e4m3") {
        return PackedFp4BackwardReplayMode::SavedE4m3;
    }
    return PackedFp4BackwardReplayMode::Bf16Roundtrip;
}

static void attn_tile_dims(int Hd, int& BR, int& BC) {
    if      (Hd <= 64)  { BR = 16; BC = 32; }
    else if (Hd <= 128) { BR = 16; BC = 16; }
    else if (Hd <= 256) { BR = 8;  BC = 16; }
    else                { BR = 8;  BC = 8;  }
}

// Per-kernel tile-size override for empirical sweeps (2026-07-28 MoE
// attention-tile investigation). BR/BC are pure runtime kernel arguments
// (verified by reading all three flash-attention kernel bodies: no
// compile-time dependence anywhere), so sweeping candidates costs nothing
// but env vars and a burn -- no rebuild needed between candidates. Only
// meaningful for Hd>256 today (MoE-exclusive; Edge=64/AI=256/Swift=32 never
// reach here in production). Unset envs leave BR/BC at whatever the caller
// already computed (attn_tile_dims' default, or a kernel-specific default
// the caller applied first).
static void attn_tile_override(const char* br_env, const char* bc_env, int& BR, int& BC) {
    if (const char* e = std::getenv(br_env)) {
        const int v = std::atoi(e);
        if (v > 0) BR = v;
    }
    if (const char* e = std::getenv(bc_env)) {
        const int v = std::atoi(e);
        if (v > 0) BC = v;
    }
}

static void validate_packed_fp4_operands(
    const PackedFp4AttentionOperands* packed_fp4,
    int BH,
    int S,
    int Hd
) {
    if (packed_fp4 == nullptr) {
        throw std::runtime_error(
            "attention backend hopper_wgmma_packed_fp4 requires packed_fp4 operands");
    }
    if (packed_fp4->qk_packed == nullptr) {
        throw std::runtime_error(
            "attention backend hopper_wgmma_packed_fp4 requires qk_packed staging bytes");
    }
    if (packed_fp4->q_scale == nullptr || packed_fp4->q_descale == nullptr ||
        packed_fp4->k_scale == nullptr || packed_fp4->k_descale == nullptr) {
        throw std::runtime_error(
            "attention backend hopper_wgmma_packed_fp4 requires q/k scale metadata");
    }
    if (packed_fp4->q_unpack_f32 == nullptr || packed_fp4->k_unpack_f32 == nullptr) {
        throw std::runtime_error(
            "attention backend hopper_wgmma_packed_fp4 requires unpack scratch buffers");
    }
    if (packed_fp4->q_tile_stage_f32 == nullptr ||
        packed_fp4->k_tile_stage_f32 == nullptr ||
        packed_fp4->v_tile_stage_f32 == nullptr) {
        throw std::runtime_error(
            "attention backend hopper_wgmma_packed_fp4 requires TMA tile stage scratch");
    }
    if (packed_fp4->tma_stage_depth < kHopperTmaStageDepth ||
        packed_fp4->tile_stage_elems < static_cast<std::size_t>(kHopperTmaStageDepth) *
                                       kHopperWgmmaTileM * kHopperPackedFp4HeadDim) {
        throw std::runtime_error(
            "attention backend hopper_wgmma_packed_fp4 requires double-buffered TMA stage capacity");
    }
    const std::size_t expected = static_cast<std::size_t>(BH) * S * Hd;
    if (packed_fp4->packed_elems != expected) {
        throw std::runtime_error(
            "attention backend hopper_wgmma_packed_fp4 received a packed_fp4 buffer with the wrong shape");
    }
}

struct HopperPackedFp4LaunchShape {
    int BH{0};
    int S{0};
    int Hd{0};
    int tile_m{kHopperWgmmaTileM};
    int tile_n{kHopperWgmmaTileN};
    int tile_k{kHopperWgmmaTileK};
    int warpgroup_threads{128};
};

static HopperPackedFp4LaunchShape build_hopper_packed_fp4_launch_shape(
    int BH,
    int S,
    int Hd
) {
    if (BH <= 0) {
        throw std::runtime_error(
            "hopper_wgmma_packed_fp4 attention backend requires BH > 0");
    }
    if (S != kHopperPackedFp4SeqLen) {
        throw std::runtime_error(
            "hopper_wgmma_packed_fp4 forward path currently requires sequence_length=2048");
    }
    if (Hd != 64 && Hd != 256) {
        throw std::runtime_error(
            "hopper_wgmma_packed_fp4 forward path supports head_dim 64 (edge) and 256 (ai)");
    }
    HopperPackedFp4LaunchShape shape{};
    shape.BH = BH;
    shape.S = S;
    shape.Hd = Hd;
    return shape;
}

// Forward declarations of the public scalar-flash entry points defined below.
void flash_attn_forward(
    const __nv_bfloat16* d_q, const __nv_bfloat16* d_k, const __nv_bfloat16* d_v,
    __nv_bfloat16* d_o, float* d_lse,
    const std::uint16_t* d_segs, int nH,
    int BH, int S, int Hd, float scale,
    cudaStream_t stream);
void flash_attn_backward(
    const __nv_bfloat16* d_q, const __nv_bfloat16* d_k, const __nv_bfloat16* d_v,
    const __nv_bfloat16* d_o, const __nv_bfloat16* d_do, const float* d_lse,
    __nv_bfloat16* d_dq, __nv_bfloat16* d_dk, __nv_bfloat16* d_dv,
    float* d_rowdot,
    const std::uint16_t* d_segs, int nH,
    int BH, int S, int Hd, float scale,
    cudaStream_t stream, bool skip_dkv = false, bool skip_dq = false);

// Advanced packed-attention orchestration is deployment-owned. The public
// binary keeps compatibility entry points but never selects private backend
// code from environment state.
static bool wgmma_compute_enabled() {
    const char* e = std::getenv("IDA_NATIVE_WGMMA");
    return !(e && std::string_view(e) == "0");
}

static bool tma_stage_probe_enabled() {
    // Default ON (mode 1) — verified default 2026-07-08.  IDA_NATIVE_TMA=0
    // disables; =2 selects the in-kernel cp.async pipeline instead.
    const char* e = std::getenv("IDA_NATIVE_TMA");
    if (!e || !e[0]) return true;
    return std::string_view(e) == "1";
}

__global__ void k_stage_first_tma_tiles_probe(
    const __nv_bfloat16* __restrict__ q_unpack,
    const __nv_bfloat16* __restrict__ k_unpack,
    const __nv_bfloat16* __restrict__ v,
    float* __restrict__ q_stage,
    float* __restrict__ k_stage,
    float* __restrict__ v_stage
) {
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int tile_elems = kHopperWgmmaTileM * kHopperPackedFp4HeadDim;
    if (idx >= tile_elems) {
        return;
    }
    const int row = idx / kHopperPackedFp4HeadDim;
    const int col = idx % kHopperPackedFp4HeadDim;
    q_stage[idx] = __bfloat162float(q_unpack[idx]);
    k_stage[idx] = __bfloat162float(k_unpack[idx]);
    v_stage[col * kHopperWgmmaTileM + row] = __bfloat162float(v[idx]);
}

static void hopper_tma_stage_first_tiles_probe(
    const PackedFp4AttentionOperands* packed_fp4,
    const __nv_bfloat16* d_v,
    cudaStream_t stream
) {
    if (!tma_stage_probe_enabled()) {
        return;
    }
    k_stage_first_tma_tiles_probe<<<16, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(packed_fp4->q_unpack_f32),
        reinterpret_cast<const __nv_bfloat16*>(packed_fp4->k_unpack_f32),
        d_v,
        packed_fp4->q_tile_stage_f32,
        packed_fp4->k_tile_stage_f32,
        packed_fp4->v_tile_stage_f32
    );
}

// ─── WGMMA↔scalar forward parity probe ────────────────────────────────────────
// IDA_NATIVE_ATTN_PARITY=1: after the WGMMA forward, re-decode the same packed
// surface and run the scalar-flash reference into scratch, then print
// max|ΔO| / max|ΔLSE|.  This is the device validation the Hd=256 extension
// armed but never ran — a stride/layout defect in the WGMMA path shows up here
// in one call, and any rebuild must drive both deltas to rounding level.
static bool attn_parity_probe_enabled() {
    const char* e = std::getenv("IDA_NATIVE_ATTN_PARITY");
    return e && e[0] == '1';
}

// ─── dS cancellation diagnostic (IDA_NATIVE_ATTN_PARITY=1) ─────────────────────
// dS = P*(dp - D) is meant to cancel: D_i is defined so that
// Sum_j P_ij*dp_ij = D_i exactly, by construction from O_i = Sum_j P_ij V_j.
// dp/D are computed via two INDEPENDENT paths (this kernel's plain
// summation loop for dp vs k_attn_rowdot's reduction, fed by O_self from
// flash_attn_forward's online-softmax tiling) — if the two paths' implicit
// P values don't agree bit-for-bit at massive-activation magnitude
// (|s|~57000), the cancellation doesn't fully cancel and ds inherits a
// residual scaling with dp/D's own magnitude, not the mismatch itself.
// Three independent maxes (not a single argmax — plain magnitude, no
// ordering needed): large dp/D with small ds = cancellation working
// (blowup is elsewhere, e.g. summed over many keys into dK); large dp/D
// AND large ds = cancellation itself failing.  Found while root-causing
// the L0.attn_norm/k_proj/q_proj explosion at mb=128 AI, 2026-07-09.
__device__ float g_cancel_dp_max = 0.0f;
__device__ float g_cancel_D_max  = 0.0f;
__device__ float g_cancel_ds_max = 0.0f;

__device__ inline void cancel_probe_accum(float dp, float D, float ds) {
    atomicMax(reinterpret_cast<int*>(&g_cancel_dp_max), __float_as_int(fabsf(dp)));
    atomicMax(reinterpret_cast<int*>(&g_cancel_D_max),  __float_as_int(fabsf(D)));
    atomicMax(reinterpret_cast<int*>(&g_cancel_ds_max), __float_as_int(fabsf(ds)));
}

void cancel_probe_reset(cudaStream_t s) {
    static const float zero = 0.0f;
    cudaMemcpyToSymbolAsync(g_cancel_dp_max, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice, s);
    cudaMemcpyToSymbolAsync(g_cancel_D_max,  &zero, sizeof(zero), 0, cudaMemcpyHostToDevice, s);
    cudaMemcpyToSymbolAsync(g_cancel_ds_max, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice, s);
}

void cancel_probe_read(float* dp_max, float* D_max, float* ds_max, cudaStream_t s) {
    cudaMemcpyFromSymbolAsync(dp_max, g_cancel_dp_max, sizeof(float), 0, cudaMemcpyDeviceToHost, s);
    cudaMemcpyFromSymbolAsync(D_max,  g_cancel_D_max,  sizeof(float), 0, cudaMemcpyDeviceToHost, s);
    cudaMemcpyFromSymbolAsync(ds_max, g_cancel_ds_max, sizeof(float), 0, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
}

// ─── Backward surface-health telemetry (scalar path) ────────────────────────
// The public runtime reports only bounded aggregate health values. Private
// backend counters are not part of the public implementation.
constexpr float kAttnPScaleProbe = 448.0f;

__device__ unsigned long long g_attn_p_zero_scalar  = 0ull;
__device__ unsigned long long g_attn_p_total_scalar = 0ull;
__device__ unsigned int g_attn_drift_key_scalar = 0u;

__device__ inline unsigned int float_to_radix_key_scalar(float f) {
    const unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ inline void attn_health_accum_scalar(
    unsigned p_zero, unsigned p_valid, float drift
) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        p_zero  += __shfl_xor_sync(0xffffffff, p_zero,  off);
        p_valid += __shfl_xor_sync(0xffffffff, p_valid, off);
        drift    = fmaxf(drift, __shfl_xor_sync(0xffffffff, drift, off));
    }
    if ((threadIdx.x & 31) == 0) {
        if (p_zero)  atomicAdd(&g_attn_p_zero_scalar,
                               static_cast<unsigned long long>(p_zero));
        if (p_valid) {
            atomicAdd(&g_attn_p_total_scalar,
                      static_cast<unsigned long long>(p_valid));
            atomicMax(&g_attn_drift_key_scalar, float_to_radix_key_scalar(drift));
        }
    }
}

void attn_bwd_health_reset_scalar(cudaStream_t stream) {
    static const unsigned long long zero64 = 0ull;
    static const unsigned int zero32 = 0u;
    cudaMemcpyToSymbolAsync(g_attn_p_zero_scalar, &zero64, sizeof(zero64), 0,
                            cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(g_attn_p_total_scalar, &zero64, sizeof(zero64), 0,
                            cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(g_attn_drift_key_scalar, &zero32, sizeof(zero32), 0,
                            cudaMemcpyHostToDevice, stream);
}

void attn_bwd_health_read_raw_scalar(
    unsigned long long* p_zero, unsigned long long* p_total,
    unsigned int* drift_key, cudaStream_t stream
) {
    cudaMemcpyFromSymbolAsync(p_zero, g_attn_p_zero_scalar, sizeof(*p_zero), 0,
                              cudaMemcpyDeviceToHost, stream);
    cudaMemcpyFromSymbolAsync(p_total, g_attn_p_total_scalar, sizeof(*p_total), 0,
                              cudaMemcpyDeviceToHost, stream);
    cudaMemcpyFromSymbolAsync(drift_key, g_attn_drift_key_scalar, sizeof(*drift_key), 0,
                              cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

__global__ void k_absdiff_max_bf16(
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
    std::size_t n, float* __restrict__ out   // out[0]=max|Δ|, out[1]=max|a|, out[2]+=nonfinite(a)
) {
    float d = 0.0f, m = 0.0f;
    unsigned nf = 0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float av = __bfloat162float(a[i]);
        const float bv = __bfloat162float(b[i]);
        if (!isfinite(av)) ++nf;
        d = fmaxf(d, fabsf(av - bv));
        m = fmaxf(m, fabsf(av));
    }
    for (int off = 16; off > 0; off >>= 1) {
        d = fmaxf(d, __shfl_xor_sync(0xffffffff, d, off));
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
        nf += __shfl_xor_sync(0xffffffff, nf, off);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicMax(reinterpret_cast<int*>(&out[0]), __float_as_int(d));
        atomicMax(reinterpret_cast<int*>(&out[1]), __float_as_int(m));
        atomicAdd(&out[2], static_cast<float>(nf));
    }
}

__global__ void k_absdiff_max_f32(
    const float* __restrict__ a, const float* __restrict__ b,
    std::size_t n, float* __restrict__ out
) {
    float d = 0.0f, m = 0.0f;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        d = fmaxf(d, fabsf(a[i] - b[i]));
        m = fmaxf(m, fabsf(a[i]));
    }
    for (int off = 16; off > 0; off >>= 1) {
        d = fmaxf(d, __shfl_xor_sync(0xffffffff, d, off));
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    }
    if ((threadIdx.x & 31) == 0) {
        atomicMax(reinterpret_cast<int*>(&out[0]), __float_as_int(d));
        atomicMax(reinterpret_cast<int*>(&out[1]), __float_as_int(m));
    }
}

static void wgmma_parity_probe(
    const PackedFp4AttentionOperands* packed_fp4,
    const __nv_bfloat16* d_v,
    const __nv_bfloat16* d_o_wgmma,
    const float* d_lse_wgmma,
    int BH, int S, int Hd, float scale,
    cudaStream_t stream
) {
    if (!attn_parity_probe_enabled()) return;
    static int probes_left = 20;
    if (probes_left <= 0) return;
    --probes_left;

    const std::size_t n = static_cast<std::size_t>(BH) * S * Hd;
    const std::size_t rows = static_cast<std::size_t>(BH) * S;
    __nv_bfloat16 *q_ref = nullptr, *k_ref = nullptr, *o_ref = nullptr;
    float *lse_ref = nullptr, *stats = nullptr;
    cudaMallocAsync(&q_ref, n * sizeof(__nv_bfloat16), stream);
    cudaMallocAsync(&k_ref, n * sizeof(__nv_bfloat16), stream);
    cudaMallocAsync(&o_ref, n * sizeof(__nv_bfloat16), stream);
    cudaMallocAsync(&lse_ref, rows * sizeof(float), stream);
    cudaMallocAsync(&stats, 4 * sizeof(float), stream);
    if (!q_ref || !k_ref || !o_ref || !lse_ref || !stats) {
        std::fprintf(stderr, "[attn-parity] scratch alloc failed — probe skipped\n");
        return;
    }
    cudaMemsetAsync(stats, 0, 4 * sizeof(float), stream);

    // Decode into private scratch (the WGMMA path reused the shared unpack
    // buffers as e4m3 byte storage — do not touch them), then roundtrip the
    // reference operands through e4m3 so both paths consume the same
    // quantized surface — otherwise the probe measures decode-path drift on
    // top of kernel error.
    fp4_unpack_pair_to_bf16(
        packed_fp4->qk_packed, q_ref, k_ref,
        packed_fp4->q_descale, packed_fp4->k_descale,
        packed_fp4->packed_elems, stream,
        packed_fp4->q_mean, packed_fp4->k_mean);
    roundtrip_bf16_through_e4m3(q_ref, n, stream);
    roundtrip_bf16_through_e4m3(k_ref, n, stream);
    flash_attn_forward(q_ref, k_ref, d_v, o_ref, lse_ref,
                       nullptr, 1, BH, S, Hd, scale, stream);

    k_absdiff_max_bf16<<<256, 256, 0, stream>>>(d_o_wgmma, o_ref, n, stats);
    k_absdiff_max_f32 <<<256, 256, 0, stream>>>(d_lse_wgmma, lse_ref, rows, stats + 2);

    float h[4] = {0, 0, 0, 0};
    cudaStreamSynchronize(stream);
    cudaMemcpy(h, stats, sizeof(h), cudaMemcpyDeviceToHost);
    std::fprintf(stderr,
        "[attn-parity] Hd=%d BH=%d  max|dO|=%.6g (|O|max=%.6g)  "
        "max|dLSE|=%.6g (|LSE|max=%.6g)\n",
        Hd, BH, h[0], h[1], h[2], h[3]);

    cudaFreeAsync(q_ref, stream); cudaFreeAsync(k_ref, stream);
    cudaFreeAsync(o_ref, stream); cudaFreeAsync(lse_ref, stream);
    cudaFreeAsync(stats, stream);
}

static void hopper_packed_fp4_attention_forward(
    const PackedFp4AttentionOperands* packed_fp4,
    const __nv_bfloat16* d_v,
    __nv_bfloat16* d_o,
    float* d_lse,
    const std::uint16_t* d_segs, int nH,
    int BH,
    int S,
    int Hd,
    float scale,
    cudaStream_t stream
) {
    validate_packed_fp4_operands(packed_fp4, BH, S, Hd);
    const auto shape = build_hopper_packed_fp4_launch_shape(BH, S, Hd);
    (void)shape;
    if (!std::isfinite(scale) || scale <= 0.0f) {
        throw std::runtime_error(
            "hopper_wgmma_packed_fp4 forward path requires a finite positive softmax scale");
    }
    if (wgmma_compute_enabled() && (Hd == 64 || Hd == 256) && (S % 64) == 0) {
        // Private deployment path; public builds fail closed before this call.
        wgmma_flash_forward_hd64(
            packed_fp4->qk_packed,
            packed_fp4->q_descale,
            packed_fp4->k_descale,
            packed_fp4->q_unpack_f32,
            packed_fp4->k_unpack_f32,
            packed_fp4->q_tile_stage_f32,
            packed_fp4->k_tile_stage_f32,
            packed_fp4->v_tile_stage_f32,
            packed_fp4->tma_stage_depth,
            d_v, d_o, d_lse, BH, S, Hd, scale, stream,
            packed_fp4->q_saved_e4m3,
            packed_fp4->k_saved_e4m3,
            packed_fp4->q_mean, packed_fp4->k_mean,
            d_segs, nH);
        wgmma_parity_probe(packed_fp4, d_v, d_o, d_lse, BH, S, Hd, scale, stream);
        return;
    }
    // Reference path: decode → scalar flash (IDA_NATIVE_WGMMA=0).
    fp4_unpack_pair_to_bf16(
        packed_fp4->qk_packed,
        reinterpret_cast<__nv_bfloat16*>(packed_fp4->q_unpack_f32),
        reinterpret_cast<__nv_bfloat16*>(packed_fp4->k_unpack_f32),
        packed_fp4->q_descale,
        packed_fp4->k_descale,
        packed_fp4->packed_elems,
        stream,
        packed_fp4->q_mean,
        packed_fp4->k_mean
    );
    hopper_tma_stage_first_tiles_probe(packed_fp4, d_v, stream);
    flash_attn_forward(
        reinterpret_cast<const __nv_bfloat16*>(packed_fp4->q_unpack_f32),
        reinterpret_cast<const __nv_bfloat16*>(packed_fp4->k_unpack_f32), d_v,
        d_o, d_lse, d_segs, nH, BH, S, Hd, scale, stream);
}

// Backward: scalar flash over freshly re-unpacked f32 operands.  The WGMMA
// forward reuses the unpack scratch as e4m3 byte storage, so the f32 decode
// must be re-run here regardless of which forward compute path executed.
// dQ/dK are gradients w.r.t. the dequantized operands (correct for the
// quantized forward; the pack kernel's scale chain handles the rest).
static void hopper_packed_fp4_attention_backward(
    const PackedFp4AttentionOperands* packed_fp4,
    const __nv_bfloat16* d_v,
    const __nv_bfloat16* d_o,
    const __nv_bfloat16* d_do,
    const float* d_lse,
    __nv_bfloat16* d_dq,
    __nv_bfloat16* d_dk,
    __nv_bfloat16* d_dv,
    float* d_rowdot,
    const std::uint16_t* d_segs, int nH,
    int BH,
    int S,
    int Hd,
    float scale,
    cudaStream_t stream
) {
    validate_packed_fp4_operands(packed_fp4, BH, S, Hd);
    const auto shape = build_hopper_packed_fp4_launch_shape(BH, S, Hd);
    (void)shape;
    if (!std::isfinite(scale) || scale <= 0.0f) {
        throw std::runtime_error(
            "hopper_wgmma_packed_fp4 backward path requires a finite positive softmax scale");
    }
    const auto replay_mode = packed_fp4_backward_replay_mode();
    auto* q_bwd = reinterpret_cast<__nv_bfloat16*>(packed_fp4->q_unpack_f32);
    auto* k_bwd = reinterpret_cast<__nv_bfloat16*>(packed_fp4->k_unpack_f32);
    if (replay_mode == PackedFp4BackwardReplayMode::SavedE4m3) {
        if (packed_fp4->q_saved_e4m3 == nullptr || packed_fp4->k_saved_e4m3 == nullptr) {
            throw std::runtime_error(
                "IDA_NATIVE_PACKED_FP4_BWD_REPLAY=saved_e4m3 requires saved forward q8/k8 buffers");
        }
        fp8_e4m3_pair_to_bf16(
            packed_fp4->q_saved_e4m3,
            packed_fp4->k_saved_e4m3,
            q_bwd,
            k_bwd,
            packed_fp4->packed_elems,
            stream);
    } else if (replay_mode == PackedFp4BackwardReplayMode::PackedE4m3) {
        fp4_unpack_pair_to_bf16_via_e4m3(
            packed_fp4->qk_packed,
            q_bwd,
            k_bwd,
            packed_fp4->q_descale,
            packed_fp4->k_descale,
            packed_fp4->packed_elems,
            stream,
            packed_fp4->q_mean,
            packed_fp4->k_mean);
    } else {
        fp4_unpack_pair_to_bf16(
            packed_fp4->qk_packed,
            q_bwd,
            k_bwd,
            packed_fp4->q_descale,
            packed_fp4->k_descale,
            packed_fp4->packed_elems,
            stream,
            packed_fp4->q_mean,
            packed_fp4->k_mean
        );
        if (wgmma_compute_enabled() && (Hd == 64 || Hd == 256) && (S % 64) == 0) {
            // The WGMMA forward computed on e4m3-rounded operands; the backward
            // must see identical rounding so exp(s − lse) stays bounded.
            roundtrip_bf16_through_e4m3(q_bwd, packed_fp4->packed_elems, stream);
            roundtrip_bf16_through_e4m3(k_bwd, packed_fp4->packed_elems, stream);
        }
    }
    // Parity probe (IDA_NATIVE_ATTN_PARITY=1): (a) how far the backward's
    // re-decoded operands drift from the surface the forward's stored LSE was
    // computed on — exp(s_bwd − lse_fwd) > 1 is the P-inflation detonator;
    // (b) raw magnitudes of the produced gradients — attention-sink columns
    // can push dV column sums toward the BF16 ceiling legitimately.
    if (attn_parity_probe_enabled()) {
        static int bwd_probes_left = 40;
        if (bwd_probes_left > 0) {
            --bwd_probes_left;
            const std::size_t n = static_cast<std::size_t>(BH) * S * Hd;
            const std::size_t rows = static_cast<std::size_t>(BH) * S;
            __nv_bfloat16* o_tmp = nullptr; float *lse_tmp = nullptr, *st = nullptr;
            cudaMallocAsync(&o_tmp, n * sizeof(__nv_bfloat16), stream);
            cudaMallocAsync(&lse_tmp, rows * sizeof(float), stream);
            cudaMallocAsync(&st, 2 * sizeof(float), stream);
            if (o_tmp && lse_tmp && st) {
                cudaMemsetAsync(st, 0, 2 * sizeof(float), stream);
                flash_attn_forward(q_bwd, k_bwd, d_v, o_tmp, lse_tmp,
                                   nullptr, 1, BH, S, Hd, scale, stream);
                k_absdiff_max_f32<<<256, 256, 0, stream>>>(d_lse, lse_tmp, rows, st);
                float h[2] = {0, 0};
                cudaStreamSynchronize(stream);
                cudaMemcpy(h, st, sizeof(h), cudaMemcpyDeviceToHost);
                std::fprintf(stderr,
                    "[attn-parity-bwd] Hd=%d  max|lse_fwd - lse(bwd_operands)|=%.6g (|lse|max=%.6g)\n",
                    Hd, h[0], h[1]);
            }
            cudaFreeAsync(o_tmp, stream); cudaFreeAsync(lse_tmp, stream);
            cudaFreeAsync(st, stream);
        }
    }

    // Backward LSE self-consistency: recompute the log-sum-exp from the exact
    // operands this backward consumes, instead of trusting the forward's
    // stored LSE to match a reconstructed surface.  At AI-scale score
    // magnitudes (|s| in the tens of thousands), even 0.3% surface drift
    // makes exp(s − lse_stored) overflow through the e30 clamp into inf dV;
    // deriving lse from the replay surface bounds P ≤ ~1 by construction, at
    // any head dim or magnitude.  Costs one extra flash-forward per layer
    // (~21% at Edge dims, ~4% at AI dims), so the default is head-dim gated:
    // Hd ≥ 256 (AI/MoE — where the magnitudes demand it) on, Edge off (its
    // drift is ≤2 at |s|≈40–70, absorbed by months of verified clean runs).
    // IDA_NATIVE_ATTN_SELF_LSE=1 forces on everywhere, =0 forces off.
    // Era 12 Stage A: WGMMA dK/dV (Hd 64/256, env-gated OFF by default until
    // the parity gate passes).  dQ + rowdot stay scalar via skip_dkv.
    // Era 13 default: the de-chunked WGMMA backward is the verified winner at
    // BOTH supported head dims (Edge +60%, AI +35% over scalar, gradient
    // parity to 4-6 decimals) — compiled-in ON.  IDA_NATIVE_WGMMA_BWD=0
    // restores the scalar backward (permanent reference implementation);
    // =2 keeps the diagnostic mode.  Same for the dQ stage.
    const char* wb = std::getenv("IDA_NATIVE_WGMMA_BWD");
    const char* wdq = std::getenv("IDA_NATIVE_WGMMA_BWD_DQ");
    const bool wgmma_dq_env = !wdq || wdq[0] == '1';
    bool wgmma_bwd = (!wb || wb[0] == '1' || wb[0] == '2')
        && (Hd == 64 || Hd == 256) && (S % 64) == 0 && wgmma_compute_enabled();
    // Hd>=256 demands full Stage B (WGMMA dQ too): a scalar dQ in the mix
    // would need a SECOND, scalar-consistent lse surface next to the WGMMA
    // one — two-surface plumbing isn't supported.  Without the dq flag the
    // scalar path below (with its Era 11 self-LSE) owns these dims.
    if (wgmma_bwd && Hd >= 256 && !wgmma_dq_env) wgmma_bwd = false;
    if (wgmma_bwd) {
        // Surface contract (root-caused 2026-07-09): the backward lse must
        // come from the same COMPUTATION that produces the backward's s.
        // The WGMMA kernels compute S in forward orientation on the same
        // e4m3 codes as the forward — the STORED d_lse/d_o ARE their
        // consistent surface (bit-identical score computation).  A scalar
        // self-LSE here sits systematically ABOVE the WGMMA s (FP8 tensor
        // cores accumulate with ~13-bit effective mantissa, truncation
        // biased low): past |s·scale|≈300 the deficit underflows every e4m3
        // P byte → whole-tensor zero gradients that a clean-looking grad
        // norm actively hides.  No self-LSE pass in this branch, ever.
        const __nv_bfloat16* o_used = d_o;
        const float* lse_used = d_lse;
        const __nv_bfloat16* q_used = q_bwd;
        const __nv_bfloat16* k_used = k_bwd;
        // Mode 2 (diagnostic): real gradients from the full scalar backward;
        // the WGMMA kernel runs into discarded scratch so its side effects
        // (allocs, quant passes) still occur.  Isolates value-bugs from
        // side-effect bugs.
        const bool diag_scalar_grads = (wb && wb[0] == '2');
        // Stage B: dQ via WGMMA too (scalar backward then runs rowdot-only).
        // Separate flag so dq/dkv gate independently through their own parity
        // runs; default OFF until the Stage B parity + GN band pass.
        const bool wgmma_dq = wgmma_dq_env && !diag_scalar_grads;
        flash_attn_backward(
            q_used, k_used, d_v, o_used, d_do, lse_used, d_dq, d_dk, d_dv, d_rowdot,
            d_segs, nH, BH, S, Hd, scale, stream,
            /*skip_dkv=*/!diag_scalar_grads, /*skip_dq=*/wgmma_dq);
        if (wgmma_dq) {
            wgmma_flash_bwd_dq(
                q_used, k_used, d_v, d_do, lse_used, d_rowdot,
                d_segs, nH, d_dq, BH, S, Hd, scale, stream);
        }
        __nv_bfloat16 *dk_t = d_dk, *dv_t = d_dv;
        __nv_bfloat16 *dk_scratch = nullptr, *dv_scratch = nullptr;
        if (diag_scalar_grads) {
            const std::size_t n = static_cast<std::size_t>(BH) * S * Hd;
            cudaMallocAsync(&dk_scratch, n * sizeof(__nv_bfloat16), stream);
            cudaMallocAsync(&dv_scratch, n * sizeof(__nv_bfloat16), stream);
            if (dk_scratch && dv_scratch) { dk_t = dk_scratch; dv_t = dv_scratch; }
        }
        wgmma_flash_bwd_dkv_hd64(
            q_used, k_used, d_v, d_do, lse_used, d_rowdot,
            d_segs, nH, dk_t, dv_t, BH, S, Hd, scale, stream);
        if (dk_scratch) cudaFreeAsync(dk_scratch, stream);
        if (dv_scratch) cudaFreeAsync(dv_scratch, stream);
        if (attn_parity_probe_enabled()) {
            static int bwd_par_left = 48;
            static int par_call = 0;
            if (bwd_par_left > 0) {
                --bwd_par_left;
                ++par_call;
                const std::size_t n = static_cast<std::size_t>(BH) * S * Hd;
                __nv_bfloat16 *dk2 = nullptr, *dv2 = nullptr, *dq2 = nullptr;
                float* st = nullptr;
                cudaMallocAsync(&dk2, n * sizeof(__nv_bfloat16), stream);
                cudaMallocAsync(&dv2, n * sizeof(__nv_bfloat16), stream);
                cudaMallocAsync(&dq2, n * sizeof(__nv_bfloat16), stream);
                cudaMallocAsync(&st, 24 * sizeof(float), stream);
                if (dk2 && dv2 && dq2 && st) {
                    // st layout (3 slots each: maxdiff, maxmag, nonfinite):
                    // dK 0-2, dV 3-5, dQ 6-8, census q 9-11, k 12-14, dO 15-17.
                    cudaMemsetAsync(st, 0, 24 * sizeof(float), stream);
                    k_absdiff_max_bf16<<<256, 256, 0, stream>>>(q_used, q_used, n, st + 9);
                    k_absdiff_max_bf16<<<256, 256, 0, stream>>>(k_used, k_used, n, st + 12);
                    k_absdiff_max_bf16<<<256, 256, 0, stream>>>(d_do, d_do, n, st + 15);
                    // The scalar reference re-derives s in scalar math, so it
                    // needs a scalar-consistent surface: feeding it the
                    // WGMMA-stored lse at Hd>=256 reproduces the Era 11
                    // stored-LSE explosion and poisons the comparison.
                    const std::size_t rows = static_cast<std::size_t>(BH) * S;
                    __nv_bfloat16* o_ref = nullptr;
                    float* lse_ref = nullptr;
                    cudaMallocAsync(&o_ref, n * sizeof(__nv_bfloat16), stream);
                    cudaMallocAsync(&lse_ref, rows * sizeof(float), stream);
                    const bool ref_surface = (o_ref && lse_ref);
                    if (ref_surface) {
                        flash_attn_forward(q_used, k_used, d_v, o_ref, lse_ref,
                                           d_segs, nH, BH, S, Hd, scale, stream);
                    } else {
                        cudaGetLastError();
                    }
                    flash_attn_backward(
                        q_used, k_used, d_v,
                        ref_surface ? o_ref : o_used, d_do,
                        ref_surface ? lse_ref : lse_used, dq2, dk2, dv2,
                        d_rowdot, d_segs, nH, BH, S, Hd, scale, stream);
                    if (o_ref) cudaFreeAsync(o_ref, stream);
                    if (lse_ref) cudaFreeAsync(lse_ref, stream);
                    k_absdiff_max_bf16<<<256, 256, 0, stream>>>(d_dk, dk2, n, st);
                    k_absdiff_max_bf16<<<256, 256, 0, stream>>>(d_dv, dv2, n, st + 3);
                    k_absdiff_max_bf16<<<256, 256, 0, stream>>>(d_dq, dq2, n, st + 6);
                    float h[24] = {0};
                    cudaStreamSynchronize(stream);
                    cudaMemcpy(h, st, sizeof(h), cudaMemcpyDeviceToHost);
                    std::fprintf(stderr,
                        "[wgmma-bwd-parity] call=%d Hd=%d max|dK diff|=%.6g (|dK|=%.6g)  "
                        "max|dV diff|=%.6g (|dV|=%.6g)  max|dQ diff|=%.6g (|dQ|=%.6g) dq_wgmma=%d\n",
                        par_call, Hd, h[0], h[1], h[3], h[4], h[6], h[7],
                        wgmma_dq ? 1 : 0);
                    std::fprintf(stderr,
                        "[wgmma-bwd-inputs] call=%d nf(q)=%.0f nf(k)=%.0f nf(dO)=%.0f "
                        "nf(dK)=%.0f nf(dV)=%.0f nf(dQ)=%.0f "
                        "max|q|=%.6g max|k|=%.6g max|dO|=%.6g\n",
                        par_call, h[11], h[14], h[17], h[2], h[5], h[8],
                        h[10], h[13], h[16]);
                }
                cudaFreeAsync(dk2, stream); cudaFreeAsync(dv2, stream);
                cudaFreeAsync(dq2, stream); cudaFreeAsync(st, stream);
            }
        }
        return;
    }

    bool self_lse = (Hd >= 256);
    if (const char* e = std::getenv("IDA_NATIVE_ATTN_SELF_LSE")) {
        if (e[0] == '1') self_lse = true;
        else if (e[0] == '0') self_lse = false;
    }
    if (self_lse) {
        const std::size_t n = static_cast<std::size_t>(BH) * S * Hd;
        const std::size_t rows = static_cast<std::size_t>(BH) * S;
        __nv_bfloat16* o_self = nullptr;
        float* lse_self = nullptr;
        cudaError_t e1 = cudaMallocAsync(&o_self, n * sizeof(__nv_bfloat16), stream);
        cudaError_t e2 = cudaMallocAsync(&lse_self, rows * sizeof(float), stream);
        if (e1 == cudaSuccess && e2 == cudaSuccess) {
            flash_attn_forward(q_bwd, k_bwd, d_v, o_self, lse_self,
                               d_segs, nH, BH, S, Hd, scale, stream);
            flash_attn_backward(
                q_bwd, k_bwd, d_v,
                o_self, d_do, lse_self, d_dq, d_dk, d_dv, d_rowdot,
                d_segs, nH, BH, S, Hd, scale, stream);
            cudaFreeAsync(o_self, stream);
            cudaFreeAsync(lse_self, stream);
        } else {
            if (o_self) cudaFreeAsync(o_self, stream);
            if (lse_self) cudaFreeAsync(lse_self, stream);
            cudaGetLastError();
            // Scratch unavailable — fall back to the stored-LSE path.
            flash_attn_backward(
                reinterpret_cast<const __nv_bfloat16*>(packed_fp4->q_unpack_f32),
                reinterpret_cast<const __nv_bfloat16*>(packed_fp4->k_unpack_f32), d_v,
                d_o, d_do, d_lse, d_dq, d_dk, d_dv, d_rowdot,
                d_segs, nH, BH, S, Hd, scale, stream);
        }
    } else {
        flash_attn_backward(
            reinterpret_cast<const __nv_bfloat16*>(packed_fp4->q_unpack_f32),
            reinterpret_cast<const __nv_bfloat16*>(packed_fp4->k_unpack_f32), d_v,
            d_o, d_do, d_lse, d_dq, d_dk, d_dv, d_rowdot,
            d_segs, nH, BH, S, Hd, scale, stream);
    }

    if (attn_parity_probe_enabled()) {
        static int grad_probes_left = 40;
        if (grad_probes_left > 0) {
            --grad_probes_left;
            const std::size_t n = static_cast<std::size_t>(BH) * S * Hd;
            float* st = nullptr;
            cudaMallocAsync(&st, 6 * sizeof(float), stream);
            if (st) {
                cudaMemsetAsync(st, 0, 6 * sizeof(float), stream);
                k_absdiff_max_bf16<<<256, 256, 0, stream>>>(d_dq, d_dq, n, st);
                k_absdiff_max_bf16<<<256, 256, 0, stream>>>(d_dk, d_dk, n, st + 2);
                k_absdiff_max_bf16<<<256, 256, 0, stream>>>(d_dv, d_dv, n, st + 4);
                float h[6] = {0, 0, 0, 0, 0, 0};
                cudaStreamSynchronize(stream);
                cudaMemcpy(h, st, sizeof(h), cudaMemcpyDeviceToHost);
                std::fprintf(stderr,
                    "[attn-parity-bwd] Hd=%d  max|dQ|=%.6g max|dK|=%.6g max|dV|=%.6g\n",
                    Hd, h[1], h[3], h[5]);
            }
            cudaFreeAsync(st, stream);
        }
    }
}

// ─── forward ─────────────────────────────────────────────────────────────────
__global__ void k_flash_attn_fwd(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    const std::uint16_t* __restrict__ segs, int nH,
    int S, int Hd, float scale, int BR, int BC, int win
) {
    extern __shared__ float sm[];
    const int pad = Hd + 1;
    float* Qs   = sm;                       // [BR][pad]
    float* Ks   = Qs + BR * pad;            // [BC][pad]
    float* Vs   = Ks + BC * pad;            // [BC][pad]
    float* As   = Vs + BC * pad;            // [BR][pad]  online-softmax acc
    float* Ss   = As + BR * pad;            // [BR][BC+1]
    float* m_s  = Ss + BR * (BC + 1);       // [BR]
    float* l_s  = m_s + BR;                 // [BR]
    float* c_s  = l_s + BR;                 // [BR] correction factors

    const int bh = blockIdx.y;
    const int i0 = blockIdx.x * BR;
    const std::size_t base = static_cast<std::size_t>(bh) * S * Hd;
    const int tid = threadIdx.x;
    const int T   = blockDim.x;

    for (int idx = tid; idx < BR * Hd; idx += T) {
        const int r = idx / Hd, d = idx % Hd;
        Qs[r * pad + d] = (i0 + r < S)
            ? __bfloat162float(Q[base + static_cast<std::size_t>(i0 + r) * Hd + d]) : 0.0f;
        As[r * pad + d] = 0.0f;
    }
    if (tid < BR) { m_s[tid] = -FLT_MAX; l_s[tid] = 0.0f; }
    __syncthreads();

    const int j_hi = min(S - 1, i0 + BR - 1);   // causal ceiling for this block
    // Sample-boundary floor: rows in this block attend no earlier than the
    // first row's sample start (seg offsets are monotonic within a row).
    const std::uint16_t* segrow = segs
        ? segs + (static_cast<std::size_t>(bh) / nH) * S : nullptr;
    int j_lo = segrow ? (static_cast<int>(segrow[i0]) / BC) * BC : 0;
    // Sliding window (IDA_NATIVE_ATTN_WINDOW): query i attends keys
    // j in [i-win+1, i].  Same monotonic-floor argument as the seg floor:
    // the block's earliest row (i0) has the lowest window floor.
    if (win > 0) j_lo = max(j_lo, (max(0, i0 - win + 1) / BC) * BC);

    for (int j0 = j_lo; j0 <= j_hi; j0 += BC) {
        const int jn = min(BC, j_hi - j0 + 1);

        for (int idx = tid; idx < BC * Hd; idx += T) {
            const int c = idx / Hd, d = idx % Hd;
            if (c < jn) {
                Ks[c * pad + d] = __bfloat162float(K[base + static_cast<std::size_t>(j0 + c) * Hd + d]);
                Vs[c * pad + d] = __bfloat162float(V[base + static_cast<std::size_t>(j0 + c) * Hd + d]);
            }
        }
        __syncthreads();

        // Scores (causal + sample-boundary + window masked to -FLT_MAX)
        for (int idx = tid; idx < BR * BC; idx += T) {
            const int r = idx / BC, c = idx % BC;
            float s = -FLT_MAX;
            if (c < jn && (i0 + r) < S && (j0 + c) <= (i0 + r)
                && (win <= 0 || (i0 + r) - (j0 + c) < win)
                && (!segrow || (j0 + c) >= static_cast<int>(segrow[i0 + r]))) {
                s = 0.0f;
                for (int d = 0; d < Hd; ++d) s += Qs[r * pad + d] * Ks[c * pad + d];
                s *= scale;
            }
            Ss[r * (BC + 1) + c] = s;
        }
        __syncthreads();

        // Per-row online-softmax bookkeeping (one thread per row)
        if (tid < BR) {
            const int r = tid;
            float m_new = m_s[r];
            for (int c = 0; c < jn; ++c) m_new = fmaxf(m_new, Ss[r * (BC + 1) + c]);
            const float corr = expf(m_s[r] - m_new);   // exp(-inf)=0 first tile
            float l = l_s[r] * corr;
            for (int c = 0; c < jn; ++c) {
                const float p = expf(Ss[r * (BC + 1) + c] - m_new);
                Ss[r * (BC + 1) + c] = p;
                l += p;
            }
            m_s[r] = m_new; l_s[r] = l; c_s[r] = corr;
        }
        __syncthreads();

        // acc[r][d] = acc[r][d]·corr_r + Σ_c P[r][c]·V[c][d]
        for (int idx = tid; idx < BR * Hd; idx += T) {
            const int r = idx / Hd, d = idx % Hd;
            float a = As[r * pad + d] * c_s[r];
            for (int c = 0; c < jn; ++c)
                a += Ss[r * (BC + 1) + c] * Vs[c * pad + d];
            As[r * pad + d] = a;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < BR * Hd; idx += T) {
        const int r = idx / Hd, d = idx % Hd;
        if (i0 + r < S)
            O[base + static_cast<std::size_t>(i0 + r) * Hd + d] =
                __float2bfloat16(As[r * pad + d] / l_s[r]);
    }
    if (tid < BR && i0 + tid < S)
        LSE[static_cast<std::size_t>(bh) * S + i0 + tid] = m_s[tid] + logf(l_s[tid]);
}

// ─── forward, register-tiled (Hd>256 / MoE only) ──────────────────────────────
// Real bottleneck at Hd=512 (see claude-handoff.md "MoE-family first
// profile"): Qs/Ks/Vs/As are each a full Hd-wide row in shared memory
// (pad=Hd+1), which is what forces BR/BC to stay tiny -- the BR*BC score
// term is negligible by comparison (64 floats vs 16512 at (8,8)/Hd=512).
// This variant moves Q and the output accumulator into per-warp registers
// instead (one warp owns one query row, Hd/32 registers/lane), leaving
// only Ks/Vs (BC-scaled) in shared memory. Validated 2026-07-29 via a
// standalone correctness+perf harness (regtile_harness.cu, not checked
// into the tree) against this same k_flash_attn_fwd: max_abs_diff=6.1e-5,
// max_rel_diff=0.78% (bf16-reassociation-noise magnitude, same class as
// every other accepted delta in this thread), 0 compute-sanitizer memcheck
// errors, 0 register spills at BC<=16 (ptxas: 64 regs at BC=8, 89 at
// BC=16), and a real 2.6x isolated-kernel speedup (79.0ms->30.5ms/launch
// at BC=8, 30.2ms at BC=16, S=512/BH=4/Hd=512). BC=32 (136 regs) collapses
// occupancy to 1 block/SM from BOTH registers and shared memory and
// regressed slightly vs BC=16 -- not compiled in here, (8,8) and (8,16)
// are the only two production instantiations.
// Requires Hd % 32 == 0 (true for MoE's Hd=512; gated to Hd==512 at the
// call site, with the original kernel as an unconditional fallback for
// anything else).
template <int BR, int BC, int HD>
__global__ void k_flash_attn_fwd_regtile(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    const std::uint16_t* __restrict__ segs, int nH,
    int S, float scale, int win
) {
    static_assert(HD % 32 == 0, "regtile requires Hd % 32 == 0");
    constexpr int REGN = HD / 32;
    constexpr int pad = HD + 1;

    extern __shared__ float sm[];
    float* Ks = sm;              // [BC][pad]
    float* Vs = Ks + BC * pad;   // [BC][pad]

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;   // row this warp owns, 0..BR-1
    const int lane = tid & 31;
    const int bh = blockIdx.y;
    const int i0 = blockIdx.x * BR;
    const int row = i0 + warp;
    const std::size_t base = static_cast<std::size_t>(bh) * S * HD;

    const std::uint16_t* segrow = segs
        ? segs + (static_cast<std::size_t>(bh) / nH) * S : nullptr;
    const int segrow_row = (segrow && row < S) ? static_cast<int>(segrow[row]) : 0;

    float qreg[REGN];
    float acc[REGN];
    #pragma unroll
    for (int k = 0; k < REGN; ++k) {
        const int d = lane + k * 32;
        qreg[k] = (row < S) ? __bfloat162float(Q[base + static_cast<std::size_t>(row) * HD + d]) : 0.0f;
        acc[k] = 0.0f;
    }
    float m = -FLT_MAX, l = 0.0f;

    const int j_hi = min(S - 1, i0 + BR - 1);
    int j_lo = segrow ? (static_cast<int>(segrow[i0]) / BC) * BC : 0;
    if (win > 0) j_lo = max(j_lo, (max(0, i0 - win + 1) / BC) * BC);

    for (int j0 = j_lo; j0 <= j_hi; j0 += BC) {
        const int jn = min(BC, j_hi - j0 + 1);

        for (int idx = tid; idx < BC * HD; idx += BR * 32) {
            const int c = idx / HD, d = idx % HD;
            if (c < jn) {
                Ks[c * pad + d] = __bfloat162float(K[base + static_cast<std::size_t>(j0 + c) * HD + d]);
                Vs[c * pad + d] = __bfloat162float(V[base + static_cast<std::size_t>(j0 + c) * HD + d]);
            }
        }
        __syncthreads();

        float s_c[BC];
        #pragma unroll
        for (int c = 0; c < BC; ++c) {
            if (c < jn && row < S && (j0 + c) <= row
                && (win <= 0 || row - (j0 + c) < win)
                && (!segrow || (j0 + c) >= segrow_row)) {
                float partial = 0.0f;
                #pragma unroll
                for (int k = 0; k < REGN; ++k) partial += qreg[k] * Ks[c * pad + lane + k * 32];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    partial += __shfl_xor_sync(0xffffffffu, partial, off);
                s_c[c] = partial * scale;
            } else {
                s_c[c] = -FLT_MAX;
            }
        }

        float m_new = m;
        #pragma unroll
        for (int c = 0; c < BC; ++c) if (c < jn) m_new = fmaxf(m_new, s_c[c]);
        const float corr = expf(m - m_new);
        float l_new = l * corr;
        float p_c[BC];
        #pragma unroll
        for (int c = 0; c < BC; ++c) {
            if (c < jn) {
                p_c[c] = expf(s_c[c] - m_new);
                l_new += p_c[c];
            } else {
                p_c[c] = 0.0f;
            }
        }
        m = m_new; l = l_new;

        #pragma unroll
        for (int k = 0; k < REGN; ++k) {
            float a = acc[k] * corr;
            #pragma unroll
            for (int c = 0; c < BC; ++c)
                if (c < jn) a += p_c[c] * Vs[c * pad + lane + k * 32];
            acc[k] = a;
        }
        __syncthreads();
    }

    if (row < S) {
        #pragma unroll
        for (int k = 0; k < REGN; ++k) {
            const int d = lane + k * 32;
            O[base + static_cast<std::size_t>(row) * HD + d] = __float2bfloat16(acc[k] / l);
        }
        if (lane == 0) LSE[static_cast<std::size_t>(bh) * S + row] = m + logf(l);
    }
}

// ─── backward: D[row] = dot(dO[row], O[row]) ─────────────────────────────────
__global__ void k_attn_rowdot(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ out,
    int Hd
) {
    const std::size_t row = blockIdx.x;
    float s = 0.0f;
    for (int d = threadIdx.x; d < Hd; d += 32)
        s += __bfloat162float(A[row * Hd + d]) * __bfloat162float(B[row * Hd + d]);
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_xor_sync(0xffffffff, s, off);
    if (threadIdx.x == 0) out[row] = s;
}

// ─── backward: dQ ────────────────────────────────────────────────────────────
// Block owns BR query rows.  P recomputed from saved LSE.
// dQ_i = scale · Σ_{j≤i} P_ij (dP_ij − D_i) K_j,  dP_ij = dO_i · V_j.
__global__ void k_flash_attn_bwd_dq(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dQ,
    const std::uint16_t* __restrict__ segs, int nH,
    int S, int Hd, float scale, int BR, int BC, int win
) {
    extern __shared__ float sm[];
    const int pad = Hd + 1;
    float* Qs    = sm;                      // [BR][pad]
    float* Os    = Qs + BR * pad;           // [BR][pad]  dO rows
    float* Ks    = Os + BR * pad;           // [BC][pad]
    float* Vs    = Ks + BC * pad;           // [BC][pad]
    float* As    = Vs + BC * pad;           // [BR][pad]  dQ accumulators
    float* dSs   = As + BR * pad;           // [BR][BC+1]
    float* lse_s = dSs + BR * (BC + 1);     // [BR]
    float* d_s   = lse_s + BR;              // [BR]

    const int bh = blockIdx.y;
    const int i0 = blockIdx.x * BR;
    const std::size_t base = static_cast<std::size_t>(bh) * S * Hd;
    const std::size_t rowbase = static_cast<std::size_t>(bh) * S;
    const int tid = threadIdx.x;
    const int T   = blockDim.x;

    for (int idx = tid; idx < BR * Hd; idx += T) {
        const int r = idx / Hd, d = idx % Hd;
        const bool ok = (i0 + r < S);
        Qs[r * pad + d] = ok ? __bfloat162float(Q[base + static_cast<std::size_t>(i0 + r) * Hd + d]) : 0.0f;
        Os[r * pad + d] = ok ? __bfloat162float(dO[base + static_cast<std::size_t>(i0 + r) * Hd + d]) : 0.0f;
        As[r * pad + d] = 0.0f;
    }
    if (tid < BR) {
        const bool ok = (i0 + tid < S);
        lse_s[tid] = ok ? LSE[rowbase + i0 + tid] : 0.0f;
        d_s  [tid] = ok ? D  [rowbase + i0 + tid] : 0.0f;
    }
    __syncthreads();

    const int j_hi = min(S - 1, i0 + BR - 1);
    const std::uint16_t* segrow = segs
        ? segs + (static_cast<std::size_t>(bh) / nH) * S : nullptr;
    int j_lo = segrow ? (static_cast<int>(segrow[i0]) / BC) * BC : 0;
    // Sliding window: same clamp as the forward — the backward must walk
    // the exact key set the forward's LSE was computed over.
    if (win > 0) j_lo = max(j_lo, (max(0, i0 - win + 1) / BC) * BC);

    for (int j0 = j_lo; j0 <= j_hi; j0 += BC) {
        const int jn = min(BC, j_hi - j0 + 1);

        for (int idx = tid; idx < BC * Hd; idx += T) {
            const int c = idx / Hd, d = idx % Hd;
            if (c < jn) {
                Ks[c * pad + d] = __bfloat162float(K[base + static_cast<std::size_t>(j0 + c) * Hd + d]);
                Vs[c * pad + d] = __bfloat162float(V[base + static_cast<std::size_t>(j0 + c) * Hd + d]);
            }
        }
        __syncthreads();

        unsigned p_zero_h = 0, p_valid_h = 0;
        float drift_max_h = -1e30f;
        for (int idx = tid; idx < BR * BC; idx += T) {
            const int r = idx / BC, c = idx % BC;
            float ds = 0.0f;
            if (c < jn && (i0 + r) < S && (j0 + c) <= (i0 + r)
                && (win <= 0 || (i0 + r) - (j0 + c) < win)
                && (!segrow || (j0 + c) >= static_cast<int>(segrow[i0 + r]))) {
                float s = 0.0f, dp = 0.0f;
                for (int d = 0; d < Hd; ++d) {
                    s  += Qs[r * pad + d] * Ks[c * pad + d];
                    dp += Os[r * pad + d] * Vs[c * pad + d];
                }
                // Clamp at 0, not 30: lse is an upper bound for s (self-LSE
                // by construction, or the stored forward LSE over the same
                // replayed operands) up to floating-point rounding — the
                // online-softmax reduction that produced lse and this raw
                // dot-product loop can differ by a tiny amount in summation
                // order alone, even from bit-identical operands.  A +30
                // clamp lets that rounding noise reach exp(30)~1e13; +0
                // saturates it at exp(0)=1, matching the P<=1 invariant.
                // Found 2026-07-09: this scalar kernel had the OLD clamp
                // the WGMMA Stage A kernel already fixed (era12-wgmma-
                // backward-design.md) — L0.attn_norm/k_proj/q_proj gradient
                // explosion at mb=128 (dK 10-30x dQ/dV, growing across
                // calls, from a bit-exact LSE and healthy dO) traced
                // directly to this line via the attn-parity-bwd probe.
                const float arg = s * scale - lse_s[r];
                drift_max_h = fmaxf(drift_max_h, arg);
                const float p = expf(fminf(arg, 0.0f));
                ds = p * (dp - d_s[r]);
                ++p_valid_h;
                p_zero_h += (p * kAttnPScaleProbe < 0.001953125f) ? 1u : 0u;
            }
            dSs[r * (BC + 1) + c] = ds;
        }
        attn_health_accum_scalar(p_zero_h, p_valid_h, drift_max_h);
        __syncthreads();

        for (int idx = tid; idx < BR * Hd; idx += T) {
            const int r = idx / Hd, d = idx % Hd;
            float a = As[r * pad + d];
            for (int c = 0; c < jn; ++c)
                a += dSs[r * (BC + 1) + c] * Ks[c * pad + d];
            As[r * pad + d] = a;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < BR * Hd; idx += T) {
        const int r = idx / Hd, d = idx % Hd;
        if (i0 + r < S)
            dQ[base + static_cast<std::size_t>(i0 + r) * Hd + d] =
                __float2bfloat16(As[r * pad + d] * scale);
    }
}

// ─── backward: dQ, register-tiled (Hd>256 / MoE only) ─────────────────────────
// Same technique and same validation methodology as k_flash_attn_fwd_regtile
// above (see its comment for the full rationale): Qs/Os/As were the
// dominant shared-memory cost (3 of 5 Hd-wide buffers), now live in
// per-warp registers instead -- one warp owns one query row, only Ks/Vs
// stay in shared memory. Validated 2026-07-29 via regtile_bwd_harness.cu
// (scratch, not checked in): max_abs_diff=1.5e-5, max_rel_diff=0.78% (bf16-
// reassociation-noise magnitude, same class as every accepted delta this
// thread), 0 compute-sanitizer errors, 0 register spills (80 regs at BC=8,
// 106 at BC=16), isolated-kernel speedup 106.9ms->49.3ms/launch at BC=8
// (2.17x), 49.6ms at BC=16 (essentially tied with BC=8 in isolation).
// Health-telemetry note: attn_health_accum_scalar internally does its own
// warp-wide SUM reduction assuming 32 threads each hold a genuinely
// different partial count. Here all 32 lanes in a warp end up computing
// IDENTICAL p_zero/p_valid/drift values (every lane redundantly repeats
// the same per-row bookkeeping after the score reduction) -- passing the
// real count from all 32 lanes would inflate the SUM fields 32x. Fix:
// only lane 0 passes its real p_zero/p_valid counts, the other 31 lanes
// pass 0 for those two fields (the MAX-based drift field is unaffected by
// duplication, every lane can pass its real value there) -- but ALL 32
// lanes still call the function collectively, since __shfl_xor_sync
// requires full-warp convergence.
template <int BR, int BC, int HD>
__global__ void k_flash_attn_bwd_dq_regtile(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dQ,
    const std::uint16_t* __restrict__ segs, int nH,
    int S, float scale, int win
) {
    constexpr int REGN = HD / 32;
    constexpr int pad = HD + 1;
    extern __shared__ float sm[];
    float* Ks = sm;
    float* Vs = Ks + BC * pad;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int bh = blockIdx.y;
    const int i0 = blockIdx.x * BR;
    const int row = i0 + warp;
    const std::size_t base = static_cast<std::size_t>(bh) * S * HD;
    const std::size_t rowbase = static_cast<std::size_t>(bh) * S;

    const std::uint16_t* segrow = segs
        ? segs + (static_cast<std::size_t>(bh) / nH) * S : nullptr;
    const int segrow_row = (segrow && row < S) ? static_cast<int>(segrow[row]) : 0;

    float qreg[REGN], oreg[REGN], acc[REGN];
    #pragma unroll
    for (int k = 0; k < REGN; ++k) {
        const int d = lane + k * 32;
        const bool ok = row < S;
        qreg[k] = ok ? __bfloat162float(Q[base + static_cast<std::size_t>(row) * HD + d]) : 0.0f;
        oreg[k] = ok ? __bfloat162float(dO[base + static_cast<std::size_t>(row) * HD + d]) : 0.0f;
        acc[k] = 0.0f;
    }
    const float lse_r = (row < S) ? LSE[rowbase + row] : 0.0f;
    const float d_r    = (row < S) ? D[rowbase + row] : 0.0f;

    const int j_hi = min(S - 1, i0 + BR - 1);
    int j_lo = segrow ? (static_cast<int>(segrow[i0]) / BC) * BC : 0;
    if (win > 0) j_lo = max(j_lo, (max(0, i0 - win + 1) / BC) * BC);

    for (int j0 = j_lo; j0 <= j_hi; j0 += BC) {
        const int jn = min(BC, j_hi - j0 + 1);

        for (int idx = tid; idx < BC * HD; idx += BR * 32) {
            const int c = idx / HD, d = idx % HD;
            if (c < jn) {
                Ks[c * pad + d] = __bfloat162float(K[base + static_cast<std::size_t>(j0 + c) * HD + d]);
                Vs[c * pad + d] = __bfloat162float(V[base + static_cast<std::size_t>(j0 + c) * HD + d]);
            }
        }
        __syncthreads();

        float ds_c[BC];
        unsigned p_zero_h = 0, p_valid_h = 0;
        float drift_max_h = -1e30f;
        #pragma unroll
        for (int c = 0; c < BC; ++c) {
            if (c < jn && row < S && (j0 + c) <= row
                && (win <= 0 || row - (j0 + c) < win)
                && (!segrow || (j0 + c) >= segrow_row)) {
                float ps = 0.0f, pdp = 0.0f;
                #pragma unroll
                for (int k = 0; k < REGN; ++k) {
                    ps  += qreg[k] * Ks[c * pad + lane + k * 32];
                    pdp += oreg[k] * Vs[c * pad + lane + k * 32];
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    ps  += __shfl_xor_sync(0xffffffffu, ps,  off);
                    pdp += __shfl_xor_sync(0xffffffffu, pdp, off);
                }
                const float arg = ps * scale - lse_r;
                drift_max_h = fmaxf(drift_max_h, arg);
                const float p = expf(fminf(arg, 0.0f));
                ds_c[c] = p * (pdp - d_r);
                ++p_valid_h;
                p_zero_h += (p * kAttnPScaleProbe < 0.001953125f) ? 1u : 0u;
            } else {
                ds_c[c] = 0.0f;
            }
        }
        attn_health_accum_scalar(lane == 0 ? p_zero_h : 0u, lane == 0 ? p_valid_h : 0u, drift_max_h);

        #pragma unroll
        for (int k = 0; k < REGN; ++k) {
            float a = acc[k];
            #pragma unroll
            for (int c = 0; c < BC; ++c)
                if (c < jn) a += ds_c[c] * Ks[c * pad + lane + k * 32];
            acc[k] = a;
        }
        __syncthreads();
    }

    if (row < S) {
        #pragma unroll
        for (int k = 0; k < REGN; ++k) {
            const int d = lane + k * 32;
            dQ[base + static_cast<std::size_t>(row) * HD + d] = __float2bfloat16(acc[k] * scale);
        }
    }
}

// ─── backward: dK, dV ────────────────────────────────────────────────────────
// Block owns BR key rows, streams query tiles i ≥ j.
__global__ void k_flash_attn_bwd_dkv(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    const std::uint16_t* __restrict__ segs, int nH,
    int S, int Hd, float scale, int BR, int BC, int win,
    bool cancel_probe = false
) {
    extern __shared__ float sm[];
    const int pad = Hd + 1;
    float* Ks    = sm;                      // [BR][pad]  block's keys
    float* Vs    = Ks + BR * pad;           // [BR][pad]
    float* AKs   = Vs + BR * pad;           // [BR][pad]  dK accumulators
    float* AVs   = AKs + BR * pad;          // [BR][pad]  dV accumulators
    float* Qs    = AVs + BR * pad;          // [BC][pad]  query tile
    float* Os    = Qs + BC * pad;           // [BC][pad]  dO tile
    float* Ps    = Os + BC * pad;           // [BR][BC+1]
    float* dSs   = Ps + BR * (BC + 1);      // [BR][BC+1]
    float* lse_s = dSs + BR * (BC + 1);     // [BC]
    float* d_s   = lse_s + BC;              // [BC]

    const int bh = blockIdx.y;
    const int j0 = blockIdx.x * BR;
    const std::size_t base = static_cast<std::size_t>(bh) * S * Hd;
    const std::size_t rowbase = static_cast<std::size_t>(bh) * S;
    const int tid = threadIdx.x;
    const int T   = blockDim.x;

    for (int idx = tid; idx < BR * Hd; idx += T) {
        const int r = idx / Hd, d = idx % Hd;
        const bool ok = (j0 + r < S);
        Ks[r * pad + d] = ok ? __bfloat162float(K[base + static_cast<std::size_t>(j0 + r) * Hd + d]) : 0.0f;
        Vs[r * pad + d] = ok ? __bfloat162float(V[base + static_cast<std::size_t>(j0 + r) * Hd + d]) : 0.0f;
        AKs[r * pad + d] = 0.0f;
        AVs[r * pad + d] = 0.0f;
    }
    __syncthreads();

    const std::uint16_t* segrow = segs
        ? segs + (static_cast<std::size_t>(bh) / nH) * S : nullptr;
    const int j_blk_hi = min(S - 1, j0 + BR - 1);   // last key this block owns

    for (int c0 = (j0 / BC) * BC; c0 < S; c0 += BC) {
        // Once every query row from this tile onward starts a sample past the
        // block's last key, nothing further can attend to these keys.
        if (segrow && static_cast<int>(segrow[c0]) > j_blk_hi) break;
        // Sliding window: query i attends key j only while i-j < win, so
        // past query j_blk_hi + win - 1 no query can reach this block's keys.
        if (win > 0 && c0 > j_blk_hi + win - 1) break;
        const int cn = min(BC, S - c0);

        for (int idx = tid; idx < BC * Hd; idx += T) {
            const int c = idx / Hd, d = idx % Hd;
            if (c < cn) {
                Qs[c * pad + d] = __bfloat162float(Q[base + static_cast<std::size_t>(c0 + c) * Hd + d]);
                Os[c * pad + d] = __bfloat162float(dO[base + static_cast<std::size_t>(c0 + c) * Hd + d]);
            }
        }
        if (tid < BC && tid < cn) {
            lse_s[tid] = LSE[rowbase + c0 + tid];
            d_s  [tid] = D  [rowbase + c0 + tid];
        }
        __syncthreads();

        unsigned p_zero_h = 0, p_valid_h = 0;
        float drift_max_h = -1e30f;
        for (int idx = tid; idx < BR * BC; idx += T) {
            const int r = idx / BC, c = idx % BC;
            float p = 0.0f, ds = 0.0f;
            if (c < cn && (j0 + r) < S && (c0 + c) >= (j0 + r)
                && (win <= 0 || (c0 + c) - (j0 + r) < win)
                && (!segrow || static_cast<int>(segrow[c0 + c]) <= (j0 + r))) {
                float s = 0.0f, dp = 0.0f;
                for (int d = 0; d < Hd; ++d) {
                    s  += Ks[r * pad + d] * Qs[c * pad + d];
                    dp += Vs[r * pad + d] * Os[c * pad + d];
                }
                // See the identical clamp fix + comment in k_flash_attn_bwd_dq
                // above — same mechanism, same fix, this kernel's dK/dV half.
                const float arg = s * scale - lse_s[c];
                drift_max_h = fmaxf(drift_max_h, arg);
                p  = expf(fminf(arg, 0.0f));
                ds = p * (dp - d_s[c]);
                if (cancel_probe) cancel_probe_accum(dp, d_s[c], ds);
                ++p_valid_h;
                p_zero_h += (p * kAttnPScaleProbe < 0.001953125f) ? 1u : 0u;
            }
            Ps [r * (BC + 1) + c] = p;
            dSs[r * (BC + 1) + c] = ds;
        }
        attn_health_accum_scalar(p_zero_h, p_valid_h, drift_max_h);
        __syncthreads();

        for (int idx = tid; idx < BR * Hd; idx += T) {
            const int r = idx / Hd, d = idx % Hd;
            float ak = AKs[r * pad + d];
            float av = AVs[r * pad + d];
            for (int c = 0; c < cn; ++c) {
                ak += dSs[r * (BC + 1) + c] * Qs[c * pad + d];
                av += Ps [r * (BC + 1) + c] * Os[c * pad + d];
            }
            AKs[r * pad + d] = ak;
            AVs[r * pad + d] = av;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < BR * Hd; idx += T) {
        const int r = idx / Hd, d = idx % Hd;
        if (j0 + r < S) {
            dK[base + static_cast<std::size_t>(j0 + r) * Hd + d] = __float2bfloat16(AKs[r * pad + d] * scale);
            dV[base + static_cast<std::size_t>(j0 + r) * Hd + d] = __float2bfloat16(AVs[r * pad + d]);
        }
    }
}

// ─── backward: dK, dV, register-tiled (Hd>256 / MoE only) ─────────────────────
// Same technique again: Ks/Vs/AKs/AVs (the block's OWNED key/value rows,
// 4 of 6 Hd-wide buffers -- the biggest register footprint of the three
// regtile kernels) now live in per-warp registers, one warp per key row.
// Qs/Os stay in shared memory -- they're genuinely tile-resident (streamed
// per query tile c0), not row-owned by this block, same as Ks/Vs in the
// fwd/bwd_dq regtile kernels. Validated 2026-07-29 via
// regtile_bwd_harness.cu: dK max_rel_diff=0.78%, dV max_rel_diff=1.2%
// (both bf16-reassociation-noise magnitude), 0 compute-sanitizer errors, 0
// register spills (127 regs at BC=8, 139 at BC=16), isolated-kernel
// speedup 137.6ms->66.0ms/launch at BC=8 (2.09x), 55.4ms at BC=16 (2.48x,
// the best of the three kernels). Same lane-0-only-for-sums health-counter
// fix as k_flash_attn_bwd_dq_regtile above (see its comment) applies here
// identically. Does NOT implement the dS-cancellation diagnostic
// (cancel_probe / IDA_NATIVE_ATTN_PARITY=1, cancel_probe_accum) -- that is
// an opt-in debug-only probe; the host wrapper falls back to the original
// k_flash_attn_bwd_dkv whenever that probe is enabled, so the diagnostic
// keeps working unmodified and doesn't need porting into this kernel too.
template <int BR, int BC, int HD>
__global__ void k_flash_attn_bwd_dkv_regtile(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    const std::uint16_t* __restrict__ segs, int nH,
    int S, float scale, int win
) {
    constexpr int REGN = HD / 32;
    constexpr int pad = HD + 1;
    extern __shared__ float sm[];
    float* Qs = sm;
    float* Os = Qs + BC * pad;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int bh = blockIdx.y;
    const int j0 = blockIdx.x * BR;
    const int row = j0 + warp;
    const std::size_t base = static_cast<std::size_t>(bh) * S * HD;
    const std::size_t rowbase = static_cast<std::size_t>(bh) * S;

    const std::uint16_t* segrow = segs
        ? segs + (static_cast<std::size_t>(bh) / nH) * S : nullptr;
    const int j_blk_hi = min(S - 1, j0 + BR - 1);

    float kreg[REGN], vreg[REGN], ak[REGN], av[REGN];
    #pragma unroll
    for (int k = 0; k < REGN; ++k) {
        const int d = lane + k * 32;
        const bool ok = row < S;
        kreg[k] = ok ? __bfloat162float(K[base + static_cast<std::size_t>(row) * HD + d]) : 0.0f;
        vreg[k] = ok ? __bfloat162float(V[base + static_cast<std::size_t>(row) * HD + d]) : 0.0f;
        ak[k] = 0.0f; av[k] = 0.0f;
    }

    for (int c0 = (j0 / BC) * BC; c0 < S; c0 += BC) {
        if (segrow && static_cast<int>(segrow[c0]) > j_blk_hi) break;
        if (win > 0 && c0 > j_blk_hi + win - 1) break;
        const int cn = min(BC, S - c0);

        for (int idx = tid; idx < BC * HD; idx += BR * 32) {
            const int c = idx / HD, d = idx % HD;
            if (c < cn) {
                Qs[c * pad + d] = __bfloat162float(Q[base + static_cast<std::size_t>(c0 + c) * HD + d]);
                Os[c * pad + d] = __bfloat162float(dO[base + static_cast<std::size_t>(c0 + c) * HD + d]);
            }
        }
        __syncthreads();

        float p_c[BC], ds_c[BC];
        unsigned p_zero_h = 0, p_valid_h = 0;
        float drift_max_h = -1e30f;
        #pragma unroll
        for (int c = 0; c < BC; ++c) {
            if (c < cn && row < S && (c0 + c) >= row
                && (win <= 0 || (c0 + c) - row < win)
                && (!segrow || static_cast<int>(segrow[c0 + c]) <= row)) {
                float s = 0.0f, dp = 0.0f;
                #pragma unroll
                for (int k = 0; k < REGN; ++k) {
                    s  += kreg[k] * Qs[c * pad + lane + k * 32];
                    dp += vreg[k] * Os[c * pad + lane + k * 32];
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    s  += __shfl_xor_sync(0xffffffffu, s,  off);
                    dp += __shfl_xor_sync(0xffffffffu, dp, off);
                }
                const float lse_c = LSE[rowbase + c0 + c];
                const float d_c   = D[rowbase + c0 + c];
                const float arg = s * scale - lse_c;
                drift_max_h = fmaxf(drift_max_h, arg);
                p_c[c]  = expf(fminf(arg, 0.0f));
                ds_c[c] = p_c[c] * (dp - d_c);
                ++p_valid_h;
                p_zero_h += (p_c[c] * kAttnPScaleProbe < 0.001953125f) ? 1u : 0u;
            } else {
                p_c[c] = 0.0f; ds_c[c] = 0.0f;
            }
        }
        attn_health_accum_scalar(lane == 0 ? p_zero_h : 0u, lane == 0 ? p_valid_h : 0u, drift_max_h);

        #pragma unroll
        for (int k = 0; k < REGN; ++k) {
            float a_k = ak[k], a_v = av[k];
            #pragma unroll
            for (int c = 0; c < BC; ++c) {
                if (c < cn) {
                    a_k += ds_c[c] * Qs[c * pad + lane + k * 32];
                    a_v += p_c[c]  * Os[c * pad + lane + k * 32];
                }
            }
            ak[k] = a_k; av[k] = a_v;
        }
        __syncthreads();
    }

    if (row < S) {
        #pragma unroll
        for (int k = 0; k < REGN; ++k) {
            const int d = lane + k * 32;
            dK[base + static_cast<std::size_t>(row) * HD + d] = __float2bfloat16(ak[k] * scale);
            dV[base + static_cast<std::size_t>(row) * HD + d] = __float2bfloat16(av[k]);
        }
    }
}

// ─── host wrappers ───────────────────────────────────────────────────────────

// Sliding-window attention.  Era 13 verified defaults are COMPILED IN
// (training_evolution.md: W=128 at Hd>=256 → AI +62% tok/s AND better loss;
// W=256 at Hd=64 → Edge 348k, better loss): unset env = the winning recipe
// per head dim.  IDA_NATIVE_ATTN_WINDOW=W overrides for ablation; =0 forces
// full causal-within-segment (pre-Era-13 behavior, bit-identical kernels).
// Composes with boundary culling: the effective key set per query is
// [max(seg_start, i-W+1), i].  Forward LSE and every backward score walk
// the same set — the window is part of the surface contract, so every
// public scalar kernel resolves identically.
// Per-burn override, e.g. from the real atlas_moe config's
// local_attention_window (2026-07-21, cognitive-architecture port). One
// burn always runs on one dedicated std::thread (both the direct-launch
// path and the shared-context multi-tenant server's per-body_key
// burn_thread, see main.cpp), so thread_local is exactly as safe as
// threading a parameter through every attention call site, without the
// invasive signature changes -- and unlike a process-wide env var, it
// can't leak into a DIFFERENT concurrently-running body's window in a
// shared multi-tenant process.
thread_local int g_attn_window_request_override = -1;

void set_attn_window_request_override(int window) {
    g_attn_window_request_override = window;
}

int attn_window_tokens(int Hd) {
    if (g_attn_window_request_override >= 0) return g_attn_window_request_override;
    // Deliberately NOT cached (no `static`): a shared multi-tenant process
    // (MPS replacement, 2026-07-21) can serve bodies of different families
    // concurrently, and a process-lifetime cache would freeze whichever
    // family's window happened to call first for every other body sharing
    // the process. The per-family default below already reproduces every
    // family's real window exactly (verified: edge Hd=64->256, ai Hd=256->128,
    // swift Hd=32->256, moe Hd=512->128) -- IDA_NATIVE_ATTN_WINDOW is kept
    // only as a manual single-run debug override, read fresh every call.
    const char* e = std::getenv("IDA_NATIVE_ATTN_WINDOW");
    const int env_w = e ? std::atoi(e) : -1;   // -1 = unset → recipe default
    if (env_w >= 0) return env_w;
    return (Hd >= 256) ? 128 : 256;
}

static void grow_smem_limit(const void* kernel, std::size_t bytes) {
    if (bytes > 48 * 1024)
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(bytes));
}

// Default ON as of 2026-07-29: validated via a standalone correctness+perf
// harness (see k_flash_attn_fwd_regtile's own comment) AND a real MoE
// mb=4/ga=8 burn -- 10,576.6 -> 11,900.0 tok/s (+12.5%, reproduced at
// 11,901.7 on a second independent run), loss/grad_norm deltas 0.12%/0.04%
// (float-reassociation-noise magnitude, same class as every other accepted
// delta in this thread), 0 skipped steps. IDA_NATIVE_ATTN_FWD_REGTILE=0 is
// the explicit escape hatch back to the original shared-memory kernel if
// this ever needs to be disabled without a rebuild.
static bool attn_fwd_regtile_enabled() {
    const char* e = std::getenv("IDA_NATIVE_ATTN_FWD_REGTILE");
    return !e || std::atoi(e) != 0;
}

static bool attn_bwd_dq_regtile_enabled() {
    const char* e = std::getenv("IDA_NATIVE_ATTN_BWD_DQ_REGTILE");
    return !e || std::atoi(e) != 0;
}

static bool attn_bwd_dkv_regtile_enabled() {
    const char* e = std::getenv("IDA_NATIVE_ATTN_BWD_DKV_REGTILE");
    return !e || std::atoi(e) != 0;
}

void flash_attn_forward(
    const __nv_bfloat16* d_q, const __nv_bfloat16* d_k, const __nv_bfloat16* d_v,
    __nv_bfloat16* d_o, float* d_lse,
    const std::uint16_t* d_segs, int nH,
    int BH, int S, int Hd, float scale,
    cudaStream_t stream
) {
    if (Hd > kMaxHdSupported) {
        std::fprintf(stderr, "flash_attn_forward: Hd=%d exceeds %d\n", Hd, kMaxHdSupported);
        return;
    }
    int BR, BC;
    attn_tile_dims(Hd, BR, BC);
    if (Hd > 256) attn_tile_override("IDA_NATIVE_ATTN_FWD_TILE_BR", "IDA_NATIVE_ATTN_FWD_TILE_BC", BR, BC);

    // Register-tiled path: default-off (IDA_NATIVE_ATTN_FWD_REGTILE=1),
    // MoE-only (Hd==512 is the only compiled HD specialization -- see the
    // kernel's own comment above k_flash_attn_fwd_regtile), and only for
    // the two validated BR/BC combos (env override above still applies to
    // pick between them, any other value silently falls through to the
    // original kernel below rather than failing closed).
    if (Hd == 512 && BR == 8 && attn_fwd_regtile_enabled()) {
        const std::size_t pad_r = Hd + 1;
        if (BC == 8) {
            const std::size_t smem_r = 2u * 8u * pad_r * sizeof(float);
            grow_smem_limit(reinterpret_cast<const void*>(k_flash_attn_fwd_regtile<8, 8, 512>), smem_r);
            dim3 grid_r((S + 8 - 1) / 8, BH);
            k_flash_attn_fwd_regtile<8, 8, 512><<<grid_r, 8 * 32, smem_r, stream>>>(
                d_q, d_k, d_v, d_o, d_lse, d_segs, nH, S, scale, attn_window_tokens(Hd));
            return;
        }
        if (BC == 16) {
            const std::size_t smem_r = 2u * 16u * pad_r * sizeof(float);
            grow_smem_limit(reinterpret_cast<const void*>(k_flash_attn_fwd_regtile<8, 16, 512>), smem_r);
            dim3 grid_r((S + 8 - 1) / 8, BH);
            k_flash_attn_fwd_regtile<8, 16, 512><<<grid_r, 8 * 32, smem_r, stream>>>(
                d_q, d_k, d_v, d_o, d_lse, d_segs, nH, S, scale, attn_window_tokens(Hd));
            return;
        }
    }

    const std::size_t pad = Hd + 1;
    const std::size_t smem =
        ((2u * BR + 2u * BC) * pad + BR * (BC + 1) + 3u * BR) * sizeof(float);
    grow_smem_limit(reinterpret_cast<const void*>(k_flash_attn_fwd), smem);
    dim3 grid((S + BR - 1) / BR, BH);
    k_flash_attn_fwd<<<grid, 256, smem, stream>>>(
        d_q, d_k, d_v, d_o, d_lse, d_segs, nH, S, Hd, scale, BR, BC,
        attn_window_tokens(Hd));
}

void flash_attn_backward(
    const __nv_bfloat16* d_q, const __nv_bfloat16* d_k, const __nv_bfloat16* d_v,
    const __nv_bfloat16* d_o, const __nv_bfloat16* d_do, const float* d_lse,
    __nv_bfloat16* d_dq, __nv_bfloat16* d_dk, __nv_bfloat16* d_dv,
    float* d_rowdot,   // scratch: [BH * S]
    const std::uint16_t* d_segs, int nH,
    int BH, int S, int Hd, float scale,
    cudaStream_t stream, bool skip_dkv, bool skip_dq
) {
    if (Hd > kMaxHdSupported) {
        std::fprintf(stderr, "flash_attn_backward: Hd=%d exceeds %d\n", Hd, kMaxHdSupported);
        return;
    }
    int BR, BC;
    attn_tile_dims(Hd, BR, BC);
    const std::size_t pad = Hd + 1;

    const unsigned rows = static_cast<unsigned>(BH) * S;
    k_attn_rowdot<<<rows, 32, 0, stream>>>(d_do, d_o, d_rowdot, Hd);

    dim3 grid((S + BR - 1) / BR, BH);
    if (!skip_dq) {   // Era 12 Stage B: dQ handled by the WGMMA kernel
        // bwd_dq gets its own, larger tile for Hd>256 (MoE only, the sole
        // consumer of this branch). Real per-kernel ncu measurement
        // (2026-07-28) showed the shared (8,8) default leaves bwd_dq's
        // score/dS loop (BR*BC=64 work items vs blockDim.x=256) only 25%
        // thread-utilized, and starts it at 2 blocks/SM of occupancy
        // headroom (80.5KB/block of 233472B/SM budget). (16,16)
        // (161.5KB/block, still 1 block/SM) gave 100% utilization in that
        // phase and halved the inner K-tile loop trip count, a real 22.8%
        // reduction in bwd_dq's own GPU time (103.3ms->79.7ms at mb=4/ga=8,
        // probe_ledger.jsonl moe_attn_bwd_dq_tile16_20260728) -- but a later
        // same-day per-kernel env-var sweep (attn_tile_override below,
        // still left in place for future ablations) found (8,16) beats
        // even that: it keeps the SAME 2 blocks/SM occupancy as the (8,8)
        // default (80.5KB/block, same as baseline) while still halving the
        // inner K-tile loop trip count via BC=16, so it gets the loop-trip
        // win without (16,16)'s occupancy cost. Measured end-to-end
        // (MoE mb=4/ga=8/32-micro burn, same recipe as the (16,16) result
        // above): 10,319.7 tok/s baseline (dq=16,16 shipped) -> 10,583.2,
        // reproduced independently at 10,574.9 (delta 0.08%, within noise)
        // -- a real +2.5% over the already-shipped (16,16) fix, not just
        // over the original (8,8) default. fwd(8,16) and bwd_dkv(8,16) were
        // swept the same way and both regressed (fwd -1.3%, bwd_dkv -5.0%);
        // bwd_dkv(16,16) was also re-checked with a clean single-variable
        // burn (not just the earlier confounded ncu read) and confirmed
        // flat (+0.08% over baseline, not a real win) -- both left at their
        // (8,8) default. See probe_ledger.jsonl
        // moe_attn_bwd_dq_tile8x16_20260728. Any future attempt to grow
        // fwd's or bwd_dkv's tile must be justified by its own real A/B
        // measurement, not by this comment's reasoning alone -- that is
        // exactly the mistake this override corrects.
        int BR_dq = BR, BC_dq = BC;
        if (Hd > 256) { BR_dq = 8; BC_dq = 16; }
        if (Hd > 256) attn_tile_override("IDA_NATIVE_ATTN_BWD_DQ_TILE_BR", "IDA_NATIVE_ATTN_BWD_DQ_TILE_BC", BR_dq, BC_dq);

        // Register-tiled path: default ON as of 2026-07-29 (see
        // k_flash_attn_bwd_dq_regtile's own comment for validation).
        // IDA_NATIVE_ATTN_BWD_DQ_REGTILE=0 is the explicit escape hatch.
        bool dispatched_dq = false;
        if (Hd == 512 && BR_dq == 8 && attn_bwd_dq_regtile_enabled()) {
            const std::size_t pad_r = Hd + 1;
            dim3 grid_dq_r((S + 8 - 1) / 8, BH);
            if (BC_dq == 8) {
                const std::size_t smem_r = 2u * 8u * pad_r * sizeof(float);
                grow_smem_limit(reinterpret_cast<const void*>(k_flash_attn_bwd_dq_regtile<8, 8, 512>), smem_r);
                k_flash_attn_bwd_dq_regtile<8, 8, 512><<<grid_dq_r, 8 * 32, smem_r, stream>>>(
                    d_q, d_k, d_v, d_do, d_lse, d_rowdot, d_dq, d_segs, nH, S, scale, attn_window_tokens(Hd));
                dispatched_dq = true;
            } else if (BC_dq == 16) {
                const std::size_t smem_r = 2u * 16u * pad_r * sizeof(float);
                grow_smem_limit(reinterpret_cast<const void*>(k_flash_attn_bwd_dq_regtile<8, 16, 512>), smem_r);
                k_flash_attn_bwd_dq_regtile<8, 16, 512><<<grid_dq_r, 8 * 32, smem_r, stream>>>(
                    d_q, d_k, d_v, d_do, d_lse, d_rowdot, d_dq, d_segs, nH, S, scale, attn_window_tokens(Hd));
                dispatched_dq = true;
            }
        }
        if (!dispatched_dq) {
            const std::size_t smem_dq =
                ((3u * BR_dq + 2u * BC_dq) * pad + BR_dq * (BC_dq + 1) + 2u * BR_dq) * sizeof(float);
            grow_smem_limit(reinterpret_cast<const void*>(k_flash_attn_bwd_dq), smem_dq);
            dim3 grid_dq((S + BR_dq - 1) / BR_dq, BH);
            k_flash_attn_bwd_dq <<<grid_dq, 256, smem_dq, stream>>>(
                d_q, d_k, d_v, d_do, d_lse, d_rowdot, d_dq, d_segs, nH, S, Hd, scale, BR_dq, BC_dq,
                attn_window_tokens(Hd));
        }
    }

    if (skip_dkv) return;   // Era 12: dK/dV handled by the WGMMA kernel
    int BR_dkv = BR, BC_dkv = BC;
    if (Hd > 256) attn_tile_override("IDA_NATIVE_ATTN_BWD_DKV_TILE_BR", "IDA_NATIVE_ATTN_BWD_DKV_TILE_BC", BR_dkv, BC_dkv);
    const bool cancel_probe = attn_parity_probe_enabled();
    if (cancel_probe) cancel_probe_reset(stream);

    // Register-tiled path: default ON as of 2026-07-29 (see
    // k_flash_attn_bwd_dkv_regtile's own comment for validation). Falls
    // back to the original kernel whenever the dS-cancellation diagnostic
    // (cancel_probe) is active -- that probe isn't ported into the regtile
    // kernel, see its comment. IDA_NATIVE_ATTN_BWD_DKV_REGTILE=0 is the
    // explicit escape hatch.
    bool dispatched_dkv = false;
    if (Hd == 512 && BR_dkv == 8 && !cancel_probe && attn_bwd_dkv_regtile_enabled()) {
        const std::size_t pad_r = Hd + 1;
        dim3 grid_dkv_r((S + 8 - 1) / 8, BH);
        if (BC_dkv == 8) {
            const std::size_t smem_r = 2u * 8u * pad_r * sizeof(float);
            grow_smem_limit(reinterpret_cast<const void*>(k_flash_attn_bwd_dkv_regtile<8, 8, 512>), smem_r);
            k_flash_attn_bwd_dkv_regtile<8, 8, 512><<<grid_dkv_r, 8 * 32, smem_r, stream>>>(
                d_q, d_k, d_v, d_do, d_lse, d_rowdot, d_dk, d_dv, d_segs, nH, S, scale, attn_window_tokens(Hd));
            dispatched_dkv = true;
        } else if (BC_dkv == 16) {
            const std::size_t smem_r = 2u * 16u * pad_r * sizeof(float);
            grow_smem_limit(reinterpret_cast<const void*>(k_flash_attn_bwd_dkv_regtile<8, 16, 512>), smem_r);
            k_flash_attn_bwd_dkv_regtile<8, 16, 512><<<grid_dkv_r, 8 * 32, smem_r, stream>>>(
                d_q, d_k, d_v, d_do, d_lse, d_rowdot, d_dk, d_dv, d_segs, nH, S, scale, attn_window_tokens(Hd));
            dispatched_dkv = true;
        }
    }
    if (!dispatched_dkv) {
        const std::size_t smem_dkv =
            ((4u * BR_dkv + 2u * BC_dkv) * pad + 2u * BR_dkv * (BC_dkv + 1) + 2u * BC_dkv) * sizeof(float);
        grow_smem_limit(reinterpret_cast<const void*>(k_flash_attn_bwd_dkv), smem_dkv);
        dim3 grid_dkv((S + BR_dkv - 1) / BR_dkv, BH);
        k_flash_attn_bwd_dkv<<<grid_dkv, 256, smem_dkv, stream>>>(
            d_q, d_k, d_v, d_do, d_lse, d_rowdot, d_dk, d_dv, d_segs, nH, S, Hd, scale, BR_dkv, BC_dkv,
            attn_window_tokens(Hd), cancel_probe);
    }
    if (cancel_probe) {
        static int cancel_probes_left = 40;
        if (cancel_probes_left > 0) {
            --cancel_probes_left;
            float dp_max = 0, D_max = 0, ds_max = 0;
            cancel_probe_read(&dp_max, &D_max, &ds_max, stream);
            std::fprintf(stderr,
                "[dS-cancel] Hd=%d  max|dp|=%.6g  max|D|=%.6g  max|ds|=%.6g\n",
                Hd, dp_max, D_max, ds_max);
        }
    }
}

// Private advanced attention contract wrapper. The implementation is supplied
// only by a deployment-owned package; this public file contains no backend
// math or telemetry forwarding.
static void hopper_fp8_attention_forward(
    const PackedFp4AttentionOperands* scratch,
    const __nv_bfloat16* d_q, const __nv_bfloat16* d_k, const __nv_bfloat16* d_v,
    __nv_bfloat16* d_o, float* d_lse,
    const std::uint16_t* d_segs, int nH,
    int BH, int S, int Hd, float scale, cudaStream_t stream
) {
    // The scratch argument is retained for ABI compatibility.
    if (scratch == nullptr || scratch->q_unpack_f32 == nullptr) {
        throw std::runtime_error("hopper_wgmma_fp8 requires attention scratch buffers");
    }
    wgmma_flash_forward_bf16src(
        d_q, d_k, d_v,
        scratch->q_unpack_f32, scratch->k_unpack_f32,
        d_o, d_lse, BH, S, Hd, scale, d_segs, nH, stream);
}

static void hopper_fp8_attention_backward(
    const PackedFp4AttentionOperands* scratch,
    const __nv_bfloat16* d_q, const __nv_bfloat16* d_k, const __nv_bfloat16* d_v,
    const __nv_bfloat16* d_o, const __nv_bfloat16* d_do, const float* d_lse,
    __nv_bfloat16* d_dq, __nv_bfloat16* d_dk, __nv_bfloat16* d_dv,
    float* d_rowdot,
    const std::uint16_t* d_segs, int nH,
    int BH, int S, int Hd, float scale, cudaStream_t stream
) {
    (void)scratch;
    // q/k are regenerated by the layer recompute each backward, so rounding
    // them through scaled e4m3 in place is safe and gives operand parity.
    const std::size_t n = static_cast<std::size_t>(BH) * S * Hd;
    roundtrip_bf16_through_scaled_e4m3(const_cast<__nv_bfloat16*>(d_q), n, stream);
    roundtrip_bf16_through_scaled_e4m3(const_cast<__nv_bfloat16*>(d_k), n, stream);
    flash_attn_backward(d_q, d_k, d_v, d_o, d_do, d_lse,
                        d_dq, d_dk, d_dv, d_rowdot, d_segs, nH, BH, S, Hd, scale, stream);
}

void attention_forward(
    AttentionBackendKind backend,
    const __nv_bfloat16* d_q, const __nv_bfloat16* d_k, const __nv_bfloat16* d_v,
    __nv_bfloat16* d_o, float* d_lse,
    const PackedFp4AttentionOperands* packed_fp4,
    const std::uint16_t* d_segs, int nH,
    int BH, int S, int Hd, float scale,
    cudaStream_t stream
) {
    switch (backend) {
        case AttentionBackendKind::ScalarFlash:
            flash_attn_forward(d_q, d_k, d_v, d_o, d_lse, d_segs, nH, BH, S, Hd, scale, stream);
            return;
        case AttentionBackendKind::HopperWgmmaPackedFp4:
            hopper_packed_fp4_attention_forward(
                packed_fp4, d_v, d_o, d_lse, d_segs, nH, BH, S, Hd, scale, stream);
            return;
        case AttentionBackendKind::HopperWgmmaFp8:
            hopper_fp8_attention_forward(
                packed_fp4, d_q, d_k, d_v, d_o, d_lse, d_segs, nH, BH, S, Hd, scale, stream);
            return;
        case AttentionBackendKind::BlackwellMxf4Fp4:
            throw std::runtime_error(
                "blackwell_mxf4_fp4 is prepared but not launchable until the sm_120a probe passes and the kernel is implemented");
        default:
            throw std::runtime_error("attention_forward received an unknown backend");
    }
}

void attention_backward(
    AttentionBackendKind backend,
    const __nv_bfloat16* d_q, const __nv_bfloat16* d_k, const __nv_bfloat16* d_v,
    const __nv_bfloat16* d_o, const __nv_bfloat16* d_do, const float* d_lse,
    __nv_bfloat16* d_dq, __nv_bfloat16* d_dk, __nv_bfloat16* d_dv,
    float* d_rowdot,
    const PackedFp4AttentionOperands* packed_fp4,
    const std::uint16_t* d_segs, int nH,
    int BH, int S, int Hd, float scale,
    cudaStream_t stream
) {
    switch (backend) {
        case AttentionBackendKind::ScalarFlash:
            flash_attn_backward(
                d_q, d_k, d_v, d_o, d_do, d_lse, d_dq, d_dk, d_dv,
                d_rowdot, d_segs, nH, BH, S, Hd, scale, stream);
            return;
        case AttentionBackendKind::HopperWgmmaPackedFp4:
            hopper_packed_fp4_attention_backward(
                packed_fp4, d_v, d_o, d_do, d_lse, d_dq, d_dk, d_dv,
                d_rowdot, d_segs, nH, BH, S, Hd, scale, stream);
            return;
        case AttentionBackendKind::HopperWgmmaFp8:
            hopper_fp8_attention_backward(
                packed_fp4, d_q, d_k, d_v, d_o, d_do, d_lse, d_dq, d_dk, d_dv,
                d_rowdot, d_segs, nH, BH, S, Hd, scale, stream);
            return;
        case AttentionBackendKind::BlackwellMxf4Fp4:
            throw std::runtime_error(
                "blackwell_mxf4_fp4 is prepared but not launchable until the sm_120a probe passes and the kernel is implemented");
        default:
            throw std::runtime_error("attention_backward received an unknown backend");
    }
}

}  // namespace ida_native
