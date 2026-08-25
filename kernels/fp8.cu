#include "ida_native/kernels.hpp"
#include "ida_native/gemm_trace.hpp"
#include "ida_native/pack_trace.hpp"
#include "ida_native/fp8_e4m3.hpp"
#include "ida_native/cuda_check.hpp"

#include <cfloat>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// FP8 quantization support for the native trainer's cuBLASLt path.
// Per-tensor scaling: scale = fp8_max / amax, quantized = x * scale,
// descale = amax / fp8_max handed to cublasLtMatmul as the dequant factor.
// E4M3 (max 448) for weights and activations, E5M2 (max 57344) for gradients.

namespace ida_native {

// amax via atomicMax on the int representation — valid because |x| ≥ 0 and
// non-negative IEEE floats order identically to their int bit patterns.
__global__ void k_amax_bf16(const __nv_bfloat16* x, std::size_t n, float* amax) {
    float local = 0.0f;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        local = fmaxf(local, fabsf(__bfloat162float(x[i])));
    for (int off = 16; off > 0; off >>= 1)
        local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, off));
    if ((threadIdx.x & 31) == 0)
        atomicMax(reinterpret_cast<int*>(amax), __float_as_int(local));
}

__global__ void k_scale_from_amax(
    const float* amax, float fp8_max, float* scale, float* descale
) {
    const float a = fmaxf(*amax, 1e-12f);
    *scale   = fp8_max / a;
    *descale = a / fp8_max;
}

__global__ void k_quant_e4m3(
    const __nv_bfloat16* x, __nv_fp8_e4m3* out, const float* scale, std::size_t n
) {
    const float s = *scale;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        out[i] = __nv_fp8_e4m3(__bfloat162float(x[i]) * s);
}

__global__ void k_quant_e5m2(
    const __nv_bfloat16* x, __nv_fp8_e5m2* out, const float* scale, std::size_t n
) {
    const float s = *scale;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        out[i] = __nv_fp8_e5m2(__bfloat162float(x[i]) * s);
}

// ─── Delayed-scaling fused kernels ───────────────────────────────────────────
// Quantize with the PREVIOUS step's scale while recording the current tensor's
// amax as a side effect — one pass instead of amax-pass + quant-pass.
// (TE-style delayed scaling; caller zeroes amax_out and derives the next
// scale from it after the consuming GEMM has been enqueued.)
__global__ void k_quant_e4m3_record(
    const __nv_bfloat16* x, __nv_fp8_e4m3* out,
    const float* scale, float* amax_out, std::size_t n
) {
    const float s = *scale;
    float local = 0.0f;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float v = __bfloat162float(x[i]);
        local = fmaxf(local, fabsf(v));
        out[i] = __nv_fp8_e4m3(v * s);
    }
    for (int off = 16; off > 0; off >>= 1)
        local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, off));
    if ((threadIdx.x & 31) == 0)
        atomicMax(reinterpret_cast<int*>(amax_out), __float_as_int(local));
}

__global__ void k_quant_e5m2_record(
    const __nv_bfloat16* x, __nv_fp8_e5m2* out,
    const float* scale, float* amax_out, std::size_t n
) {
    const float s = *scale;
    float local = 0.0f;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float v = __bfloat162float(x[i]);
        local = fmaxf(local, fabsf(v));
        out[i] = __nv_fp8_e5m2(v * s);
    }
    for (int off = 16; off > 0; off >>= 1)
        local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, off));
    if ((threadIdx.x & 31) == 0)
        atomicMax(reinterpret_cast<int*>(amax_out), __float_as_int(local));
}

__global__ void k_update_delayed_scale(
    const float* amax,
    float fp8_max,
    float* scale,
    float* descale,
    float* history,
    int history_len,
    int* history_cursor,
    int* history_count
) {
    const float a = fmaxf(*amax, 1e-12f);
    int cursor = *history_cursor;
    int count = *history_count;
    if (history_len <= 0) {
        *scale = fp8_max / a;
        *descale = a / fp8_max;
        return;
    }

    if (cursor < 0 || cursor >= history_len) {
        cursor = 0;
    }
    history[cursor] = a;
    cursor = (cursor + 1) % history_len;
    if (count < history_len) {
        ++count;
    }

    float hist_max = history[0];
    for (int i = 1; i < count; ++i) {
        hist_max = fmaxf(hist_max, history[i]);
    }
    hist_max = fmaxf(hist_max, 1e-12f);
    *history_cursor = cursor;
    *history_count = count;
    *scale = fp8_max / hist_max;
    *descale = hist_max / fp8_max;
}

// Fused BF16-quantize + dual output: reads in[K,N] BF16 once and writes:
//   act8_out[K,N] E4M3  — same layout as input (for the consuming forward GEMM)
//   snap_out[N,K] E4M3  — transposed (for cuBLASLt TN dW GEMM)
// Saves the act8 HBM read-back that the prior two-step
// (fp8_quant_act → transpose_u8) required.
// Tile: 64×64 elements, 256 threads (16×16), each thread handles 16 elems.
// Row stride 80 avoids shared-memory bank conflicts (same as k_transpose_u8_v16).
__global__ void k_quant_bf16_e4m3_dual(
    const __nv_bfloat16* __restrict__ in,
    __nv_fp8_e4m3*       __restrict__ act8_out,
    __nv_fp8_e4m3*       __restrict__ snap_out,
    const float*         __restrict__ scale,
    int K, int N
) {
    __shared__ uint8_t tile[64][80];

    const int tile_r = blockIdx.y * 64;
    const int tile_c = blockIdx.x * 64;
    const int tid    = threadIdx.y * blockDim.x + threadIdx.x;
    const int lr     = tid / 4;
    const int lc     = (tid % 4) * 16;
    const float s    = *scale;

    // ── Phase 1: load 16 BF16 → quantize to E4M3 → store to tile ────────────
    const int r = tile_r + lr;
    const int c = tile_c + lc;
    if (r < K) {
        alignas(16) __nv_bfloat16 v[16];
        if (c + 15 < N) {
            *reinterpret_cast<uint4*>(&v[0]) =
                *reinterpret_cast<const uint4*>(in + static_cast<std::size_t>(r) * N + c);
            *reinterpret_cast<uint4*>(&v[8]) =
                *reinterpret_cast<const uint4*>(in + static_cast<std::size_t>(r) * N + c + 8);
        } else {
            for (int b = 0; b < 16; ++b)
                v[b] = (c + b < N) ? in[static_cast<std::size_t>(r) * N + c + b]
                                    : __nv_bfloat16(0.0f);
        }
        #pragma unroll
        for (int b = 0; b < 16; ++b) {
            const __nv_fp8_e4m3 fp8_v(__bfloat162float(v[b]) * s);
            tile[lr][lc + b] = fp8_v.__x;
        }
    } else {
        #pragma unroll
        for (int b = 0; b < 16; ++b) tile[lr][lc + b] = 0;
    }
    __syncthreads();

    // ── Phase 2: coalesced write to act8_out[K,N] (same row-major layout) ───
    if (r < K) {
        alignas(16) uint8_t va[16];
        #pragma unroll
        for (int b = 0; b < 16; ++b) va[b] = tile[lr][lc + b];
        if (c + 15 < N) {
            *reinterpret_cast<uint4*>(
                reinterpret_cast<uint8_t*>(act8_out) +
                static_cast<std::size_t>(r) * N + c) =
                *reinterpret_cast<const uint4*>(va);
        } else {
            for (int b = 0; b < 16 && c + b < N; ++b)
                reinterpret_cast<uint8_t*>(act8_out)[static_cast<std::size_t>(r) * N + c + b] = va[b];
        }
    }

    // ── Phase 3: transposed coalesced write to snap_out[N,K] ────────────────
    // out row = input col (tile_c + lr), out col = input row (tile_r + lc)
    const int orow = tile_c + lr;
    const int ocol = tile_r + lc;
    if (orow < N) {
        alignas(16) uint8_t vs[16];
        #pragma unroll
        for (int b = 0; b < 16; ++b) vs[b] = tile[lc + b][lr];
        if (ocol + 15 < K) {
            *reinterpret_cast<uint4*>(
                reinterpret_cast<uint8_t*>(snap_out) +
                static_cast<std::size_t>(orow) * K + ocol) =
                *reinterpret_cast<const uint4*>(vs);
        } else {
            for (int b = 0; b < 16 && ocol + b < K; ++b)
                reinterpret_cast<uint8_t*>(snap_out)[static_cast<std::size_t>(orow) * K + ocol + b] = vs[b];
        }
    }
}

// Quantize + transpose: w[K,N] row-major → out[N,K] row-major (fwd layout).
__global__ void k_quant_transpose_e4m3(
    const __nv_bfloat16* w, __nv_fp8_e4m3* out, const float* scale, int K, int N
) {
    const float s = *scale;
    const std::size_t total = static_cast<std::size_t>(K) * N;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int k = static_cast<int>(i / N);
        const int c = static_cast<int>(i % N);
        out[static_cast<std::size_t>(c) * K + k] =
            __nv_fp8_e4m3(__bfloat162float(w[i]) * s);
    }
}

static unsigned fp8_blocks(std::size_t n) {
    const std::size_t b = (n + 255) / 256;
    return static_cast<unsigned>(b < 1024 ? b : 1024);
}

void fp8_amax_bf16(const __nv_bfloat16* d_x, std::size_t n, float* d_amax, cudaStream_t s) {
    cudaMemsetAsync(d_amax, 0, sizeof(float), s);
    k_amax_bf16<<<fp8_blocks(n), 256, 0, s>>>(d_x, n, d_amax);
}

void fp8_scale_from_amax(
    const float* d_amax, float fp8_max, float* d_scale, float* d_descale, cudaStream_t s
) {
    k_scale_from_amax<<<1, 1, 0, s>>>(d_amax, fp8_max, d_scale, d_descale);
}

void fp8_quantize_e4m3(
    const __nv_bfloat16* d_x, void* d_out, const float* d_scale, std::size_t n, cudaStream_t s
) {
    pack_trace::record(
        pack_trace::K_FP8_PACK, n,
        n * sizeof(__nv_bfloat16), n);
    k_quant_e4m3<<<fp8_blocks(n), 256, 0, s>>>(
        d_x, static_cast<__nv_fp8_e4m3*>(d_out), d_scale, n);
}

void fp8_quantize_e5m2(
    const __nv_bfloat16* d_x, void* d_out, const float* d_scale, std::size_t n, cudaStream_t s
) {
    pack_trace::record(
        pack_trace::K_FP8_PACK, n,
        n * sizeof(__nv_bfloat16), n);
    k_quant_e5m2<<<fp8_blocks(n), 256, 0, s>>>(
        d_x, static_cast<__nv_fp8_e5m2*>(d_out), d_scale, n);
}

void fp8_quantize_transpose_e4m3(
    const __nv_bfloat16* d_w, void* d_out, const float* d_scale, int K, int N, cudaStream_t s
) {
    const std::size_t elements = static_cast<std::size_t>(K) * N;
    pack_trace::record(
        pack_trace::K_FP8_PACK_TRANSPOSE, elements,
        elements * sizeof(__nv_bfloat16), elements);
    k_quant_transpose_e4m3<<<fp8_blocks(static_cast<std::size_t>(K) * N), 256, 0, s>>>(
        d_w, static_cast<__nv_fp8_e4m3*>(d_out), d_scale, K, N);
}

void fp8_quant_and_transpose_e4m3(
    const __nv_bfloat16* d_in, void* d_act8_out, void* d_snap_out,
    const float* d_scale, int K, int N, cudaStream_t s
) {
    const std::size_t elements = static_cast<std::size_t>(K) * N;
    pack_trace::record(
        pack_trace::K_FP8_PACK_FUSED_TRANSPOSE, elements,
        elements * sizeof(__nv_bfloat16), 2 * elements);
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((N + 63) / 64),
              static_cast<unsigned>((K + 63) / 64));
    k_quant_bf16_e4m3_dual<<<grid, block, 0, s>>>(
        d_in,
        static_cast<__nv_fp8_e4m3*>(d_act8_out),
        static_cast<__nv_fp8_e4m3*>(d_snap_out),
        d_scale, K, N);
}

void fp8_quantize_e4m3_record(
    const __nv_bfloat16* d_x, void* d_out, const float* d_scale, float* d_amax,
    std::size_t n, cudaStream_t s
) {
    pack_trace::record(
        pack_trace::K_FP8_PACK, n,
        n * sizeof(__nv_bfloat16), n);
    k_quant_e4m3_record<<<fp8_blocks(n), 256, 0, s>>>(
        d_x, static_cast<__nv_fp8_e4m3*>(d_out), d_scale, d_amax, n);
}

void fp8_quantize_e5m2_record(
    const __nv_bfloat16* d_x, void* d_out, const float* d_scale, float* d_amax,
    std::size_t n, cudaStream_t s
) {
    pack_trace::record(
        pack_trace::K_FP8_PACK, n,
        n * sizeof(__nv_bfloat16), n);
    k_quant_e5m2_record<<<fp8_blocks(n), 256, 0, s>>>(
        d_x, static_cast<__nv_fp8_e5m2*>(d_out), d_scale, d_amax, n);
}


// Delayed-scaling saturation probe. Quantization uses the PREVIOUS call's
// scale (see fp8_quantize_*_record), so any element whose magnitude exceeds
// that scale's window silently clamps to fp8_max -- __nv_fp8_e4m3(float) uses
// __NV_SATFINITE, producing no inf and no nan, which makes the clipping
// invisible in every metric this engine currently records (nonfinite counters
// included). Clipping occurred on a call iff recorded_amax * scale_used >
// fp8_max; both are already device scalars, so detecting it costs one
// 1-thread kernel and requires no change to the quantize hot path.
// stats[0] = calls that clipped, stats[1] = max overflow ratio, stats[2] = total calls.
__global__ void k_fp8_clip_probe(
    const float* __restrict__ amax, const float* __restrict__ scale,
    float fp8_max, float* __restrict__ stats
) {
    const float a = *amax;
    const float s = *scale;
    stats[2] += 1.0f;
    if (fp8_max > 0.0f && s > 0.0f) {
        const float ratio = a * s / fp8_max;
        if (ratio > 1.0f) {
            stats[0] += 1.0f;
            if (ratio > stats[1]) stats[1] = ratio;
        }
    }
}

void fp8_clip_probe(
    const float* d_amax, const float* d_scale, float fp8_max,
    float* d_stats, cudaStream_t s
) {
    k_fp8_clip_probe<<<1, 1, 0, s>>>(d_amax, d_scale, fp8_max, d_stats);
}


// Element-level FP8 saturation counter. The call-level probe only reports
// whether a call's LARGEST element clipped; that cannot distinguish "one
// outlier clamped" (benign) from "a third of the tensor clamped"
// (catastrophic). This counts the actual elements that exceed the fp8 window
// under the scale the quantize used. Separate pass, debug-only, so the
// record-quantize hot kernel stays byte-identical.
// out[0] += clipped elements, out[1] += total elements
__global__ void k_fp8_elem_clip_count(
    const __nv_bfloat16* __restrict__ x, const float* __restrict__ scale,
    float fp8_max, std::size_t n, float* __restrict__ out
) {
    const float s = *scale;
    std::size_t clipped = 0;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float v = fabsf(__bfloat162float(x[i])) * s;
        if (v > fp8_max) ++clipped;
    }
    if (clipped) atomicAdd(&out[0], static_cast<float>(clipped));
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(&out[1], static_cast<float>(n));
}

void fp8_elem_clip_count(
    const __nv_bfloat16* d_x, const float* d_scale, float fp8_max,
    std::size_t n, float* d_out, cudaStream_t s
) {
    k_fp8_elem_clip_count<<<fp8_blocks(n), 256, 0, s>>>(d_x, d_scale, fp8_max, n, d_out);
}

void fp8_update_delayed_scale(
    const float* d_amax,
    float fp8_max,
    float* d_scale,
    float* d_descale,
    float* d_history,
    int history_len,
    int* d_history_cursor,
    int* d_history_count,
    cudaStream_t s
) {
    k_update_delayed_scale<<<1, 1, 0, s>>>(
        d_amax,
        fp8_max,
        d_scale,
        d_descale,
        d_history,
        history_len,
        d_history_cursor,
        d_history_count
    );
}

// Raw-byte E4M3 storage path used by ampere_fp8_packed. It deliberately does
// not instantiate __nv_fp8_e4m3 and does not call cuBLASLt: the byte is a
// derived cache, while BF16 remains the only GEMM input type on sm_86.
__global__ void k_pack_raw_e4m3_stats(
    const __nv_bfloat16* x, std::size_t n, float* amax, int* bad
) {
    float local_max = 0.0f;
    int local_bad = 0;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float value = __bfloat162float(x[i]);
        if (!isfinite(value)) {
            local_bad = 1;
        } else {
            local_max = fmaxf(local_max, fabsf(value));
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));
        local_bad |= __shfl_xor_sync(0xffffffff, local_bad, off);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicMax(reinterpret_cast<int*>(amax), __float_as_int(local_max));
        if (local_bad) atomicAdd(bad, local_bad);
    }
}

__global__ void k_pack_raw_e4m3(
    const __nv_bfloat16* x, std::uint8_t* out, const float* scale,
    int* bad, std::size_t n
) {
    const float s = *scale;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const auto packed = fp8_e4m3::pack(__bfloat162float(x[i]) * s);
        out[i] = packed.bits;
        if (!packed.finite) atomicAdd(bad, 1);
    }
}

__global__ void k_unpack_raw_e4m3(
    const std::uint8_t* in, const float* descale,
    __nv_bfloat16* out, std::size_t n
) {
    const float d = *descale;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        out[i] = __float2bfloat16(fp8_e4m3::unpack(in[i]) * d);
    }
}

__global__ void k_pack_raw_e4m3_and_dequant(
    const __nv_bfloat16* x,
    std::uint8_t* packed,
    __nv_bfloat16* dequant,
    const float* scale,
    const float* descale,
    int* bad,
    std::size_t n
) {
    const float s = *scale;
    const float d = *descale;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const auto code = fp8_e4m3::pack(__bfloat162float(x[i]) * s);
        packed[i] = code.bits;
        dequant[i] = __float2bfloat16(fp8_e4m3::unpack(code.bits) * d);
        if (!code.finite) atomicAdd(bad, 1);
    }
}

void fp8_pack_bf16_e4m3_raw(
    const __nv_bfloat16* d_x, std::uint8_t* d_out,
    float* d_amax, float* d_scale, float* d_descale, int* d_bad_count,
    std::size_t n, cudaStream_t s
) {
    const unsigned blocks = fp8_blocks(n);
    IDA_CUDA_CHECK(cudaMemsetAsync(d_amax, 0, sizeof(float), s));
    IDA_CUDA_CHECK(cudaMemsetAsync(d_bad_count, 0, sizeof(int), s));
    k_pack_raw_e4m3_stats<<<blocks, 256, 0, s>>>(d_x, n, d_amax, d_bad_count);
    k_scale_from_amax<<<1, 1, 0, s>>>(d_amax, 448.0f, d_scale, d_descale);
    k_pack_raw_e4m3<<<blocks, 256, 0, s>>>(d_x, d_out, d_scale, d_bad_count, n);
    pack_trace::record(pack_trace::K_FP8_PACK, n, n * sizeof(__nv_bfloat16), n);
}

void fp8_pack_bf16_e4m3_raw_and_dequant(
    const __nv_bfloat16* d_x, std::uint8_t* d_out, __nv_bfloat16* d_dequant,
    float* d_amax, float* d_scale, float* d_descale, int* d_bad_count,
    std::size_t n, cudaStream_t s
) {
    const unsigned blocks = fp8_blocks(n);
    IDA_CUDA_CHECK(cudaMemsetAsync(d_amax, 0, sizeof(float), s));
    IDA_CUDA_CHECK(cudaMemsetAsync(d_bad_count, 0, sizeof(int), s));
    k_pack_raw_e4m3_stats<<<blocks, 256, 0, s>>>(d_x, n, d_amax, d_bad_count);
    k_scale_from_amax<<<1, 1, 0, s>>>(d_amax, 448.0f, d_scale, d_descale);
    k_pack_raw_e4m3_and_dequant<<<blocks, 256, 0, s>>>(
        d_x, d_out, d_dequant, d_scale, d_descale, d_bad_count, n);
    pack_trace::record(
        pack_trace::K_FP8_PACK_DEQUANT, n,
        n * sizeof(__nv_bfloat16),
        n * sizeof(std::uint8_t) + n * sizeof(__nv_bfloat16));
}

void fp8_unpack_e4m3_raw_bf16(
    const std::uint8_t* d_in, const float* d_descale,
    __nv_bfloat16* d_out, std::size_t n, cudaStream_t s
) {
    k_unpack_raw_e4m3<<<fp8_blocks(n), 256, 0, s>>>(d_in, d_descale, d_out, n);
}

}  // namespace ida_native
