#include "ida_native/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include <cstdio>
#include <cstdlib>

namespace ida_native {

__global__ void k_embedding_fwd(
    const uint32_t*      token_ids,
    const __nv_bfloat16* weight,
    __nv_bfloat16*       out,
    int H
) {
    const int bs = blockIdx.x;
    const int h  = static_cast<int>(blockIdx.y) * blockDim.x + static_cast<int>(threadIdx.x);
    if (h >= H) return;
    const uint32_t tok = token_ids[bs];
    out[bs * H + h] = weight[static_cast<long long>(tok) * H + h];
}

__global__ void k_embedding_bwd(
    const __nv_bfloat16* d_out,
    const uint32_t*      token_ids,
    float*               d_weight,
    int BS, int H
) {
    const int bs = blockIdx.x;
    const int h  = static_cast<int>(blockIdx.y) * blockDim.x + static_cast<int>(threadIdx.x);
    if (bs >= BS || h >= H) return;
    const uint32_t tok = token_ids[bs];
    const float g = __bfloat162float(d_out[bs * H + h]);
    atomicAdd(&d_weight[static_cast<long long>(tok) * H + h], g);
}

// BF16 gradient-storage variant used by the explicit Ampere 1F1B MoE
// profile.  The scatter remains atomic because repeated token ids are
// expected; rounding occurs at the durable gradient buffer boundary rather
// than retaining a full FP32 embedding gradient for the whole burn.
__global__ void k_embedding_bwd_bf16(
    const __nv_bfloat16* d_out,
    const uint32_t*      token_ids,
    __nv_bfloat16*       d_weight,
    int BS, int H
) {
    const int bs = blockIdx.x;
    const int h  = static_cast<int>(blockIdx.y) * blockDim.x + static_cast<int>(threadIdx.x);
    if (bs >= BS || h >= H) return;
    const uint32_t tok = token_ids[bs];
    const float g = __bfloat162float(d_out[bs * H + h]);
    atomicAdd(&d_weight[static_cast<long long>(tok) * H + h], __float2bfloat16(g));
}

void embedding_forward(
    const uint32_t* d_tokens, const __nv_bfloat16* d_weight,
    __nv_bfloat16* d_out, int B, int S, int H, cudaStream_t stream
) {
    const int BS = B * S;
    const int T  = 128;
    dim3 grid(BS, (H + T - 1) / T);
    k_embedding_fwd<<<grid, T, 0, stream>>>(d_tokens, d_weight, d_out, H);
}

void embedding_backward(
    const __nv_bfloat16* d_grad_out,
    const uint32_t*      d_tokens,
    __nv_bfloat16*       d_grad_weight,
    int B, int S, int H,
    cudaStream_t stream
) {
    const int BS = B * S;
    const int T  = 128;
    dim3 grid(BS, (H + T - 1) / T);
    k_embedding_bwd_bf16<<<grid, T, 0, stream>>>(d_grad_out, d_tokens, d_grad_weight, BS, H);
}

// ── Deterministic embedding backward (2026-07-18) ────────────────────────────
// k_embedding_bwd above scatter-accumulates with float atomicAdd — the arrival
// order of those atomics is scheduling-dependent, and float addition is not
// associative, so the same binary on the same request produces a slightly
// different g_embed every run. Measured consequence: Edge (Hd=64) drifts
// ~6.5e-4 loss/step run-to-run with embed as its dominant grad slot (~68% of
// grad norm); Swift concentrates even harder (~94%) through this same path.
// That noise floor is what forces every Edge parity probe to run
// same-binary determinism controls instead of judging against identity.
//
// This path removes the non-determinism at the source:
//   1. cub stable radix sort of (token_id, position) pairs — identical input
//      always yields the identical sorted order.
//   2. cub run-length encode — one contiguous run per unique token.
//   3. k_embedding_bwd_det: one block per run. Threads stride H; each thread
//      accumulates its column across the run's positions IN SORTED ORDER in
//      an fp32 register, then does a single non-atomic `+=` to the grad row.
//      Each unique token appears in exactly one run, so the block owns its
//      row exclusively — no atomics anywhere, fixed FP summation order,
//      bit-identical output for identical input.
//
// Throughput note: this is not just a determinism tax. The file's own row-clip
// comment documents ~50 high-frequency tokens covering ~70% of all positions —
// under the atomic path those rows take thousands of SAME-ADDRESS atomicAdds
// per micro-step, which serialize through the L2 atomic unit anyway. The
// sequential register accumulation here does the same serial work without
// the atomic round-trips.
//
// Gated behind IDA_NATIVE_DET_EMBED_BWD (default OFF) until it passes the
// standard probe-ledger validation (same-binary determinism control +
// throughput A/B at production shape). Do not flip the default without that.

__global__ void k_embedding_bwd_det(
    const __nv_bfloat16* d_out,          // [BS, H] upstream grads (position-major)
    const uint32_t*      sorted_tokens,  // [BS] token ids after stable sort
    const uint32_t*      sorted_pos,     // [BS] original positions after sort
    const uint32_t*      run_offsets,    // [num_runs+1] exclusive prefix of run lengths
    const int*           d_num_runs,     // device scalar: number of unique tokens
    float*               d_weight,       // [V, H] fp32 grad accumulator
    int H
) {
    const int run = blockIdx.x;
    if (run >= *d_num_runs) return;
    const uint32_t begin = run_offsets[run];
    const uint32_t end   = run_offsets[run + 1];
    const uint32_t tok   = sorted_tokens[begin];
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t pos = sorted_pos[i];
            acc += __bfloat162float(d_out[static_cast<long long>(pos) * H + h]);
        }
        d_weight[static_cast<long long>(tok) * H + h] += acc;
    }
}

// iota fill for the position values fed into the sort.
__global__ void k_iota_u32(uint32_t* p, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = static_cast<uint32_t>(i);
}

// Turn run lengths from RunLengthEncode into the exclusive-prefix offsets the
// reduce kernel indexes runs by. Single block; num_runs <= min(V, BS) so a
// simple sequential scan on one thread is microseconds and keeps this file
// free of a second cub temp-storage round-trip.
__global__ void k_run_offsets(
    const uint32_t* run_lengths, const int* d_num_runs, uint32_t* offsets
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int n = *d_num_runs;
    uint32_t acc = 0;
    for (int i = 0; i < n; ++i) {
        offsets[i] = acc;
        acc += run_lengths[i];
    }
    offsets[n] = acc;
}

// Per-stream workspace for the deterministic path, grown on demand and
// reused across micro-steps. The persistent worker trains one body on one
// stream per process, so a single cached workspace is safe in this engine's
// process model (mirrors the AsyncWeightSaver's single-saver assumption).
namespace {
struct DetEmbedWorkspace {
    void*     cub_temp        = nullptr;
    size_t    cub_temp_bytes  = 0;
    uint32_t* pos_in          = nullptr;   // [BS] iota
    uint32_t* sorted_tokens   = nullptr;   // [BS]
    uint32_t* sorted_pos      = nullptr;   // [BS]
    uint32_t* unique_tokens   = nullptr;   // [BS] (RLE unique output)
    uint32_t* run_lengths     = nullptr;   // [BS]
    uint32_t* run_offsets     = nullptr;   // [BS+1]
    int*      num_runs        = nullptr;   // device scalar
    int       capacity_bs     = 0;

    bool ensure(int BS, cudaStream_t stream) {
        if (BS <= capacity_bs && cub_temp != nullptr) return true;
        release(stream);
        size_t sort_bytes = 0, rle_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, sort_bytes,
            static_cast<const uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            static_cast<const uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            BS, 0, 32, stream);
        cub::DeviceRunLengthEncode::Encode(
            nullptr, rle_bytes,
            static_cast<const uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), static_cast<int*>(nullptr),
            BS, stream);
        cub_temp_bytes = sort_bytes > rle_bytes ? sort_bytes : rle_bytes;
        const size_t n = static_cast<size_t>(BS);
        bool ok = true;
        ok = ok && cudaMallocAsync(&cub_temp, cub_temp_bytes, stream) == cudaSuccess;
        ok = ok && cudaMallocAsync(&pos_in,        n * sizeof(uint32_t), stream) == cudaSuccess;
        ok = ok && cudaMallocAsync(&sorted_tokens, n * sizeof(uint32_t), stream) == cudaSuccess;
        ok = ok && cudaMallocAsync(&sorted_pos,    n * sizeof(uint32_t), stream) == cudaSuccess;
        ok = ok && cudaMallocAsync(&unique_tokens, n * sizeof(uint32_t), stream) == cudaSuccess;
        ok = ok && cudaMallocAsync(&run_lengths,   n * sizeof(uint32_t), stream) == cudaSuccess;
        ok = ok && cudaMallocAsync(&run_offsets,  (n + 1) * sizeof(uint32_t), stream) == cudaSuccess;
        ok = ok && cudaMallocAsync(&num_runs,      sizeof(int), stream) == cudaSuccess;
        if (!ok) { release(stream); return false; }
        capacity_bs = BS;
        return true;
    }

    void release(cudaStream_t stream) {
        auto f = [&](void* p) { if (p) cudaFreeAsync(p, stream); };
        f(cub_temp); f(pos_in); f(sorted_tokens); f(sorted_pos);
        f(unique_tokens); f(run_lengths); f(run_offsets); f(num_runs);
        cub_temp = nullptr; pos_in = sorted_tokens = sorted_pos = nullptr;
        unique_tokens = run_lengths = run_offsets = nullptr; num_runs = nullptr;
        cub_temp_bytes = 0; capacity_bs = 0;
    }
};

DetEmbedWorkspace g_det_ws;

bool det_embed_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = std::getenv("IDA_NATIVE_DET_EMBED_BWD");
        cached = (v != nullptr && v[0] == '1') ? 1 : 0;
    }
    return cached == 1;
}
}  // namespace

void embedding_backward(
    const __nv_bfloat16* d_grad_out, const uint32_t* d_tokens,
    float* d_grad_weight, int B, int S, int H, cudaStream_t stream
) {
    const int BS = B * S;
    if (det_embed_enabled() && g_det_ws.ensure(BS, stream)) {
        k_iota_u32<<<(BS + 255) / 256, 256, 0, stream>>>(g_det_ws.pos_in, BS);
        size_t temp_bytes = g_det_ws.cub_temp_bytes;
        cub::DeviceRadixSort::SortPairs(
            g_det_ws.cub_temp, temp_bytes,
            d_tokens, g_det_ws.sorted_tokens,
            g_det_ws.pos_in, g_det_ws.sorted_pos,
            BS, 0, 32, stream);
        temp_bytes = g_det_ws.cub_temp_bytes;
        cub::DeviceRunLengthEncode::Encode(
            g_det_ws.cub_temp, temp_bytes,
            g_det_ws.sorted_tokens, g_det_ws.unique_tokens,
            g_det_ws.run_lengths, g_det_ws.num_runs,
            BS, stream);
        k_run_offsets<<<1, 1, 0, stream>>>(
            g_det_ws.run_lengths, g_det_ws.num_runs, g_det_ws.run_offsets);
        // Grid = BS (upper bound on unique tokens); blocks past num_runs
        // early-return. Avoids a host sync to read the device-side count.
        k_embedding_bwd_det<<<BS, 128, 0, stream>>>(
            d_grad_out, g_det_ws.sorted_tokens, g_det_ws.sorted_pos,
            g_det_ws.run_offsets, g_det_ws.num_runs, d_grad_weight, H);
        return;
    }
    const int T  = 128;
    dim3 grid(BS, (H + T - 1) / T);
    k_embedding_bwd<<<grid, T, 0, stream>>>(d_grad_out, d_tokens, d_grad_weight, BS, H);
}

// Per-row (per-vocab-token) gradient clip for the embedding table. Found
// 2026-07-04: a small set of extremely high-frequency, near-universal tokens
// (JSON structural punctuation on JSONL-formatted corpora — quote-colon,
// quote-comma, newline, underscore) accumulate gradient via this file's own
// atomicAdd scatter (k_embedding_bwd) far faster than any content token,
// since every occurrence across every sequence in a grad-accum window adds
// another contribution to the SAME vocab row. Measured directly: at the
// first grad-norm spike in a genesis-phase run, one embedding row alone
// accounted for ~99.99% of the GLOBAL gradient norm — identically in both
// plain int4-symmetric and mean-centered packed-FP4 attention, confirming
// this is an embedding-table issue, not an attention-quantization one.
// Independent confirmation: scripts/build_lrss_token_vocab.py's existing
// high-frequency/high-doc-coverage token detector (built for an unrelated
// purpose — LRSS content pooling) flags the same token IDs as "common" when
// pointed at this corpus's real tokenizer (50 tokens covering 70% of all
// positions). Called once per micro-step, right after embedding_backward,
// so no single micro-step's contribution to any one row can exceed
// max_row_norm — bounding the worst case across a grad-accum window of N
// micro-steps to at most N * max_row_norm, rather than letting an unbounded
// scatter-sum compound toward float32 overflow.
__global__ void k_embed_row_clip(
    float* g, int V, int H, float max_row_norm, float* stats
) {
    const int v = blockIdx.x;
    if (v >= V) return;
    __shared__ float row_normsq;
    if (threadIdx.x == 0) row_normsq = 0.0f;
    __syncthreads();
    float local = 0.0f;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        const float x = g[static_cast<std::size_t>(v) * H + h];
        local += x * x;
    }
    for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xffffffff, local, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(&row_normsq, local);
    __syncthreads();
    const float row_norm = sqrtf(row_normsq);
    if (row_norm > max_row_norm && row_norm > 0.0f) {
        if (stats) {
            atomicAdd(&stats[0], 1.0f);
            atomicMax(reinterpret_cast<int*>(&stats[1]), __float_as_int(row_norm));
        }
        const float scale = max_row_norm / row_norm;
        for (int h = threadIdx.x; h < H; h += blockDim.x) {
            g[static_cast<std::size_t>(v) * H + h] *= scale;
        }
    }
}

void embed_row_clip(
    float* d_grad_weight, int V, int H, float max_row_norm, cudaStream_t stream
) {
    k_embed_row_clip<<<V, 128, 0, stream>>>(d_grad_weight, V, H, max_row_norm, nullptr);
}

void embed_row_clip_with_stats(
    float* d_grad_weight,
    int V,
    int H,
    float max_row_norm,
    float* d_stats,
    cudaStream_t stream
) {
    k_embed_row_clip<<<V, 128, 0, stream>>>(d_grad_weight, V, H, max_row_norm, d_stats);
}

// Same row clip, but rows flagged in common_mask use common_row_norm (a
// tighter ceiling) instead of max_row_norm. See k_embed_row_clip's comment:
// the LRSS common-token detector independently flags this exact same set of
// rows as the ones scatter-accumulating fastest via k_embedding_bwd's
// atomicAdd, so they get extra protection rather than a uniform ceiling.
__global__ void k_embed_row_clip_masked(
    float* g, int V, int H, float max_row_norm, float common_row_norm,
    const uint8_t* common_mask, float* stats
) {
    const int v = blockIdx.x;
    if (v >= V) return;
    const float ceiling = (common_mask && common_mask[v]) ? common_row_norm : max_row_norm;
    __shared__ float row_normsq;
    if (threadIdx.x == 0) row_normsq = 0.0f;
    __syncthreads();
    float local = 0.0f;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        const float x = g[static_cast<std::size_t>(v) * H + h];
        local += x * x;
    }
    for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xffffffff, local, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(&row_normsq, local);
    __syncthreads();
    const float row_norm = sqrtf(row_normsq);
    if (row_norm > ceiling && row_norm > 0.0f) {
        if (stats) {
            atomicAdd(&stats[0], 1.0f);
            atomicMax(reinterpret_cast<int*>(&stats[1]), __float_as_int(row_norm));
        }
        const float scale = ceiling / row_norm;
        for (int h = threadIdx.x; h < H; h += blockDim.x) {
            g[static_cast<std::size_t>(v) * H + h] *= scale;
        }
    }
}

void embed_row_clip_masked_with_stats(
    float* d_grad_weight,
    int V,
    int H,
    float max_row_norm,
    float common_row_norm,
    const uint8_t* d_common_mask,
    float* d_stats,
    cudaStream_t stream
) {
    k_embed_row_clip_masked<<<V, 128, 0, stream>>>(
        d_grad_weight, V, H, max_row_norm, common_row_norm, d_common_mask, d_stats);
}

}  // namespace ida_native
