#include "ida_native/trainer.hpp"
#include "ida_native/checkpoint.hpp"
#include "ida_native/kernels.hpp"
#include "ida_native/ontology.hpp"
#include "ida_native/gemm_trace.hpp"
#include "ida_native/backend_contract.hpp"
#include "ida_native/status.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublasLt.h>
// Header-only C API (dlopens the injection library lazily on first push, no
// link dependency) -- lets a profiler correlate each optimizer-kernel launch
// back to which weight slot it belongs to. Added because nsys/ncu profiling
// of a real burn (2026-08-14) showed the Lion update kernel consuming over a
// third of total GPU time with no way to tell which of the ~280 slot calls
// were the expensive ones -- grid size alone doesn't disambiguate slots that
// happen to share an element count.
#include <nvtx3/nvToolsExt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "ida_native/cuda_check.hpp"

namespace ida_native {

// ─── cuBLAS check ────────────────────────────────────────────────────────────

static void cublas_check(cublasStatus_t s, const char* expr) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLAS failure in ") + expr + ": " +
                                 std::to_string(static_cast<int>(s)));
}
#define CUBLAS_CHECK(expr) cublas_check((expr), #expr)

// ─── helpers ─────────────────────────────────────────────────────────────────

static unsigned ceildiv(std::size_t n, unsigned d) {
    return static_cast<unsigned>((n + d - 1) / d);
}

static __nv_bfloat16* alloc_bf16(std::size_t n, NativeArena& arena) {
    __nv_bfloat16* p{};
    IDA_CUDA_CHECK(ida_malloc_async(&p, n * sizeof(__nv_bfloat16), arena.pool, arena.stream));
    return p;
}
static float* alloc_f32(std::size_t n, NativeArena& arena) {
    float* p{};
    IDA_CUDA_CHECK(ida_malloc_async(&p, n * sizeof(float), arena.pool, arena.stream));
    return p;
}
static std::uint8_t* alloc_u8(std::size_t n, NativeArena& arena) {
    std::uint8_t* p{};
    IDA_CUDA_CHECK(ida_malloc_async(&p, n * sizeof(std::uint8_t), arena.pool, arena.stream));
    return p;
}
static int* alloc_i32(std::size_t n, NativeArena& arena) {
    int* p{};
    IDA_CUDA_CHECK(ida_malloc_async(&p, n * sizeof(int), arena.pool, arena.stream));
    return p;
}
static __nv_fp8_e4m3* alloc_fp8_e4m3(std::size_t n, NativeArena& arena) {
    __nv_fp8_e4m3* p{};
    IDA_CUDA_CHECK(ida_malloc_async(&p, n * sizeof(__nv_fp8_e4m3), arena.pool, arena.stream));
    return p;
}

// ─── initialisation kernels ──────────────────────────────────────────────────

__global__ void k_uniform_bf16(
    __nv_bfloat16* d, std::size_t n, float scale, uint64_t seed
) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t s = seed ^ (i * 6364136223846793005ULL + 1442695040888963407ULL);
    s ^= s >> 33; s *= 0xff51afd7ed558ccdULL;
    s ^= s >> 33; s *= 0xc4ceb9fe1a85ec53ULL;
    s ^= s >> 33;
    // Interpret the full 64-bit hash as SIGNED so v spans [-1, 1). The
    // original `s >> 1` dropped into an always-positive 63-bit value —
    // v in [0, 1), i.e. every weight AND embedding uniform in [0, scale],
    // never negative. All-positive operands turn every genesis GEMM into a
    // coherent (mean-shifted) sum instead of a zero-mean random walk:
    // amplification ~N instead of ~sqrt(N). Measured 2026-07-09 on the
    // 16-layer AI body: gate_out absmax 58 (Xavier predicts ~9), down_proj
    // output absmax 1248 (predicts ~15), residual-stream rms 1248 entering
    // every layer past L0 — and, downstream, the L0 rmsnorm-backward
    // x10,000 gradient amplification (embedding rms 0.0115 vs mid-stack
    // 1248) that this week's entire clip stack was built to contain.
    float v = static_cast<float>(static_cast<int64_t>(s)) *
              (1.0f / static_cast<float>(INT64_MAX));
    d[i] = __float2bfloat16(v * scale);
}
__global__ void k_ones_bf16(__nv_bfloat16* d, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) d[i] = __float2bfloat16(1.0f);
}
__global__ void k_zeros_f32(float* d, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) d[i] = 0.0f;
}

// Debug only: absmax + nonfinite count of a BF16 activation buffer.
__global__ void k_bf16_absmax_nonfinite(
    const __nv_bfloat16* x, std::size_t n, float* out /*[2]: absmax, nonfinite*/
) {
    float local_absmax = 0.0f, local_bad = 0.0f;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float v = __bfloat162float(x[i]);
        if (!isfinite(v)) { local_bad += 1.0f; continue; }
        local_absmax = fmaxf(local_absmax, fabsf(v));
    }
    atomicAdd(&out[1], local_bad);
    atomicMax(reinterpret_cast<int*>(&out[0]), __float_as_int(local_absmax));
}

__global__ void k_f32_absmax_nonfinite(
    const float* x, std::size_t n, float* out /*[2]: absmax, nonfinite*/
) {
    float local_absmax = 0.0f, local_bad = 0.0f;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float v = x[i];
        if (!isfinite(v)) { local_bad += 1.0f; continue; }
        local_absmax = fmaxf(local_absmax, fabsf(v));
    }
    atomicAdd(&out[1], local_bad);
    atomicMax(reinterpret_cast<int*>(&out[0]), __float_as_int(local_absmax));
}

static bool gradnorm_debug_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("IDA_NATIVE_GRADNORM_DEBUG");
        return e && e[0] == '1';
    }();
    return v;
}

// IDA_FP8_SNAP_FUSE=0: disable the fused fp8_quant_and_transpose_e4m3 kernel in
// the backward recompute snap path.  Reads BF16 once, writes act8 (for the
// forward GEMM) and snap_X (transposed, for the dW GEMM) in one pass,
// eliminating the act8 HBM read-back that the two-step path requires.
static bool fp8_snap_fuse_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("IDA_FP8_SNAP_FUSE");
        return !(e && e[0] == '0');
    }();
    return v;
}

static bool lss_feedback_skip_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("IDA_NATIVE_LSS_FEEDBACK_SKIP");
        return !(e && e[0] == '0');
    }();
    return v;
}

static float lss_feedback_aux_max() {
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_LSS_FEEDBACK_AUX_MAX");
        if (!e || !e[0]) return 8.0f;
        char* end = nullptr;
        const float parsed = std::strtof(e, &end);
        return (end == e || parsed < 0.0f) ? 8.0f : parsed;
    }();
    return v;
}

static int lss_feedback_max_tail_layers() {
    static const int v = [] {
        const char* e = std::getenv("IDA_NATIVE_LSS_FEEDBACK_MAX_LAYERS");
        const int parsed = (e && e[0]) ? std::atoi(e) : 1;
        return std::max(0, std::min(parsed, 4));
    }();
    return v;
}

static int lss_feedback_min_optimizer_steps() {
    static const int v = [] {
        const char* e = std::getenv("IDA_NATIVE_LSS_FEEDBACK_MIN_OPT_STEPS");
        const int parsed = (e && e[0]) ? std::atoi(e) : 4;
        return std::max(0, parsed);
    }();
    return v;
}

static float lss_feedback_residual_scale() {
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_LSS_FEEDBACK_RESIDUAL_SCALE");
        if (!e || !e[0]) return 0.5f;
        char* end = nullptr;
        const float parsed = std::strtof(e, &end);
        if (end == e) return 0.5f;
        return std::max(0.0f, std::min(parsed, 1.0f));
    }();
    return v;
}

// PSS Stage 2: shadow-mode tail-layer FFN-output predictor. Activation is
// per-burn: request.pss_pred_rank (wrapper-supplied, family-gated) wins,
// with the IDA_NATIVE_PSS_PRED / _RANK env pair as the fallback for direct
// launches -- resolved once in allocate_lattice_weights, after which every
// hook gates on w.pss_pred_rank / the allocated buffers. No static-cached
// env helper here: the socket worker serves many burns and a process-wide
// cache cannot represent a per-burn decision.
// Coverage threshold: an element counts as "covered" when
// |pred-real| < eps * max(|real|, 1e-6). Purely a scoring/telemetry knob in
// Stage 2 -- it never gates training or body consumption.
static float pss_pred_eps() {
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_PSS_PRED_EPS");
        if (!e || !e[0]) return 0.5f;
        char* end = nullptr;
        const float parsed = std::strtof(e, &end);
        return (end == e || parsed < 0.0f) ? 0.5f : parsed;
    }();
    return v;
}
static float pss_pred_aux_weight(const NativeRequest& request) {
    if (request.pss_pred_aux_weight >= 0.0f) return request.pss_pred_aux_weight;
    const char* e = std::getenv("IDA_NATIVE_PSS_PRED_AUX_WEIGHT");
    if (!e || !e[0]) return 0.01f;
    char* end = nullptr;
    const float parsed = std::strtof(e, &end);
    return (end == e || parsed < 0.0f) ? 0.01f : parsed;
}

static bool pss_state_persistence_requested() {
    const char* keep = std::getenv("IDA_NATIVE_PSS_PRED_STATE_KEEP_ON_GENESIS");
    const char* init = std::getenv("IDA_NATIVE_PSS_PRED_STATE_INIT_PATH");
    const char* out = std::getenv("IDA_NATIVE_PSS_PRED_STATE_OUT");
    return (keep && keep[0] == '1') || (init && init[0]) || (out && out[0]);
}

// IDA_NATIVE_PSS_CONDITIONING:
//   auto/unset: condition PSS only for routed/expert bodies
//   1:          force conditioning when the routed hidden surface exists
//   0:          force the original dense pre-tail-hidden predictor input
//
// This is intentionally not static-cached: the socket worker can serve burns
// with different architecture contracts in the same process.
static bool pss_conditioning_enabled(
    const NativeRequest& request, int num_routes, int num_experts) {
    const std::string& mode = request.pss_conditioning_mode;
    if (!mode.empty()) {
        if (mode == "0" || mode == "off" || mode == "false") return false;
        if (mode == "1" || mode == "on" || mode == "true") return true;
    } else {
        const char* e = std::getenv("IDA_NATIVE_PSS_CONDITIONING");
        if (e && e[0]) {
            if (e[0] == '0') return false;
            if (e[0] == '1') return true;
        }
    }
    return num_routes > 0 || num_experts > 0;
}

// IDA_NATIVE_PSS_AUX_NORMALIZE:
//   auto/unset: use relative-error aux on routed/expert or wide bodies
//   1:          force relative-error aux
//   0:          force the original raw-MSE aux
//
// Dense Edge/Swift learned under raw MSE; AI/MoE-scale bodies showed healthy
// trunk GN with PSS stuck at random int2 agreement, so auto shifts only the
// high-width/routed cases to target-normalized learning pressure.
static bool pss_aux_normalize_enabled(
    const NativeRequest& request, int hidden_size, int num_routes, int num_experts) {
    const std::string& mode = request.pss_aux_normalize_mode;
    if (!mode.empty()) {
        if (mode == "0" || mode == "off" || mode == "false") return false;
        if (mode == "1" || mode == "on" || mode == "true") return true;
    } else {
        const char* e = std::getenv("IDA_NATIVE_PSS_AUX_NORMALIZE");
        if (e && e[0]) {
            if (e[0] == '0') return false;
            if (e[0] == '1') return true;
        }
    }
    return hidden_size >= 1024 || num_routes > 0 || num_experts > 0;
}

static std::string pss_policy_mode(
    const std::string& request_mode, const char* env_name) {
    if (!request_mode.empty()) return request_mode;
    const char* e = std::getenv(env_name);
    return (e && e[0]) ? std::string(e) : std::string("auto");
}

// Support transition governor: adjusts LRSS/LSS/PSS learning pressure between
// optimizer windows. It is deliberately a transition, not a static throttle:
// scale lowers only while the target space expands faster than reconstruction
// improves, and recovers when the relative error is low and PSS is not
// dominating. Env gates exist for ablation and emergency disable.
static bool support_transition_enabled() {
    const char* e = std::getenv("IDA_NATIVE_SUPPORT_TRANSITION");
    return !(e && e[0] == '0');
}
static float support_transition_step_down() {
    const char* e = std::getenv("IDA_NATIVE_SUPPORT_TRANSITION_STEP_DOWN");
    if (!e || !e[0]) return 0.70f;
    char* end = nullptr;
    const float parsed = std::strtof(e, &end);
    return (end == e) ? 0.70f : std::max(0.05f, std::min(parsed, 1.0f));
}
static float support_transition_step_up() {
    const char* e = std::getenv("IDA_NATIVE_SUPPORT_TRANSITION_STEP_UP");
    if (!e || !e[0]) return 1.10f;
    char* end = nullptr;
    const float parsed = std::strtof(e, &end);
    return (end == e) ? 1.10f : std::max(1.0f, std::min(parsed, 2.0f));
}
static float support_transition_floor() {
    const char* e = std::getenv("IDA_NATIVE_SUPPORT_TRANSITION_FLOOR");
    if (!e || !e[0]) return 0.25f;
    char* end = nullptr;
    const float parsed = std::strtof(e, &end);
    return (end == e) ? 0.25f : std::max(0.0f, std::min(parsed, 1.0f));
}
static float support_transition_target_growth_max() {
    const char* e = std::getenv("IDA_NATIVE_SUPPORT_TRANSITION_TARGET_GROWTH_MAX");
    if (!e || !e[0]) return 1.50f;
    char* end = nullptr;
    const float parsed = std::strtof(e, &end);
    return (end == e || parsed < 1.0f) ? 1.50f : parsed;
}
static float support_transition_recover_rmse() {
    const char* e = std::getenv("IDA_NATIVE_SUPPORT_TRANSITION_RECOVER_RMSE");
    if (!e || !e[0]) return 0.65f;
    char* end = nullptr;
    const float parsed = std::strtof(e, &end);
    return (end == e || parsed < 0.0f) ? 0.65f : parsed;
}
static float support_transition_dominant_frac_max() {
    const char* e = std::getenv("IDA_NATIVE_SUPPORT_TRANSITION_DOM_FRAC_MAX");
    if (!e || !e[0]) return 0.35f;
    char* end = nullptr;
    const float parsed = std::strtof(e, &end);
    return (end == e) ? 0.35f : std::max(0.0f, std::min(parsed, 1.0f));
}

static float pss_pred_up_init_gain() {
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_PSS_PRED_UP_INIT_GAIN");
        const float legacy_default = [] {
            const char* fix = std::getenv("IDA_NATIVE_NONRELU_GAIN_FIX");
            return (fix && fix[0] == '1') ? (1.0f / std::sqrt(2.0f)) : 1.0f;
        }();
        if (!e || !e[0]) return legacy_default;
        char* end = nullptr;
        const float parsed = std::strtof(e, &end);
        return (end == e || parsed < 0.0f) ? legacy_default : parsed;
    }();
    return v;
}

// PSS Stage 4: governor-gated engagement. Extends the existing
// feedback-controller pattern (lss_feedback_* above) to the Stage 2
// predictor. This is the one stage where PSS actually changes what the
// trunk computes -- everything through Stage 3 was shadow/external. Master
// gate is separate from IDA_NATIVE_PSS_PRED: the predictor can run in
// shadow mode (scored, never blended) with the governor off, which is the
// default and the only mode validated on GPU so far.
static bool pss_governor_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("IDA_NATIVE_PSS_GOVERNOR");
        return e && e[0] == '1';
    }();
    return v;
}
// Nudge-in threshold: confidence at or above this ramps engagement up by
// IDA_NATIVE_PSS_ENGAGE_STEP per window, capped at IDA_NATIVE_PSS_ENGAGE_MAX.
static float pss_engage_in() {
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_PSS_ENGAGE_IN");
        if (!e || !e[0]) return 0.60f;
        char* end = nullptr;
        const float parsed = std::strtof(e, &end);
        return (end == e) ? 0.60f : std::max(0.0f, std::min(parsed, 1.0f));
    }();
    return v;
}
// Pull-out threshold: confidence below this snaps engagement to 0
// immediately (no ramp-down). Hysteresis requires this strictly below
// pss_engage_in() -- validated at read time, not just by convention, so a
// misconfigured override fails loud instead of oscillating silently.
static float pss_engage_out() {
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_PSS_ENGAGE_OUT");
        float parsed = 0.45f;
        if (e && e[0]) {
            char* end = nullptr;
            const float p = std::strtof(e, &end);
            if (end != e) parsed = std::max(0.0f, std::min(p, 1.0f));
        }
        if (parsed >= pss_engage_in()) {
            std::fprintf(stderr,
                "[ida_native_train] FATAL: IDA_NATIVE_PSS_ENGAGE_OUT (%.4f) must be "
                "strictly less than IDA_NATIVE_PSS_ENGAGE_IN (%.4f) -- hysteresis "
                "requires a gap, or the engagement fraction will chatter at the "
                "boundary every window.\n", parsed, pss_engage_in());
            std::exit(78);
        }
        return parsed;
    }();
    return v;
}
static float pss_engage_step() {
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_PSS_ENGAGE_STEP");
        if (!e || !e[0]) return 0.05f;
        char* end = nullptr;
        const float parsed = std::strtof(e, &end);
        return (end == e || parsed <= 0.0f) ? 0.05f : parsed;
    }();
    return v;
}
// Deliberately conservative default: full engagement (1.0) is not the
// starting ceiling. Raising this is an operator decision that should cite a
// calibration_ledger.jsonl row (Stage 3) showing the confidence claim
// actually holds up out of sample -- see docs/predictive-supersampling-int2.md.
static float pss_engage_max() {
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_PSS_ENGAGE_MAX");
        if (!e || !e[0]) return 0.25f;
        char* end = nullptr;
        const float parsed = std::strtof(e, &end);
        return (end == e) ? 0.25f : std::max(0.0f, std::min(parsed, 1.0f));
    }();
    return v;
}
static int pss_engage_min_optimizer_steps() {
    static const int v = [] {
        const char* e = std::getenv("IDA_NATIVE_PSS_ENGAGE_MIN_OPT_STEPS");
        const int parsed = (e && e[0]) ? std::atoi(e) : 4;
        return std::max(0, parsed);
    }();
    return v;
}
// Number of H-elements per int2 symbol scale. 0 (default) = one global scale
// for the whole tensor, exactly the pre-2026-08-13 behavior. N>0 = one scale
// per N-element group along H, borrowing the block-scaled-quantization
// convention used by the private deployment precision package -- an ablation
// target, not an assumed-correct value.
static int pss_int2_block_size() {
    static const int v = [] {
        const char* e = std::getenv("IDA_NATIVE_PSS_INT2_BLOCK_SIZE");
        const int parsed = (e && e[0]) ? std::atoi(e) : 0;
        return parsed > 0 ? parsed : 0;
    }();
    return v;
}
// Number of trunk optimizer steps PSS's own gradient accumulates across
// before its Lion update fires (see StepBuffers::pss_window_accum_down/up
// for the full design comment). 1 (default) = today's behavior, PSS updates
// every trunk step exactly as before.
static int pss_accum_steps() {
    static const int v = [] {
        const char* e = std::getenv("IDA_NATIVE_PSS_ACCUM_STEPS");
        const int parsed = (e && e[0]) ? std::atoi(e) : 1;
        return parsed > 0 ? parsed : 1;
    }();
    return v;
}
// Debug-only: forces the confidence value the governor's state machine sees,
// bypassing the real measurement. Exists so the ramp/snap/hysteresis logic
// can be proven with a deterministic square wave instead of waiting on real
// training dynamics to produce specific confidence values on demand.
//
// IDA_NATIVE_PSS_CONFIDENCE_OVERRIDE_FILE (preferred): re-read fresh every
// call, no caching -- a companion script can rewrite the file's contents
// between optimizer windows while the trainer runs continuously, producing
// a genuine time-varying square wave within one process. This is the only
// way to exercise ramp-then-hold-then-snap in a single run:
// IDA_NATIVE_PSS_ENGAGE_MIN_OPT_STEPS and sb.pss_engaged_frac are both
// run-scoped state that a fresh process restart would reset to zero, so
// separate constant-override runs cannot observe the transition itself.
//
// IDA_NATIVE_PSS_CONFIDENCE_OVERRIDE (legacy): read once, cached -- only
// ever produces a constant value for the run's lifetime. Falls back to this
// if the file variant isn't set, kept for the simpler "does override work
// at all" case.
static bool pss_confidence_override(float* out) {
    static const char* file_path = std::getenv("IDA_NATIVE_PSS_CONFIDENCE_OVERRIDE_FILE");
    if (file_path && file_path[0]) {
        std::ifstream in(file_path);
        if (in) {
            float parsed = 0.0f;
            if (in >> parsed) {
                *out = std::max(0.0f, std::min(parsed, 1.0f));
                return true;
            }
        }
    }
    static const char* e = std::getenv("IDA_NATIVE_PSS_CONFIDENCE_OVERRIDE");
    if (!e || !e[0]) return false;
    char* end = nullptr;
    const float parsed = std::strtof(e, &end);
    if (end == e) return false;
    *out = std::max(0.0f, std::min(parsed, 1.0f));
    return true;
}

// Max per-vocab-row L2 gradient norm for the embedding table before it gets
// scaled down (see embedding.cu's k_embed_row_clip). 0 or unset = disabled
// (existing behavior). Found 2026-07-04: a handful of high-frequency,
// near-universal tokens (JSON structural punctuation on JSONL corpora)
// scatter-accumulate gradient via embedding_backward's atomicAdd far faster
// than any content token, at times accounting for ~99.99% of the global
// grad-norm — identically across every attention/quantization format tested,
// confirming this is upstream of attention entirely.
static float embed_row_clip_threshold() {
    static const float v = [] {
        // Era 13 default: OFF.  The clip was containment for the
        // all-positive-init gradient explosion; post-fix ablation
        // (2026-07-10, both families) shows removing it changes loss by
        // <=0.004 with zero skips.  IDA_NATIVE_EMBED_ROW_CLIP=<t> re-arms
        // it for ablation or emergencies.
        const char* e = std::getenv("IDA_NATIVE_EMBED_ROW_CLIP");
        if (!e || !e[0]) return 0.0f;
        char* end = nullptr;
        const float parsed = std::strtof(e, &end);
        return (end == e) ? 0.0f : parsed;
    }();
    return v;
}

// LRSS-informed tighter ceiling for rows flagged "common" by
// scripts/build_lrss_token_vocab.py's frequency+doc-frac detector — the same
// detector the PyTorch-era LRSSCommonTokenFilter used to down-weight these
// tokens (function words, punctuation, single-letter BPE fragments) out of
// content pooling. 0/unset = disabled (lab-test flag; the blanket
// embed_row_clip_threshold above still applies to every row either way).
static float lrss_common_clip_threshold() {
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_LRSS_COMMON_CLIP");
        if (!e || !e[0]) return 0.0f;
        char* end = nullptr;
        const float parsed = std::strtof(e, &end);
        return (end == e) ? 0.0f : parsed;
    }();
    return v;
}

// Loads artifacts/lrss/common_token_ids.json's "token_ids" array (or
// IDA_NATIVE_LRSS_COMMON_TOKEN_IDS_PATH if set) into a device-resident
// per-vocab-row mask, once per process. Returns nullptr if the file is
// missing/empty/unparsable — treated as "mask unavailable", not an error,
// so callers fall back to the blanket clip.
static const uint8_t* lrss_common_token_mask(int V) {
    // A CUDA pointer is valid only on the device that allocated it. The
    // original process-wide singleton was correct for one GPU but made the
    // second model-parallel stage read GPU0 memory as if it belonged to GPU1.
    static std::map<std::pair<int, int>, const uint8_t*> masks;
    static std::mutex masks_mutex;
    std::lock_guard<std::mutex> lock(masks_mutex);
    int device = -1;
    IDA_CUDA_CHECK(cudaGetDevice(&device));
    const auto key = std::make_pair(device, V);
    if (const auto it = masks.find(key); it != masks.end()) return it->second;

    const char* path_env = std::getenv("IDA_NATIVE_LRSS_COMMON_TOKEN_IDS_PATH");
    const std::string path =
        (path_env && path_env[0]) ? path_env : "artifacts/lrss/common_token_ids.json";
    std::ifstream f(path);
    if (!f.good()) {
        std::fprintf(stderr, "[lrss] common-token file not found: %s (mask disabled)\n",
                     path.c_str());
        masks.emplace(key, nullptr);
        return nullptr;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    const auto key_pos = text.find("\"token_ids\"");
    const auto lb = (key_pos == std::string::npos) ? std::string::npos : text.find('[', key_pos);
    const auto rb = (lb == std::string::npos) ? std::string::npos : text.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) {
        std::fprintf(stderr, "[lrss] %s: no parsable \"token_ids\" array (mask disabled)\n",
                     path.c_str());
        masks.emplace(key, nullptr);
        return nullptr;
    }
    std::vector<uint8_t> h_mask(static_cast<std::size_t>(V), 0);
    std::stringstream ids(text.substr(lb + 1, rb - lb - 1));
    std::string tok;
    int n_flagged = 0;
    while (std::getline(ids, tok, ',')) {
        char* end = nullptr;
        const long id = std::strtol(tok.c_str(), &end, 10);
        if (end != tok.c_str() && id >= 0 && id < static_cast<long>(V)) {
            h_mask[static_cast<std::size_t>(id)] = 1;
            ++n_flagged;
        }
    }
    if (n_flagged == 0) {
        masks.emplace(key, nullptr);
        return nullptr;
    }
    uint8_t* dev = nullptr;
    IDA_CUDA_CHECK(cudaMalloc(&dev, h_mask.size()));
    IDA_CUDA_CHECK(cudaMemcpy(dev, h_mask.data(), h_mask.size(), cudaMemcpyHostToDevice));
    std::fprintf(stderr, "[lrss] loaded %d common-token ids on cuda:%d from %s\n",
                 n_flagged, device, path.c_str());
    masks.emplace(key, dev);
    return dev;
}

// forward()-scoped counter so layer_forward_body can label its debug prints
// with which micro-step they belong to (correlate with the gradnorm spike
// print, which fires later in the same micro-step's backward pass).
static int g_debug_micro_step = 0;

static void debug_print_scalar(const char* label, const float* d_x, cudaStream_t stream) {
    float h = 0.0f;
    cudaMemcpyAsync(&h, d_x, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::fprintf(stderr, "[scalestats] micro_step=%d %-14s value=%.10g\n",
                 g_debug_micro_step, label, h);
}

static void debug_print_activation_stats(
    const char* label, const __nv_bfloat16* d_x, std::size_t n, cudaStream_t stream
) {
    float* d_stats = nullptr;
    cudaMallocAsync(&d_stats, 2 * sizeof(float), stream);
    cudaMemsetAsync(d_stats, 0, 2 * sizeof(float), stream);
    const std::size_t b = (n + 255) / 256;
    const unsigned blocks = static_cast<unsigned>(b < 1024 ? b : 1024);
    k_bf16_absmax_nonfinite<<<blocks, 256, 0, stream>>>(d_x, n, d_stats);
    float h[2] = {0.0f, 0.0f};
    cudaMemcpyAsync(h, d_stats, sizeof(h), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_stats, stream);
    std::fprintf(stderr, "[actstats] micro_step=%d %-14s absmax=%g nonfinite=%.0f\n",
                 g_debug_micro_step, label, h[0], h[1]);
}

static void debug_print_f32_activation_stats(
    const char* label, const float* d_x, std::size_t n, cudaStream_t stream
) {
    float* d_stats = nullptr;
    cudaMallocAsync(&d_stats, 2 * sizeof(float), stream);
    cudaMemsetAsync(d_stats, 0, 2 * sizeof(float), stream);
    const std::size_t b = (n + 255) / 256;
    const unsigned blocks = static_cast<unsigned>(b < 1024 ? b : 1024);
    k_f32_absmax_nonfinite<<<blocks, 256, 0, stream>>>(d_x, n, d_stats);
    float h[2] = {0.0f, 0.0f};
    cudaMemcpyAsync(h, d_stats, sizeof(h), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_stats, stream);
    std::fprintf(stderr, "[actstats-f32] micro_step=%d %-14s absmax=%g nonfinite=%.0f\n",
                 g_debug_micro_step, label, h[0], h[1]);
}

static void debug_print_f32_vector_stats(
    int opt_step, const char* label, const float* d_x, std::size_t n, cudaStream_t stream
) {
    std::vector<float> h(n);
    cudaMemcpyAsync(h.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    double sum = 0.0, sumsq = 0.0;
    float min_v = INFINITY, max_v = -INFINITY, absmax = 0.0f;
    std::size_t nonfinite = 0;
    for (float v : h) {
        if (!std::isfinite(v)) { ++nonfinite; continue; }
        sum += v;
        sumsq += static_cast<double>(v) * v;
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        absmax = std::max(absmax, std::fabs(v));
    }
    const std::size_t finite = n - nonfinite;
    const double denom = static_cast<double>(std::max<std::size_t>(1, finite));
    std::fprintf(stderr,
        "[lrss-bias-debug] opt_step=%d %-12s n=%zu mean=%.9g rms=%.9g "
        "min=%.9g max=%.9g absmax=%.9g nonfinite=%zu\n",
        opt_step, label, n, sum / denom, std::sqrt(sumsq / denom),
        min_v, max_v, absmax, nonfinite);
}

static void debug_print_bf16_vector_stats(
    int opt_step, const char* label, const __nv_bfloat16* d_x, std::size_t n,
    cudaStream_t stream
) {
    std::vector<__nv_bfloat16> raw(n);
    cudaMemcpyAsync(raw.data(), d_x, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::vector<float> h(n);
    std::transform(raw.begin(), raw.end(), h.begin(),
                   [](__nv_bfloat16 v) { return __bfloat162float(v); });
    double sum = 0.0, sumsq = 0.0;
    float min_v = INFINITY, max_v = -INFINITY, absmax = 0.0f;
    std::size_t nonfinite = 0;
    for (float v : h) {
        if (!std::isfinite(v)) { ++nonfinite; continue; }
        sum += v;
        sumsq += static_cast<double>(v) * v;
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        absmax = std::max(absmax, std::fabs(v));
    }
    const std::size_t finite = n - nonfinite;
    const double denom = static_cast<double>(std::max<std::size_t>(1, finite));
    std::fprintf(stderr,
        "[lrss-bias-debug] opt_step=%d %-12s n=%zu mean=%.9g rms=%.9g "
        "min=%.9g max=%.9g absmax=%.9g nonfinite=%zu\n",
        opt_step, label, n, sum / denom, std::sqrt(sumsq / denom),
        min_v, max_v, absmax, nonfinite);
}

// ── Cross-backend intermediate dump (Phase 4, 2026-07-22) ───────────────────
// Every other debug knob in this file (gradnorm_debug_enabled() and its ~80
// call sites, lrss-bias-debug above) prints summary STATISTICS to stderr --
// none of them write full tensor VALUES to a file for external diffing
// against a PyTorch reference (the gap scripts/native_parity_check.py's own
// harness needs closed to extend past the MoE router it started with).
// IDA_NATIVE_DUMP_INTERMEDIATES=<path> appends one JSON line per call to
// that file (JSONL, not a single JSON array, so a crashed/killed run still
// leaves whatever was written so far readable) -- {"name", "micro_step",
// "shape", "values"}. Deliberately opt-in and unbounded-size per call (no
// truncation): this is for small, fixed verification fixtures (matching
// scripts/native_parity_check.py's own H=8-scale dimensions), not a
// production-scale telemetry stream -- do not enable on a real production
// burn with real tensor widths.
static bool dump_intermediates_enabled() {
#if !IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
    return false;
#else
    static const bool v = [] {
        const char* e = std::getenv("IDA_NATIVE_DUMP_INTERMEDIATES");
        return e && e[0];
    }();
    return v;
#endif
}

static std::string dump_intermediates_path() {
#if !IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
    return {};
#else
    static const std::string v = [] {
        const char* e = std::getenv("IDA_NATIVE_DUMP_INTERMEDIATES");
        return std::string(e ? e : "");
    }();
    return v;
#endif
}

static void dump_intermediate_values(
    const char* name, const std::vector<float>& host,
    const std::vector<std::int64_t>& shape, int micro_step
) {
    if (!dump_intermediates_enabled()) return;
    // Safety cap: this mechanism is designed for small, fixed verification
    // fixtures (H=8-scale, matching scripts/native_parity_check.py's own
    // dims), not production-scale tensors -- confirmed the hard way
    // (2026-07-22): one real MoE burn's "pooled" tensor alone was
    // [16384, 4096], and the resulting file reached ~50 GB in seconds.
    // Skip (once-per-name warning, not silent) rather than let an operator
    // repeat that mistake.
    constexpr std::size_t kMaxDumpElems = 1'000'000;  // ~4 MB per tensor as f32 text
    if (host.size() > kMaxDumpElems) {
        static std::vector<std::string> warned;
        if (std::find(warned.begin(), warned.end(), name) == warned.end()) {
            warned.push_back(name);
            std::fprintf(stderr,
                "[dump-intermediates] skipping \"%s\": %zu elements exceeds the "
                "%zu-element safety cap (this mechanism is for small fixed "
                "verification fixtures, not production-scale tensors)\n",
                name, host.size(), kMaxDumpElems);
        }
        return;
    }
    std::ofstream out(dump_intermediates_path(), std::ios::app);
    if (!out) return;
    out << "{\"name\":\"" << name << "\",\"micro_step\":" << micro_step << ",\"shape\":[";
    for (std::size_t i = 0; i < shape.size(); ++i) out << (i ? "," : "") << shape[i];
    out << "],\"values\":[";
    for (std::size_t i = 0; i < host.size(); ++i) out << (i ? "," : "") << host[i];
    out << "]}\n";
}

// bf16 device tensor -> host -> JSONL row. `shape` is the tensor's own
// dims (e.g. {BS, H}); pass {} to just record the flat element count.
static void dump_intermediate_bf16(
    const char* name, const __nv_bfloat16* d_x, std::size_t n,
    std::vector<std::int64_t> shape, int micro_step, cudaStream_t stream
) {
    if (!dump_intermediates_enabled()) return;
    std::vector<__nv_bfloat16> raw(n);
    cudaMemcpyAsync(raw.data(), d_x, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::vector<float> h(n);
    std::transform(raw.begin(), raw.end(), h.begin(),
                   [](__nv_bfloat16 v) { return __bfloat162float(v); });
    if (shape.empty()) shape.push_back(static_cast<std::int64_t>(n));
    dump_intermediate_values(name, h, shape, micro_step);
}

static void dump_intermediate_f32(
    const char* name, const float* d_x, std::size_t n,
    std::vector<std::int64_t> shape, int micro_step, cudaStream_t stream
) {
    if (!dump_intermediates_enabled()) return;
    std::vector<float> h(n);
    cudaMemcpyAsync(h.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (shape.empty()) shape.push_back(static_cast<std::int64_t>(n));
    dump_intermediate_values(name, h, shape, micro_step);
}

// Direct measurement (not an extreme-value-theory estimate) of: mean,
// variance/std, and what fraction of the ACTUAL tensor would land in the
// "rounds to exactly zero" bucket under int2 (N=1, boundary=0.5*amax) vs
// int4 (N=7, boundary=amax/14) linear-symmetric quantization. amax is
// whatever was already computed for this call (read fresh, not recomputed).
__global__ void k_qk_dist_stats(
    const __nv_bfloat16* x, std::size_t n, const float* amax,
    float* out /* [4]: sum, sum_sq, count_zero_int2, count_zero_int4 */
) {
    const float a = *amax;
    const float int2_boundary = 0.5f * a;         // N=1: |x| < 0.5*amax -> code 0
    const float int4_boundary = a / 14.0f;        // N=7: |x| < amax/14  -> code 0
    float local_sum = 0.0f, local_sumsq = 0.0f, local_c2 = 0.0f, local_c4 = 0.0f;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float v = __bfloat162float(x[i]);
        local_sum += v;
        local_sumsq += v * v;
        if (fabsf(v) < int2_boundary) local_c2 += 1.0f;
        if (fabsf(v) < int4_boundary) local_c4 += 1.0f;
    }
    atomicAdd(&out[0], local_sum);
    atomicAdd(&out[1], local_sumsq);
    atomicAdd(&out[2], local_c2);
    atomicAdd(&out[3], local_c4);
}

static void debug_print_qk_distribution(
    const char* label, const __nv_bfloat16* d_x, std::size_t n,
    const float* d_amax, cudaStream_t stream
) {
    float* d_stats = nullptr;
    cudaMallocAsync(&d_stats, 4 * sizeof(float), stream);
    cudaMemsetAsync(d_stats, 0, 4 * sizeof(float), stream);
    const std::size_t b = (n + 255) / 256;
    const unsigned blocks = static_cast<unsigned>(b < 1024 ? b : 1024);
    k_qk_dist_stats<<<blocks, 256, 0, stream>>>(d_x, n, d_amax, d_stats);
    float amax_h = 0.0f;
    float h[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    cudaMemcpyAsync(&amax_h, d_amax, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h, d_stats, sizeof(h), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_stats, stream);
    const double nn = static_cast<double>(n);
    const double mean = h[0] / nn;
    const double var = h[1] / nn - mean * mean;
    const double std_ = var > 0.0 ? std::sqrt(var) : 0.0;
    std::fprintf(stderr,
        "[qkdist] micro_step=%d %-8s n=%zu mean=%.6g std=%.6g amax=%.6g amax/std=%.4g "
        "frac_zero_int2=%.4f frac_zero_int4=%.4f\n",
        g_debug_micro_step, label, n, mean, std_, static_cast<double>(amax_h),
        std_ > 0.0 ? static_cast<double>(amax_h) / std_ : 0.0,
        static_cast<double>(h[2]) / nn, static_cast<double>(h[3]) / nn);
}

// Mean-centered-quantization-specific diagnostic: measures, against the
// mean/scale THIS call is about to encode with, (a) what fraction of
// elements actually saturate (clip to the max code), and (b) the CURRENT
// batch's own fresh mean/std/max-deviation, so the calibrated window
// (kFp4Max/scale, derived from last call's max_abs_dev) can be compared
// against what the data actually needs this call. Tests the hypothesis that
// centered mode's much narrower window (max_abs_dev, ~5) vs uncentered's
// wide amax-based window (~24) saturates far more often as the true
// distribution drifts during early training, biasing next call's mean/scale
// in a way that compounds rather than damps.
__global__ void k_fp4_centered_clip_stats(
    const __nv_bfloat16* x, std::size_t n,
    const float* mean, const float* scale,
    float* out /* [4]: count_clip, sum, sumsq, max_abs_dev_fresh */
) {
    const float m = *mean;
    const float sc = *scale;
    float local_clip = 0.0f, local_sum = 0.0f, local_sumsq = 0.0f, local_maxdev = 0.0f;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float v = __bfloat162float(x[i]);
        const float dev = fabsf(v - m);
        if (dev * sc > 7.0f) local_clip += 1.0f;   // kFp4Max
        local_sum += v;
        local_sumsq += v * v;
        local_maxdev = fmaxf(local_maxdev, dev);
    }
    atomicAdd(&out[0], local_clip);
    atomicAdd(&out[1], local_sum);
    atomicAdd(&out[2], local_sumsq);
    atomicMax(reinterpret_cast<int*>(&out[3]), __float_as_int(local_maxdev));
}

// Per-vocab-row L2 norm-squared of the embedding gradient — localizes
// whether a spike is dominated by one (or a few) specific token rows
// (e.g. a very frequent token like padding, accumulating via the
// atomicAdd scatter in embedding_backward across every position that uses
// it) versus being spread broadly across the vocabulary.
__global__ void k_embed_row_normsq(
    const float* g, int V, int H, float* row_normsq /* [V], caller zeroes */
) {
    const int v = blockIdx.x;
    if (v >= V) return;
    float local = 0.0f;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        const float x = g[static_cast<std::size_t>(v) * H + h];
        local += x * x;
    }
    for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xffffffff, local, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(&row_normsq[v], local);
}

static void debug_print_embed_row_breakdown(
    const float* d_grad_embed, int V, int H, cudaStream_t stream
) {
    float* d_row_normsq = nullptr;
    cudaMallocAsync(&d_row_normsq, static_cast<std::size_t>(V) * sizeof(float), stream);
    cudaMemsetAsync(d_row_normsq, 0, static_cast<std::size_t>(V) * sizeof(float), stream);
    k_embed_row_normsq<<<V, 32, 0, stream>>>(d_grad_embed, V, H, d_row_normsq);
    std::vector<float> h(V);
    cudaMemcpyAsync(h.data(), d_row_normsq, static_cast<std::size_t>(V) * sizeof(float),
                     cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_row_normsq, stream);

    std::vector<int> idx(V);
    for (int i = 0; i < V; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + std::min(5, V), idx.end(),
                       [&](int a, int b) { return h[a] > h[b]; });
    double total = 0.0;
    for (int i = 0; i < V; ++i) total += h[i];
    std::fprintf(stderr, "[embedrows] micro_step=%d top-5 vocab rows by grad norm (total_sq=%.6g):\n",
                 g_debug_micro_step, total);
    for (int k = 0; k < std::min(5, V); ++k) {
        const int i = idx[k];
        std::fprintf(stderr, "  vocab_id=%-8d norm=%.6g  (%.2f%% of total_sq)\n",
                     i, std::sqrt(static_cast<double>(h[i])),
                     total > 0.0 ? 100.0 * h[i] / total : 0.0);
    }
}

static void debug_print_centered_clip_stats(
    const char* label, const __nv_bfloat16* d_x, std::size_t n,
    const float* d_mean, const float* d_scale, cudaStream_t stream
) {
    float* d_stats = nullptr;
    cudaMallocAsync(&d_stats, 4 * sizeof(float), stream);
    cudaMemsetAsync(d_stats, 0, 4 * sizeof(float), stream);
    const std::size_t b = (n + 255) / 256;
    const unsigned blocks = static_cast<unsigned>(b < 1024 ? b : 1024);
    k_fp4_centered_clip_stats<<<blocks, 256, 0, stream>>>(d_x, n, d_mean, d_scale, d_stats);
    float mean_h = 0.0f, scale_h = 0.0f;
    float h[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    cudaMemcpyAsync(&mean_h, d_mean, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&scale_h, d_scale, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h, d_stats, sizeof(h), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_stats, stream);
    const double nn = static_cast<double>(n);
    const double fresh_mean = h[1] / nn;
    const double fresh_var = h[2] / nn - fresh_mean * fresh_mean;
    const double fresh_std = fresh_var > 0.0 ? std::sqrt(fresh_var) : 0.0;
    const double window = scale_h > 0.0 ? 7.0 / static_cast<double>(scale_h) : 0.0;
    std::fprintf(stderr,
        "[fp4clip] micro_step=%d %-8s n=%zu stale_mean=%.6g window=%.6g "
        "fresh_mean=%.6g fresh_std=%.6g fresh_maxdev=%.6g frac_clip=%.6f\n",
        g_debug_micro_step, label, n, static_cast<double>(mean_h), window,
        fresh_mean, fresh_std, static_cast<double>(h[3]),
        static_cast<double>(h[0]) / nn);
}

__global__ void k_add_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, std::size_t n
) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}
__global__ void k_inplace_add_bf16(
    __nv_bfloat16* acc, const __nv_bfloat16* src, std::size_t n
) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        acc[i] = __float2bfloat16(__bfloat162float(acc[i]) + __bfloat162float(src[i]));
}
// acc += src (FP32 — norm-scale gradient accumulation across micro-steps)
__global__ void k_acc_f32(float* acc, const float* src, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += src[i];
}
// Scale a FP32 buffer in place.
__global__ void k_scale_f32(float* x, std::size_t n, float s) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}
// Scale a BF16 buffer in place.
__global__ void k_scale_bf16(__nv_bfloat16* x, std::size_t n, float s) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2bfloat16(__bfloat162float(x[i]) * s);
}

// ── PSS Stage 2: estimated tail-layer FFN output (shadow-mode prediction
// head), IDA_NATIVE_PSS_PRED. Elementwise helpers for the low-rank
// down->relu->up predictor bolted onto the tail layer's FFN output. The GEMMs
// themselves reuse gemm_bf16/gemm_bf16_tn_f32/gemm_bf16_nt -- the same
// [in,out]-convention path down_proj already uses -- because this predictor
// operates on the full [BS,H] token stream, not a pooled per-example summary
// (that's why it doesn't reuse LRSS's small k_lrss_linear elementwise path).

__global__ void k_relu_bf16(__nv_bfloat16* x, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2bfloat16(fmaxf(__bfloat162float(x[i]), 0.0f));
}
// Post-relu activation zeroing: act[i]<=0 iff the pre-relu value was <=0
// (relu is monotone and non-negative), so this needs no separate pre-relu
// save -- same trick k_lss_relu_bwd already relies on.
__global__ void k_relu_bwd_bf16(
    const __nv_bfloat16* __restrict__ act, __nv_bfloat16* __restrict__ grad, std::size_t n
) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && __bfloat162float(act[i]) <= 0.0f) grad[i] = __float2bfloat16(0.0f);
}
// Shadow-mode aux gradient: pure MSE against the detached real FFN output --
// there is no trunk (d_hidden) seed here the way LSS's inject mode has one,
// because Stage 2 never influences the body (that's Stage 4's job).
__global__ void k_pss_pred_daux(
    const __nv_bfloat16* __restrict__ pred, const __nv_bfloat16* __restrict__ real,
    __nv_bfloat16* __restrict__ d_pred, float aux_weight, std::size_t n,
    const float* __restrict__ target_sq
) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float e = __bfloat162float(pred[i]) - __bfloat162float(real[i]);
    // IDA_NATIVE_PSS_AUX_NORMALIZE: divide by the squared TARGET norm rather
    // than element count, so the auxiliary gradient does not scale with the
    // magnitude of whatever the tail FFN happens to be emitting. A nullptr
    // target_sq (flag off) falls back to the element-count form, keeping the
    // disabled path bit-identical to the pre-feature behaviour.
    const float denom = (target_sq && isfinite(*target_sq) && *target_sq > 1.0e-12f)
        ? *target_sq : static_cast<float>(n);
    d_pred[i] = __float2bfloat16(aux_weight * 2.0f * e / denom);
}
// Shadow-mode scoring: device-accumulated squared error, squared target norm
// (for a normalized error ratio), and an epsilon-coverage count -- the raw
// material for the externally-computed confidence claim (Stage 3). Runs
// immediately after the real tail FFN is computed, so both tensors are live;
// no extra sync beyond the existing per-optimizer-step fence.
// PSS Stage 4: blend the real activation toward the predicted one by
// fraction e. dst = dst*(1-e) + pred*e. Only ever called with e > 0 (the
// governor gates the call site, not this kernel) -- e=0 would be a
// correct no-op but the caller skips the launch entirely in that case.
__global__ void k_pss_blend_bf16(
    __nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ pred,
    std::size_t n, float e
) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float d = __bfloat162float(dst[i]);
    const float p = __bfloat162float(pred[i]);
    dst[i] = __float2bfloat16(d * (1.0f - e) + p * e);
}

// PSS Stage 4 backward: routes the engaged fraction of the incoming trunk
// gradient into the predictor's own gradient, on top of its aux-loss
// contribution: dst[i] += scale * src[i]. The complementary (1-scale)
// fraction is applied separately (k_scale_bf16, in place) to the buffer the
// real down_proj backward reads.
__global__ void k_pss_add_scaled_bf16(
    __nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ src,
    std::size_t n, float scale
) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + scale * __bfloat162float(src[i]));
}

// Plain f32 elementwise dst += src. Used to fold one trunk-step's
// already-averaged PSS gradient into the cross-window accumulator (see
// StepBuffers::pss_window_accum_down/up) -- deliberately dumb, no scaling,
// so the division into a true average happens once, at apply time.
__global__ void k_accumulate_f32(
    float* __restrict__ dst, const float* __restrict__ src, std::size_t n
) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += src[i];
}

// covered is an UNSIGNED INT counter (2026-07-18 fix): it was a float
// accumulator incremented atomicAdd(+1.0f) per covered element, and float32
// integers saturate at 2^24 = 16,777,216 -- with the AI shape scoring
// n = 64*2048*2048 = 2^28 elements, any real coverage >= 1/16 clamped the
// counter at exactly 2^24, so confidence read EXACTLY 2^24/2^28 = 0.0625
// forever (production fingerprint: bit-frozen 0.0625 across every window of
// every AI burn while pred_err improved 6x). uint32 counts to 4.29e9 --
// headroom for shapes ~16x larger than today's.
// Four symbol buckets for the PSS subconscious surface. The boundaries are
// the quartiles of a zero-mean unit Gaussian, so a calibrated Gaussian target
// occupies each symbol equally. Both prediction and target use the same
// target-derived delayed RMS; the prediction can never choose its own scale.
__device__ __forceinline__ unsigned int pss_int2_gauss_symbol(
    float x, float frozen_inv_rms
) {
    constexpr float kGaussianQuartile = 0.67448975f;
    const float z = x * frozen_inv_rms;
    if (z < -kGaussianQuartile) return 0u;
    if (z < 0.0f) return 1u;
    if (z < kGaussianQuartile) return 2u;
    return 3u;
}

// Magnitude-bucketed err/hit-rate diagnostic (2026-08-15). pred_err (=
// err_sq/target_sq, energy-weighted -- dominated by the few large-|real|
// elements) and confidence (= covered/n, an UNWEIGHTED per-element hit-rate
// against a fixed eps*|real| relative tolerance, every element counted
// equally) were observed moving in OPPOSITE directions over a real 64-step
// rank256 run: pred_err improved ~33% while confidence fell ~15% the whole
// time. That's only possible if the predictor's error is NOT uniform across
// the magnitude distribution -- these buckets test that directly by
// splitting the same err_sq/target_sq/covered accounting by log2(|real|),
// so pred_err and hit-rate can be recomputed per magnitude decile instead
// of only in aggregate. Log2 range [-20, 6) covers |real| from ~1e-6 to 64
// in kPssMagBuckets equal-width bins.
constexpr int kPssMagBuckets = 8;
constexpr float kPssMagLog2Min = -20.0f;
constexpr float kPssMagLog2Width = 26.0f / static_cast<float>(kPssMagBuckets);

__device__ __forceinline__ int pss_mag_bucket(float r) {
    const float lr = log2f(fmaxf(fabsf(r), 1e-12f));
    int b = static_cast<int>((lr - kPssMagLog2Min) / kPssMagLog2Width);
    return b < 0 ? 0 : (b >= kPssMagBuckets ? kPssMagBuckets - 1 : b);
}

__global__ void k_pss_pred_score(
    const __nv_bfloat16* __restrict__ pred, const __nv_bfloat16* __restrict__ real,
    float eps, std::size_t n,
    float* __restrict__ err_sq, float* __restrict__ target_sq,
    unsigned int* __restrict__ covered,
    const float* __restrict__ int2_inv_rms,
    unsigned int* __restrict__ int2_matched,
    unsigned int* __restrict__ int2_scored,
    // Block-scaled int2 ablation instrument (2026-08-13). hidden_size/
    // block_size/num_blocks determine which of int2_inv_rms_blocks each
    // element reads its scale from, and which slot of
    // int2_target_sq_blocks it contributes to -- SEPARATE from the
    // existing target_sq (that one still feeds aux-loss normalization
    // unchanged). block_size == 0 collapses every element to block 0,
    // reproducing the single-global-scale behavior exactly via the
    // pre-existing int2_inv_rms/int2_matched/int2_scored path above,
    // which stays live and unmodified for that case.
    int hidden_size, int block_size, int num_blocks,
    float* __restrict__ int2_target_sq_blocks,
    const float* __restrict__ int2_inv_rms_blocks,
    // Magnitude-bucketed diagnostic (see pss_mag_bucket above). Nullptr-
    // guarded so this is a pure addition -- every existing call site keeps
    // working unmodified by passing nullptr for these four.
    float* __restrict__ mag_bucket_err_sq,
    float* __restrict__ mag_bucket_target_sq,
    unsigned int* __restrict__ mag_bucket_covered,
    unsigned int* __restrict__ mag_bucket_count
) {
    // Block-level reduction, one atomic per block per accumulator (2026-07-18):
    // the per-element atomicAdd version lost accuracy two ways at the AI shape
    // (n = 2^28 elements): err/target float sums soft-saturate once the running
    // sum dwarfs each increment (additions round to nothing past sum*2^-24),
    // and covered hard-saturated at 2^24 because float32 can't count past it in
    // +1.0 steps -- which pinned pss_confidence at EXACTLY 2^24/2^28 = 0.0625
    // across every window of every AI burn. covered is now a uint counter and
    // all three reduce through shared memory first: 256x fewer atomics, 256x
    // less accumulation rounding.
    __shared__ float s_err[256], s_tgt[256];
    __shared__ unsigned int s_cov[256], s_int2_match[256], s_int2_count[256];
    const int tid = threadIdx.x;
    float e2 = 0.0f, r2 = 0.0f;
    unsigned int cov = 0, int2_match = 0, int2_count = 0;
    const float frozen_inv_rms = *int2_inv_rms;
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    if (i < n) {
        const float p = __bfloat162float(pred[i]);
        const float r = __bfloat162float(real[i]);
        const float e = p - r;
        e2 = e * e;
        r2 = r * r;
        cov = (fabsf(e) < eps * fmaxf(fabsf(r), 1e-6f)) ? 1u : 0u;
        if (mag_bucket_err_sq) {
            const int mb = pss_mag_bucket(r);
            atomicAdd(&mag_bucket_err_sq[mb], e2);
            atomicAdd(&mag_bucket_target_sq[mb], r2);
            if (cov) atomicAdd(&mag_bucket_covered[mb], 1u);
            atomicAdd(&mag_bucket_count[mb], 1u);
        }
        if (block_size > 0 && hidden_size > 0) {
            int blk = static_cast<int>(i % static_cast<std::size_t>(hidden_size)) / block_size;
            if (blk >= num_blocks) blk = num_blocks - 1;  // guard: hidden_size not evenly divisible
            atomicAdd(&int2_target_sq_blocks[blk], r2);
            const float blk_inv_rms = int2_inv_rms_blocks[blk];
            if (blk_inv_rms > 0.0f && isfinite(blk_inv_rms)) {
                int2_match = pss_int2_gauss_symbol(p, blk_inv_rms) ==
                             pss_int2_gauss_symbol(r, blk_inv_rms) ? 1u : 0u;
                int2_count = 1u;
            }
        } else if (frozen_inv_rms > 0.0f && isfinite(frozen_inv_rms)) {
            // A zero scale marks the first-call bootstrap. It is deliberately
            // excluded rather than pretending that an arbitrary scale is evidence.
            int2_match = pss_int2_gauss_symbol(p, frozen_inv_rms) ==
                         pss_int2_gauss_symbol(r, frozen_inv_rms) ? 1u : 0u;
            int2_count = 1u;
        }
    }
    s_err[tid] = e2; s_tgt[tid] = r2; s_cov[tid] = cov;
    s_int2_match[tid] = int2_match; s_int2_count[tid] = int2_count;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            s_err[tid] += s_err[tid + off];
            s_tgt[tid] += s_tgt[tid + off];
            s_cov[tid] += s_cov[tid + off];
            s_int2_match[tid] += s_int2_match[tid + off];
            s_int2_count[tid] += s_int2_count[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(err_sq, s_err[0]);
        atomicAdd(target_sq, s_tgt[0]);
        atomicAdd(covered, s_cov[0]);
        atomicAdd(int2_matched, s_int2_match[0]);
        atomicAdd(int2_scored, s_int2_count[0]);
    }
}

// The score kernel above is launched once per true forward micro-step. Keep a
// min/max confidence range for the current optimizer window without another
// tensor reduction: this one-thread kernel runs after the score on the same
// stream and only reads the already-reduced covered counter.
__global__ void k_pss_record_confidence(
    const float* __restrict__ covered, std::size_t n,
    float* __restrict__ confidence_minmax
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || n == 0) return;
    const unsigned int covered_u = *reinterpret_cast<const unsigned int*>(covered);
    const float confidence = static_cast<float>(
        static_cast<double>(covered_u) / static_cast<double>(n));
    confidence_minmax[0] = fminf(confidence_minmax[0], confidence);
    confidence_minmax[1] = fmaxf(confidence_minmax[1], confidence);
}

__global__ void k_pss_reset_confidence_minmax(float* confidence_minmax) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        confidence_minmax[0] = 1.0f;
        confidence_minmax[1] = 0.0f;
    }
}

// Advances the delayed symbol scale only after the scoring kernel has
// completed. The next call therefore sees a frozen scale derived solely from
// the preceding real target, never a value advanced by its own prediction.
//
// EMA-smoothed, not a hard replace (2026-08-13 fix; was previously a bare
// overwrite from ONE micro-step's target_sq -- an effective averaging window
// of 1). Real evidence for why that's wrong: pss_int2_agreement measured
// ~24.1% across two independent 22-step runs, statistically indistinguishable
// from the 25% chance rate for 4 equally-likely Gaussian-quartile symbols --
// i.e. the diagnostic showed no detectable signal, for a WHOLE run, on both
// arms. Hypothesis (not yet independently confirmed, flagged as such): a
// 2-bit/4-symbol quantization boundary re-estimated from a single
// micro-step's mean-square is too noisy to be stable, so the symbol
// assignment jitters independently of any real prediction quality --
// matching the established coarseness-vs-window pattern elsewhere in this
// engine (fp8 delayed-scaling: history window 16; fp4: 32) extrapolated one
// step further (int2, coarser than fp4: window 64). Window is env-tunable
// specifically because 64 is an extrapolation, not a measured constant --
// rerun with different windows and check whether pss_int2_agreement moves
// meaningfully off chance before trusting this default.
inline int pss_int2_scale_window() {
    static const int v = [] {
        const char* e = std::getenv("IDA_NATIVE_PSS_INT2_SCALE_WINDOW");
        const int parsed = (e && e[0]) ? std::atoi(e) : 64;
        return parsed > 0 ? parsed : 64;
    }();
    return v;
}
__global__ void k_pss_update_int2_inv_rms(
    const float* __restrict__ target_sq, std::size_t n,
    float* __restrict__ int2_inv_rms, float ema_alpha
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || n == 0) return;
    const float mean_sq = *target_sq / static_cast<float>(n);
    if (!(mean_sq > 0.0f) || !isfinite(mean_sq)) return;  // bad sample: keep the running scale, don't zero it
    const float new_inv_rms = rsqrtf(fmaxf(mean_sq, 1e-20f));
    const float prev = *int2_inv_rms;
    *int2_inv_rms = (prev > 0.0f && isfinite(prev))
        ? (ema_alpha * new_inv_rms + (1.0f - ema_alpha) * prev)
        : new_inv_rms;  // bootstrap: first real sample seeds the EMA directly
}

// Per-block sibling of k_pss_update_int2_inv_rms -- one thread per block,
// same EMA math, reading elements_per_block = n / num_blocks worth of
// accumulated target_sq_blocks[blk] instead of one global sum. Zeroes
// target_sq_blocks after reading so the next micro-step's k_pss_pred_score
// atomicAdds start from zero (that kernel only ever adds, never resets).
__global__ void k_pss_update_int2_inv_rms_blocks(
    const float* __restrict__ target_sq_blocks, std::size_t n, int num_blocks,
    float* __restrict__ int2_inv_rms_blocks, float ema_alpha,
    float* __restrict__ target_sq_blocks_rw
) {
    const int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks || num_blocks <= 0) return;
    const float elements_per_block = static_cast<float>(n) / static_cast<float>(num_blocks);
    const float mean_sq = target_sq_blocks[blk] / fmaxf(elements_per_block, 1.0f);
    target_sq_blocks_rw[blk] = 0.0f;
    if (!(mean_sq > 0.0f) || !isfinite(mean_sq)) return;
    const float new_inv_rms = rsqrtf(fmaxf(mean_sq, 1e-20f));
    const float prev = int2_inv_rms_blocks[blk];
    int2_inv_rms_blocks[blk] = (prev > 0.0f && isfinite(prev))
        ? (ema_alpha * new_inv_rms + (1.0f - ema_alpha) * prev)
        : new_inv_rms;
}

// Reads a precomputed sum-of-squares and conditionally scales g down to an
// absolute L2-norm ceiling — fully device-side, no host readback (this runs
// per weight slot per layer per micro-step; a host sync at that call
// frequency would stall the pipeline badly).
__global__ void k_slot_grad_abs_clip_apply(
    float* __restrict__ g, std::size_t n,
    const float* __restrict__ normsq, float ceiling
) {
    const float norm = sqrtf(*normsq);
    if (norm <= ceiling) return;
    const float scale = ceiling / norm;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        g[i] *= scale;
}
__global__ void k_zeros_bf16(__nv_bfloat16* d, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) d[i] = __float2bfloat16(0.0f);
}
__global__ void k_acc_bf16(__nv_bfloat16* acc, const float* src, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) acc[i] = __float2bfloat16(__bfloat162float(acc[i]) + src[i]);
}
__global__ void k_slot_grad_abs_clip_apply_bf16(
    __nv_bfloat16* __restrict__ g, std::size_t n,
    const float* __restrict__ normsq, float ceiling
) {
    const float norm = sqrtf(*normsq);
    if (norm <= ceiling) return;
    const float scale = ceiling / norm;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        g[i] = __float2bfloat16(__bfloat162float(g[i]) * scale);
}
// Tiled byte transpose: in[rows, cols] → out[cols, rows].  Used to build the
// pre-transposed FP8 operands the cuBLASLt TN-only dW GEMMs require.
// Vectorized: each thread loads 16 consecutive bytes (uint4) so global reads
// are fully coalesced at 128B/warp; the 64×64 byte tile is transposed through
// shared memory and written back as 16-byte columns-of-the-output.
__global__ void k_transpose_u8_v16(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int rows, int cols
) {
    // 64×64 byte tile; 256 threads; each thread moves exactly 16 bytes in and
    // 16 bytes out (256 × 16 = 4096 = 64×64).
    // Row stride must stay 16B-aligned for the uint4 shared stores; 80 breaks
    // the power-of-2 bank pattern without breaking alignment (68 faults).
    __shared__ uint8_t tile[64][80];
    const int tile_r = blockIdx.y * 64;
    const int tile_c = blockIdx.x * 64;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;  // 0..255
    const int lr = tid / 4;          // tile row 0..63
    const int lc = (tid % 4) * 16;   // tile col group 0/16/32/48
    // load 16 consecutive bytes of one input row
    {
        const int r = tile_r + lr;
        const int c = tile_c + lc;
        if (r < rows) {
            if (c + 15 < cols) {
                *reinterpret_cast<uint4*>(&tile[lr][lc]) =
                    *reinterpret_cast<const uint4*>(
                        in + static_cast<std::size_t>(r) * cols + c);
            } else {
                for (int b = 0; b < 16; ++b)
                    tile[lr][lc + b] =
                        (c + b < cols) ? in[static_cast<std::size_t>(r) * cols + c + b] : 0;
            }
        }
    }
    __syncthreads();
    // write 16 consecutive bytes of one output row (gathered down a tile column)
    {
        const int orow = tile_c + lr;  // out row = in col
        const int ocol = tile_r + lc;  // out col = in row
        if (orow < cols) {
            alignas(16) uint8_t v[16];
            #pragma unroll
            for (int b = 0; b < 16; ++b) v[b] = tile[lc + b][lr];
            if (ocol + 15 < rows) {
                *reinterpret_cast<uint4*>(
                    out + static_cast<std::size_t>(orow) * rows + ocol) =
                    *reinterpret_cast<const uint4*>(v);
            } else {
                for (int b = 0; b < 16 && ocol + b < rows; ++b)
                    out[static_cast<std::size_t>(orow) * rows + ocol + b] = v[b];
            }
        }
    }
}
static void transpose_u8(
    const void* in, void* out, int rows, int cols, cudaStream_t s
) {
    dim3 block(16, 16);
    dim3 grid(ceildiv(cols, 64), ceildiv(rows, 64));
    k_transpose_u8_v16<<<grid, block, 0, s>>>(
        static_cast<const uint8_t*>(in), static_cast<uint8_t*>(out), rows, cols);
}

// ─── weight allocation & init ────────────────────────────────────────────────

struct LatticeShardSpec {
    int layer_begin{0};
    int layer_end{0};
    bool owns_embedding{true};
    bool owns_output{true};
};

static LatticeWeights allocate_lattice_weights_shard(
    const NativeRequest& req,
    NativeArena& arena,
    const LatticeShardSpec& shard
) {
    const int H = req.model.hidden_size;
    const int I = req.model.intermediate_size;
    const int global_L = req.model.layers;
    const int V = req.model.vocab_size;
    const int kv_heads = req.model.kv_heads > 0 ? req.model.kv_heads : req.model.heads;
    const int head_dim = H / req.model.heads;
    const int KV = kv_heads * head_dim;
    if (shard.layer_begin < 0 || shard.layer_end <= shard.layer_begin ||
        shard.layer_end > global_L) {
        throw std::runtime_error(
            "invalid native model-parallel layer shard [" +
            std::to_string(shard.layer_begin) + ", " +
            std::to_string(shard.layer_end) + ") for " +
            std::to_string(global_L) + " layers");
    }
    const int L = shard.layer_end - shard.layer_begin;
    const bool full_body_shard = shard.layer_begin == 0 &&
        shard.layer_end == global_L && shard.owns_embedding && shard.owns_output;
    if (!full_body_shard && (!req.parent.resume_from_checkpoint.empty() ||
                             !req.pss_pred_state_init_path.empty())) {
        throw std::runtime_error(
            "native model-parallel shards support parent weights but not exact "
            "resume or standalone PSS state loading until stage optimizer "
            "persistence is implemented");
    }
    const int moe_num_routes = req.model.num_cognitive_routes;
    const bool moe_shared_trunk = req.model.use_personality_residual_experts;
    const int generic_moe_num_experts = req.model.generic_moe_num_experts;
    if (generic_moe_num_experts > 0 && moe_num_routes > 0) {
        throw std::runtime_error(
            "generic_moe_num_experts and num_cognitive_routes are mutually exclusive "
            "MoE modes; set only one");
    }
    if (generic_moe_num_experts > 0 &&
        (req.model.generic_moe_top_k <= 0 || req.model.generic_moe_top_k > generic_moe_num_experts)) {
        throw std::runtime_error(
            "generic_moe_top_k must be in (0, generic_moe_num_experts]");
    }
    if (req.model.generic_moe_shared_expert_width > 0 && generic_moe_num_experts <= 0) {
        throw std::runtime_error(
            "generic_moe_shared_expert_width requires generic_moe_num_experts > 0");
    }
    const int moe_num_experts = moe_num_routes > 0
        ? (moe_shared_trunk
               ? (req.model.num_personality_experts > 0
                      ? req.model.num_personality_experts
                      : moe_num_routes)
               : moe_num_routes)
        : 0;
    const int moe_expert_width = moe_num_routes > 0
        ? (moe_shared_trunk
               ? (req.model.personality_residual_expert_width > 0
                      ? req.model.personality_residual_expert_width
                      : I)
               : I)
        : 0;
    const int moe_top_k = moe_num_routes > 0
        ? (moe_shared_trunk
               ? (req.model.top_k_experts > 0 ? req.model.top_k_experts : moe_num_experts)
               : (req.model.top_k_routes > 0 ? req.model.top_k_routes : moe_num_experts))
        : 0;

    const std::string model_contract = req.model.architecture_contract.empty()
        ? req.architecture_contract : req.model.architecture_contract;
    const bool gpt2 = model_contract == "hf_gpt2_native_v1";
    LatticeWeights w{};
    w.hidden_size       = H;
    w.intermediate_size = I;
    w.num_layers        = L;
    w.layer_offset      = shard.layer_begin;
    w.global_num_layers = global_L;
    w.vocab_size        = V;
    w.max_position_embeddings = req.model.max_position_embeddings;
    w.heads             = req.model.heads;
    w.kv_heads          = kv_heads;
    w.architecture_contract = model_contract;
    w.owns_embedding    = shard.owns_embedding;
    w.owns_output       = shard.owns_output;

    std::size_t bytes = 0;
    auto trk = [&](std::size_t n) { bytes += n * sizeof(__nv_bfloat16); };
    if (gpt2 && w.max_position_embeddings <= 0)
        throw std::runtime_error("GPT-2 contract requires max_position_embeddings > 0");

    if (w.owns_embedding) {
        w.embed = alloc_bf16(V * H, arena); trk(V * H);
        if (gpt2) {
            w.position_embeddings = alloc_bf16(static_cast<std::size_t>(w.max_position_embeddings) * H, arena);
            trk(static_cast<std::size_t>(w.max_position_embeddings) * H);
        }
    }
    if (w.owns_output) {
        w.final_norm = alloc_bf16(H, arena); trk(H);
        if (gpt2) { w.final_norm_bias = alloc_bf16(H, arena); trk(H); }
        w.lm_head = alloc_bf16(V * H, arena); trk(V * H);
    }

    w.layers = new LatticeLayerWeights[L];
    for (int l = 0; l < L; ++l) {
        w.layers[l].attn_norm = alloc_bf16(H,     arena); trk(H);
        if (gpt2) { w.layers[l].attn_norm_bias = alloc_bf16(H, arena); trk(H); }
        w.layers[l].q_proj    = alloc_bf16(H * H, arena); trk(H * H);
        w.layers[l].k_proj    = alloc_bf16(H * KV, arena); trk(H * KV);
        w.layers[l].v_proj    = alloc_bf16(H * KV, arena); trk(H * KV);
        w.layers[l].o_proj    = alloc_bf16(H * H, arena); trk(H * H);
        w.layers[l].ffn_norm  = alloc_bf16(H,     arena); trk(H);
        if (gpt2) { w.layers[l].ffn_norm_bias = alloc_bf16(H, arena); trk(H); }
        w.layers[l].gate_proj = alloc_bf16(H * I, arena); trk(H * I);
        w.layers[l].up_proj   = gpt2 ? nullptr : alloc_bf16(H * I, arena);
        if (!gpt2) trk(H * I);
        w.layers[l].down_proj = alloc_bf16(I * H, arena); trk(I * H);
        if (gpt2 || req.model.qkv_bias) {
            w.layers[l].q_bias = alloc_bf16(H,  arena); trk(H);
            w.layers[l].k_bias = alloc_bf16(KV, arena); trk(KV);
            w.layers[l].v_bias = alloc_bf16(KV, arena); trk(KV);
            if (gpt2) {
                w.layers[l].o_bias = alloc_bf16(H, arena); trk(H);
                w.layers[l].ffn_in_bias = alloc_bf16(I, arena); trk(I);
                w.layers[l].ffn_out_bias = alloc_bf16(H, arena); trk(H);
            }
        }
    }

    // Cognitive-architecture sparse MoE port (2026-07-21). Master gate:
    // num_cognitive_routes>0 -- present in both Mode A (independent
    // experts) and Mode B (shared trunk + personality residual experts).
    // 0 = off, dense gate/up/down above trains unchanged, byte-identical
    // checkpoint to the pre-port shape (same convention as lrss_enabled).
    if (moe_num_experts > 0) {
        for (int l = 0; l < L; ++l) {
            auto& lw = w.layers[l];
            lw.num_experts = moe_num_experts;
            lw.expert_balancing_loss_coef =
                req.model.expert_balancing_loss_coef;
            lw.expert_intermediate_size = moe_expert_width;
            lw.moe_shared_trunk = moe_shared_trunk;
            lw.num_routes = moe_num_routes;
            lw.top_k = moe_top_k;

            if (moe_shared_trunk) {
                lw.trunk_fc_in_w  = alloc_bf16(static_cast<std::size_t>(I) * H, arena);
                trk(static_cast<std::size_t>(I) * H);
                lw.trunk_fc_out_w = alloc_bf16(static_cast<std::size_t>(H) * I, arena);
                trk(static_cast<std::size_t>(H) * I);
            }
            const std::size_t expert_in_n =
                static_cast<std::size_t>(moe_num_experts) * moe_expert_width * H;
            const std::size_t expert_out_n =
                static_cast<std::size_t>(moe_num_experts) * H * moe_expert_width;
            lw.expert_fc_in_w  = alloc_bf16(expert_in_n, arena);  trk(expert_in_n);
            lw.expert_fc_out_w = alloc_bf16(expert_out_n, arena); trk(expert_out_n);

            lw.pressure_proj_w = alloc_bf16(static_cast<std::size_t>(moe_num_routes) * H, arena);
            trk(static_cast<std::size_t>(moe_num_routes) * H);
            lw.pressure_mod_w  = alloc_bf16(static_cast<std::size_t>(H) * moe_num_routes, arena);
            trk(static_cast<std::size_t>(H) * moe_num_routes);
            lw.router_score_w  = alloc_bf16(static_cast<std::size_t>(moe_num_experts) * H, arena);
            trk(static_cast<std::size_t>(moe_num_experts) * H);
            // nn.Identity() in the reference whenever num_routes==num_experts.
            if (moe_num_routes != moe_num_experts) {
                lw.pressure_to_routes_w = alloc_bf16(
                    static_cast<std::size_t>(moe_num_experts) * moe_num_routes, arena);
                trk(static_cast<std::size_t>(moe_num_experts) * moe_num_routes);
            }
        }
    }

    // Generic per-token top-k SwiGLU MoE (2026-08-23). Independent gate from
    // the block above -- mutual exclusivity already enforced where
    // generic_moe_num_experts is read. Reuses lw.num_experts/top_k/
    // router_score_w (routing math is identical apart from the explicit
    // norm_topk_prob policy); only the expert weights and dispatch differ.
    const int generic_moe_top_k = req.model.generic_moe_top_k;
    const int generic_moe_expert_width = generic_moe_num_experts > 0
        ? (req.model.generic_moe_expert_width > 0 ? req.model.generic_moe_expert_width : I)
        : 0;
    if (generic_moe_num_experts > 0) {
        for (int l = 0; l < L; ++l) {
            auto& lw = w.layers[l];
            lw.moe_kind = 1;
            lw.generic_moe_normalize_topk = req.model.generic_moe_normalize_topk;
            lw.num_experts = generic_moe_num_experts;
            lw.expert_intermediate_size = generic_moe_expert_width;
            lw.top_k = generic_moe_top_k;

            const std::size_t expert_gu_n =
                static_cast<std::size_t>(generic_moe_num_experts) * generic_moe_expert_width * H;
            const std::size_t expert_down_n =
                static_cast<std::size_t>(generic_moe_num_experts) * H * generic_moe_expert_width;
            lw.generic_gate_proj_w = alloc_bf16(expert_gu_n, arena);   trk(expert_gu_n);
            lw.generic_up_proj_w   = alloc_bf16(expert_gu_n, arena);   trk(expert_gu_n);
            lw.generic_down_proj_w = alloc_bf16(expert_down_n, arena); trk(expert_down_n);
            lw.router_score_w = alloc_bf16(
                static_cast<std::size_t>(generic_moe_num_experts) * H, arena);
            trk(static_cast<std::size_t>(generic_moe_num_experts) * H);

            lw.shared_expert_width = req.model.generic_moe_shared_expert_width;
            if (lw.shared_expert_width > 0) {
                const std::size_t Ws = static_cast<std::size_t>(lw.shared_expert_width);
                lw.generic_shared_gate_proj_w = alloc_bf16(Ws * H, arena); trk(Ws * H);
                lw.generic_shared_up_proj_w   = alloc_bf16(Ws * H, arena); trk(Ws * H);
                lw.generic_shared_down_proj_w = alloc_bf16(static_cast<std::size_t>(H) * Ws, arena);
                trk(static_cast<std::size_t>(H) * Ws);
                lw.generic_shared_gate_score_w = alloc_bf16(static_cast<std::size_t>(H), arena);
                trk(static_cast<std::size_t>(H));
            }
        }
    }

    // LRSS multiscale memory bank weights. Default-on for the native body;
    // IDA_NATIVE_LRSS=0 retains the governed ablation path.
    if (w.owns_output) {
        const char* e = std::getenv("IDA_NATIVE_LRSS");
        w.lrss_enabled = !(e && e[0] == '0');
        if (w.lrss_enabled) {
            const int J = w.lrss_scales;
            w.lrss_query   = alloc_bf16(static_cast<std::size_t>(H) * H, arena); trk(static_cast<std::size_t>(H) * H);
            w.lrss_key     = alloc_bf16(static_cast<std::size_t>(H) * H, arena); trk(static_cast<std::size_t>(H) * H);
            w.lrss_gate_w  = alloc_bf16(static_cast<std::size_t>(H) * 2 * H, arena); trk(static_cast<std::size_t>(H) * 2 * H);
            w.lrss_gate_b  = alloc_bf16(H, arena); trk(H);
            w.lrss_log_tau = alloc_bf16(J, arena); trk(J);
            w.lrss_scale_w = alloc_bf16(J, arena); trk(J);
            // LSS supersampler head: rank-128 reconstruct from the int2
            // LSS is also default-on; IDA_NATIVE_LSS=0 disables only the
            // supersampler while preserving the LRSS bank.
            const char* lss = std::getenv("IDA_NATIVE_LSS");
            if (!(lss && lss[0] == '0')) {
                const char* rk = std::getenv("IDA_NATIVE_LSS_RANK");
                w.lss_rank = rk ? std::max(1, std::atoi(rk)) : 128;
                // PSS Stage 1b: default off (lab flag). When on, the LSS
                // joint input widens from [pooled;anchor] to
                // [pooled;anchor;spike] -- this is the one field every
                // lss_down/lss_joint size computation in this file reads.
                const char* sj = std::getenv("IDA_NATIVE_PSS_SPIKE_JOINT");
                w.pss_spike_joint_dim = (sj && sj[0] == '1') ? kPssSpikeBuckets : 0;
                const std::size_t joint_dim = static_cast<std::size_t>(2 * H) +
                    static_cast<std::size_t>(w.pss_spike_joint_dim);
                w.lss_down = alloc_bf16(static_cast<std::size_t>(w.lss_rank) * joint_dim, arena);
                trk(static_cast<std::size_t>(w.lss_rank) * joint_dim);
                w.lss_up = alloc_bf16(static_cast<std::size_t>(H) * w.lss_rank, arena);
                trk(static_cast<std::size_t>(H) * w.lss_rank);
            }
        }
    }

    // PSS Stage 2 (default off): shadow-mode low-rank predictor of the tail
    // layer's FFN output. Architecturally independent of the LRSS bank -- it
    // hooks the trunk's own tail-layer FFN, not the multiscale memory -- so
    // it is not nested under w.lrss_enabled.
    //
    // Per-burn request field wins over env (2026-07-15): the persistent
    // socket worker's getenv() sees the WORKER process environment, not the
    // launching wrapper's -- env-based activation silently never reached
    // worker-served burns. req.pss_pred_rank: -1 = unset (env fallback),
    // 0 = off, >0 = rank.
    if (w.owns_output) {
        int rank = 0;
        if (req.pss_pred_rank >= 0) {
            rank = req.pss_pred_rank;
        } else {
            const char* e = std::getenv("IDA_NATIVE_PSS_PRED");
            if (e && e[0] == '1') {
                const char* rk = std::getenv("IDA_NATIVE_PSS_PRED_RANK");
                // rk[0] (non-empty), not just rk (non-null): an env-var passed
                // through as "" (e.g. a launcher's ${VAR:-} default when unset)
                // is a non-null pointer to an empty string -- atoi("") == 0,
                // which would silently force rank=1 instead of this default.
                rank = (rk && rk[0]) ? std::max(1, std::atoi(rk)) : 64;
            }
        }
        if (rank > 0) {
            w.pss_pred_rank = rank;
            const auto R = static_cast<std::size_t>(w.pss_pred_rank);
            w.pss_pred_down = alloc_bf16(static_cast<std::size_t>(H) * R, arena);
            trk(static_cast<std::size_t>(H) * R);
            w.pss_pred_up = alloc_bf16(R * static_cast<std::size_t>(H), arena);
            trk(R * static_cast<std::size_t>(H));
        }
    }
    w.total_bytes = bytes;

    // ── Per-student initialisation ───────────────────────────────────────────
    // Every student gets its own weight universe: the seed mixes the seat and
    // version (FNV-1a) into the config seed, so no two students share genesis
    // weights.  On top of that, each student draws per-layer Q/K init gains in
    // [0.85, 1.20] from its own hash — distinct attention sharpness profiles
    // at genesis to help personality separation instead of a flattened start.
    uint64_t seat_hash = 1469598103934665603ULL;   // FNV-1a
    for (const char c : req.seat + ":" + req.version) {
        seat_hash ^= static_cast<uint64_t>(static_cast<unsigned char>(c));
        seat_hash *= 1099511628211ULL;
    }
    uint64_t off = (static_cast<uint64_t>(req.seed) * 1000003ULL) ^ seat_hash;

    auto next_hash = [&](uint64_t salt) {
        uint64_t x = seat_hash ^ (salt * 0x9e3779b97f4a7c15ULL);
        x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    };
    auto personality_gain = [&](uint64_t salt) {
        // uniform in [0.85, 1.20]
        const double u = static_cast<double>(next_hash(salt) >> 11) /
                         static_cast<double>(1ULL << 53);
        return static_cast<float>(0.85 + 0.35 * u);
    };

    auto init_u = [&](auto* p, std::size_t n, float fan_in, float gain) {
        float lim = gain * sqrtf(6.0f / fan_in);
        if (p) {
            k_uniform_bf16<<<ceildiv(n, 256), 256, 0, arena.stream>>>(p, n, lim, off);
        }
        off += n;
    };
    // Residual-scaled init (GPT-2/Megatron recipe): the two projections that
    // WRITE into the residual stream (o_proj, down_proj) get an extra
    // 1/sqrt(2L) on their init gain. Without it each layer's backward factor
    // (I + J_block) has norm > 1 at genesis and the gradient magnitude
    // compounds exponentially through depth — measured 2026-07-09 at
    // 10-36x/layer (dS-cancellation probe), 16-layer GN ~3.3x the 4-layer
    // GN at identical mb/ga, L0 always the dominant blown-up slot because
    // the backward sweep touches it last. The clips (slot/row/interlayer)
    // contain that compounding; this removes its source at genesis.
    // Parent-loaded bodies are unaffected (init is overwritten by the
    // parent weights below). Env-gated for A/B: 0/unset = legacy unscaled.
    const bool residual_scaled_init = [] {
        const char* e = std::getenv("IDA_NATIVE_RESIDUAL_SCALED_INIT");
        return e && e[0] == '1';
    }();
    const float resid_write_gain =
        residual_scaled_init ? (1.0f / sqrtf(2.0f * static_cast<float>(global_L))) : 1.0f;
    if (residual_scaled_init) {
        std::fprintf(stderr,
            "[ida_native_train] residual-scaled init: o_proj/down_proj gain = %.4f (L=%d)\n",
            resid_write_gain, global_L);
    }
    // init_u's formula (lim = gain*sqrt(6/fan_in), uniform) gives
    // Var(output) = 2*gain^2*Var(input) -- gain=1.0 is only variance-
    // preserving when the layer is immediately followed by ReLU (ReLU
    // halves variance back down, exactly cancelling the doubling -- this
    // is the actual He/Kaiming design intent for this formula shape,
    // fan_in-only, uniform). For a layer NOT followed by ReLU, gain=1.0
    // silently doubles variance at every application; the variance-
    // preserving choice there is gain=1/sqrt(2). Found 2026-07-30: three
    // call sites use gain=1.0 despite not being followed by ReLU --
    // pressure_proj_w (-> tanh), pressure_mod_w (-> sigmoid), and
    // pss_pred_up (-> direct output, no activation) -- and all three show
    // the same signature (already-saturated/over-scaled output present at
    // micro_step=1, before any real training). Env-gated for A/B, same
    // discipline as residual_scaled_init above: 0/unset = legacy gain=1.0.
    const bool nonrelu_gain_fix = [] {
        const char* e = std::getenv("IDA_NATIVE_NONRELU_GAIN_FIX");
        return e && e[0] == '1';
    }();
    const float nonrelu_gain = nonrelu_gain_fix ? (1.0f / sqrtf(2.0f)) : 1.0f;
    if (nonrelu_gain_fix) {
        std::fprintf(stderr,
            "[ida_native_train] non-ReLU gain fix: pressure_proj/pressure_mod/pss_pred_up gain = %.4f\n",
            nonrelu_gain);
    }
    auto init_embed = [&](auto* p, std::size_t n) {
        if (p) {
            k_uniform_bf16<<<ceildiv(n, 256), 256, 0, arena.stream>>>(p, n, 0.02f, off);
        }
        off += n;
    };
    auto init_ones = [&](auto* p, std::size_t n) {
        if (p) {
            k_ones_bf16<<<ceildiv(n, 256), 256, 0, arena.stream>>>(p, n);
        }
    };
    // Qwen2-family QKV bias: zero-init, standard nn.Linear bias convention --
    // any real sideloaded checkpoint overwrites this with real values anyway.
    auto init_zero_bf16 = [&](__nv_bfloat16* p, std::size_t n) {
        if (p) {
            IDA_CUDA_CHECK(cudaMemsetAsync(p, 0, n * sizeof(__nv_bfloat16), arena.stream));
        }
    };

    init_embed(w.embed,   static_cast<std::size_t>(V) * H);
    init_embed(w.position_embeddings,
               static_cast<std::size_t>(w.max_position_embeddings) * H);
    init_embed(w.lm_head, static_cast<std::size_t>(V) * H);
    init_ones(w.final_norm, H);

    // Iterate the logical body, not only this shard, so every stage consumes
    // the same deterministic seed offsets as the unsplit model. A null target
    // advances the offset without allocating or launching a kernel.
    for (int global_l = 0; global_l < global_L; ++global_l) {
        LatticeLayerWeights* lw = nullptr;
        if (global_l >= w.layer_offset && global_l < w.layer_offset + L) {
            lw = &w.layers[global_l - w.layer_offset];
        }
        const float qk_gain = personality_gain(static_cast<uint64_t>(global_l) * 2 + 1);
        init_ones(lw ? lw->attn_norm : nullptr, H);
        init_ones(lw ? lw->ffn_norm : nullptr,  H);
        init_u(lw ? lw->q_proj : nullptr,    H * H, float(H), qk_gain);
        init_u(lw ? lw->k_proj : nullptr,    H * KV, float(H), qk_gain);
        init_u(lw ? lw->v_proj : nullptr,    H * KV, float(H), 1.0f);
        init_u(lw ? lw->o_proj : nullptr,    H * H, float(H), resid_write_gain);
        init_zero_bf16(lw ? lw->q_bias : nullptr, static_cast<std::size_t>(H));
        init_zero_bf16(lw ? lw->k_bias : nullptr, static_cast<std::size_t>(KV));
        init_zero_bf16(lw ? lw->v_bias : nullptr, static_cast<std::size_t>(KV));
        init_zero_bf16(lw ? lw->o_bias : nullptr, static_cast<std::size_t>(H));
        init_zero_bf16(lw ? lw->attn_norm_bias : nullptr, static_cast<std::size_t>(H));
        init_zero_bf16(lw ? lw->ffn_norm_bias : nullptr, static_cast<std::size_t>(H));
        init_zero_bf16(lw ? lw->ffn_in_bias : nullptr, static_cast<std::size_t>(I));
        init_zero_bf16(lw ? lw->ffn_out_bias : nullptr, static_cast<std::size_t>(H));
        init_u(lw ? lw->gate_proj : nullptr, H * I, float(H), 1.0f);
        init_u(lw ? lw->up_proj : nullptr,   H * I, float(H), 1.0f);
        init_u(lw ? lw->down_proj : nullptr, I * H, float(I), resid_write_gain);

        // Cognitive-architecture sparse MoE port: fc_out/trunk_fc_out write
        // into the residual stream (mixed becomes the FFN-residual delta,
        // same role as down_proj) so they get the same resid_write_gain
        // treatment; fc_in/trunk_fc_in/routing projections don't, matching
        // gate_proj/up_proj's plain gain=1.0.
        if (moe_num_experts > 0) {
            const std::size_t I_e = static_cast<std::size_t>(moe_expert_width);
            if (moe_shared_trunk) {
                init_u(lw ? lw->trunk_fc_in_w : nullptr,
                       static_cast<std::size_t>(I) * H, float(H), 1.0f);
                init_u(lw ? lw->trunk_fc_out_w : nullptr,
                       static_cast<std::size_t>(H) * I, float(I), resid_write_gain);
            }
            init_u(lw ? lw->expert_fc_in_w : nullptr,
                   static_cast<std::size_t>(moe_num_experts) * I_e * H, float(H), 1.0f);
            init_u(lw ? lw->expert_fc_out_w : nullptr,
                   static_cast<std::size_t>(moe_num_experts) * H * I_e, float(I_e), resid_write_gain);
            init_u(lw ? lw->pressure_proj_w : nullptr,
                   static_cast<std::size_t>(moe_num_routes) * H, float(H), nonrelu_gain);
            init_u(lw ? lw->pressure_mod_w : nullptr,
                   static_cast<std::size_t>(H) * moe_num_routes, float(moe_num_routes), nonrelu_gain);
            init_u(lw ? lw->router_score_w : nullptr,
                   static_cast<std::size_t>(moe_num_experts) * H, float(H), 1.0f);
            if (moe_num_routes != moe_num_experts) {
                init_u(lw ? lw->pressure_to_routes_w : nullptr,
                       static_cast<std::size_t>(moe_num_experts) * moe_num_routes,
                       float(moe_num_routes), 1.0f);
            }
        }

        // Generic per-token top-k SwiGLU MoE. Same gain convention as the
        // dense FFN's own gate/up/down: gate/up plain gain=1.0 (matches
        // gate_proj/up_proj above), down_proj writes into the residual
        // stream so it gets resid_write_gain (matches down_proj above).
        if (generic_moe_num_experts > 0) {
            const std::size_t I_e = static_cast<std::size_t>(generic_moe_expert_width);
            init_u(lw ? lw->generic_gate_proj_w : nullptr,
                   static_cast<std::size_t>(generic_moe_num_experts) * I_e * H, float(H), 1.0f);
            init_u(lw ? lw->generic_up_proj_w : nullptr,
                   static_cast<std::size_t>(generic_moe_num_experts) * I_e * H, float(H), 1.0f);
            init_u(lw ? lw->generic_down_proj_w : nullptr,
                   static_cast<std::size_t>(generic_moe_num_experts) * H * I_e, float(I_e), resid_write_gain);
            init_u(lw ? lw->router_score_w : nullptr,
                   static_cast<std::size_t>(generic_moe_num_experts) * H, float(H), 1.0f);

            // Shared expert: same gain convention as the routed experts above
            // (gate/up plain 1.0, down resid_write_gain); the gate-score
            // projection is a plain [1,H] linear feeding a sigmoid, gain 1.0.
            const int shared_w = lw ? lw->shared_expert_width : 0;
            if (shared_w > 0) {
                const std::size_t Ws = static_cast<std::size_t>(shared_w);
                init_u(lw ? lw->generic_shared_gate_proj_w : nullptr, Ws * H, float(H), 1.0f);
                init_u(lw ? lw->generic_shared_up_proj_w : nullptr, Ws * H, float(H), 1.0f);
                init_u(lw ? lw->generic_shared_down_proj_w : nullptr,
                       static_cast<std::size_t>(H) * Ws, float(Ws), resid_write_gain);
                init_u(lw ? lw->generic_shared_gate_score_w : nullptr,
                       static_cast<std::size_t>(H), float(H), 1.0f);
            }
        }
    }

    if (w.lrss_enabled) {
        const int J = w.lrss_scales;
        init_u(w.lrss_query,  static_cast<std::size_t>(H) * H,     float(H),     1.0f);
        init_u(w.lrss_key,    static_cast<std::size_t>(H) * H,     float(H),     1.0f);
        init_u(w.lrss_gate_w, static_cast<std::size_t>(H) * 2 * H, float(2 * H), 1.0f);
        if (w.lss_rank > 0) {
            const std::size_t joint_dim = static_cast<std::size_t>(2 * H) +
                static_cast<std::size_t>(w.pss_spike_joint_dim);
            init_u(w.lss_down, static_cast<std::size_t>(w.lss_rank) * joint_dim, float(joint_dim), 1.0f);
            init_u(w.lss_up, static_cast<std::size_t>(H) * w.lss_rank, float(w.lss_rank), 1.0f);
        }
        IDA_CUDA_CHECK(cudaMemsetAsync(w.lrss_gate_b, 0, H * sizeof(__nv_bfloat16), arena.stream));
        IDA_CUDA_CHECK(cudaMemsetAsync(w.lrss_scale_w, 0, J * sizeof(__nv_bfloat16), arena.stream));
        // log_tau: linspace(ln tau_min, ln tau_max) — matches the Python init.
        std::vector<__nv_bfloat16> lt(J);
        const float lo = std::log(1.0f), hi = std::log(64.0f);
        for (int j = 0; j < J; ++j)
            lt[j] = __float2bfloat16(lo + (hi - lo) * (J > 1 ? float(j) / (J - 1) : 0.0f));
        IDA_CUDA_CHECK(cudaMemcpyAsync(w.lrss_log_tau, lt.data(),
            J * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, arena.stream));
        IDA_CUDA_CHECK(cudaStreamSynchronize(arena.stream));  // lt is stack-local
    }
    if (w.pss_pred_rank > 0) {
        const std::size_t R = w.pss_pred_rank;
        init_u(w.pss_pred_down, static_cast<std::size_t>(H) * R, float(H), 1.0f);
        init_u(w.pss_pred_up,   R * static_cast<std::size_t>(H), float(R), pss_pred_up_init_gain());
    }
    IDA_CUDA_CHECK(cudaStreamSynchronize(arena.stream));

    // ── Parent lineage ────────────────────────────────────────────────────────
    // Band/version children start from the parent body, not genesis noise.
    // A named parent that cannot be loaded is a hard failure: silently training
    // from random init would produce a lineage-invalid artifact (the exact
    // wedge the promotion gates exist to catch).
    if (!req.parent.init_from_model.empty()) {
        std::string err;
        if (!load_lattice_weights_safetensors(
                req.parent.init_from_model, w, arena.stream, err)) {
            throw std::runtime_error(
                "init_from_model load failed (" +
                req.parent.init_from_model.string() + "): " + err);
        }
        std::fprintf(stderr,
            "[ida_native_train] parent weights loaded: %s\n",
            req.parent.init_from_model.string().c_str());
    }

    // ── PSS predictor standalone state (per-family×seat weights repo) ─────────
    // Runs AFTER the parent load so the dedicated state file — the fresher,
    // cross-burn source — wins over whatever head the parent checkpoint
    // carried. Request field wins over env (same worker-caching rationale as
    // pss_pred_rank: a shared persistent worker's getenv() is cached at
    // spawn, so a per-seat path can only survive the handoff via the
    // per-burn request). A RESET/absent state is normal operation, never a
    // failure.
    if (w.pss_pred_rank > 0) {
        // A partial peer shard cannot restore a standalone PSS state without
        // also restoring the matching stage-owned optimizer and trunk state.
        // Do not let an inherited process env silently contaminate a genesis
        // model-parallel canary; full single-device bodies retain the legacy
        // request-then-env lookup.
        const std::string init_path_str = !req.pss_pred_state_init_path.empty()
            ? req.pss_pred_state_init_path
            : full_body_shard
                ? [] {
                      const char* e = std::getenv("IDA_NATIVE_PSS_PRED_INIT_PATH");
                      return std::string(e ? e : "");
                  }()
                : std::string{};
        const char* sp = init_path_str.c_str();
        if (sp && sp[0]) {
            std::string state_detail;
            const auto verdict = load_pss_pred_state_safetensors(
                req, w, sp, arena.stream, state_detail);
            const char* tag =
                verdict == PssPredStateLoad::kResumed ? "resumed"
                : verdict == PssPredStateLoad::kReset ? "RESET"
                                                      : "fresh";
            std::fprintf(stderr,
                "[ida_native_train] pss-pred state %s: %s\n",
                tag, state_detail.c_str());
        }
    }
    return w;
}

LatticeWeights allocate_lattice_weights(const NativeRequest& req, NativeArena& arena) {
    return allocate_lattice_weights_shard(
        req, arena, LatticeShardSpec{0, req.model.layers, true, true});
}

void free_lattice_weights(LatticeWeights& w, NativeArena& arena) {
    auto f = [&](auto* p) { if (p) IDA_CUDA_CHECK(cudaFreeAsync(p, arena.stream)); };
    f(w.embed); f(w.position_embeddings); f(w.final_norm); f(w.final_norm_bias); f(w.lm_head);
    f(w.lrss_query); f(w.lrss_key); f(w.lrss_gate_w);
    f(w.lrss_gate_b); f(w.lrss_log_tau); f(w.lrss_scale_w);
    f(w.lss_down); f(w.lss_up);
    f(w.pss_pred_down); f(w.pss_pred_up);
    for (int l = 0; l < w.num_layers; ++l) {
        f(w.layers[l].attn_norm); f(w.layers[l].q_proj); f(w.layers[l].k_proj);
        f(w.layers[l].v_proj);    f(w.layers[l].o_proj);  f(w.layers[l].ffn_norm);
        f(w.layers[l].q_bias); f(w.layers[l].k_bias); f(w.layers[l].v_bias);
        f(w.layers[l].o_bias); f(w.layers[l].attn_norm_bias); f(w.layers[l].ffn_norm_bias);
        f(w.layers[l].gate_proj); f(w.layers[l].up_proj); f(w.layers[l].down_proj);
        f(w.layers[l].ffn_in_bias); f(w.layers[l].ffn_out_bias);
        f(w.layers[l].trunk_fc_in_w);  f(w.layers[l].trunk_fc_out_w);
        f(w.layers[l].expert_fc_in_w); f(w.layers[l].expert_fc_out_w);
        f(w.layers[l].pressure_proj_w); f(w.layers[l].pressure_mod_w);
        f(w.layers[l].router_score_w);  f(w.layers[l].pressure_to_routes_w);
        f(w.layers[l].generic_gate_proj_w); f(w.layers[l].generic_up_proj_w);
        f(w.layers[l].generic_down_proj_w);
        f(w.layers[l].generic_shared_gate_proj_w); f(w.layers[l].generic_shared_up_proj_w);
        f(w.layers[l].generic_shared_down_proj_w); f(w.layers[l].generic_shared_gate_score_w);
    }
    delete[] w.layers; w.layers = nullptr;
    IDA_CUDA_CHECK(cudaStreamSynchronize(arena.stream));
}

// ─── optimizer state ─────────────────────────────────────────────────────────

static bool optimizer_state_uses_bf16(const NativeRequest& request) {
    return request.optimizer_state_precision == "bf16";
}

// IDA optimizer plan Part B, Stage 2 (2026-07-15, never GPU-validated in
// production -- probe only). Stage 2a scope: validate Lion's update rule
// using the EXISTING m/v allocation (v is allocated but simply unused by
// Lion) rather than also refactoring every alloc/free/accumulate site to
// skip v -- that memory-halving follow-up is deferred to Stage 2b once the
// update rule itself is proven on real data.
static bool optimizer_uses_lion(const NativeRequest& request) {
    return request.optimizer_type == "lion";
}

// Lion needs a materially different LR/WD than AdamW (paper guidance:
// ~3-10x lower LR, ~3-10x higher WD) -- probed, not assumed.
//
// Phase 2 (2026-07-22): request.lion_*_override wins over the cached env
// value, same sentinel pattern as global_clip_override/pss_pred_rank --
// the persistent worker caches getenv() once at spawn, so a raw env var
// alone never reaches a burn served by an already-running worker whose
// environment was set for a DIFFERENT prior burn (launch_family_queue.sh's
// own comment, ~1049-1058, already documented this exact gap for Lion
// specifically). The env fallback stays cached (still process-lifetime-
// appropriate for the direct-launch/no-override case); only the override
// check itself must read the live per-call request.
static float lion_lr_scale(const NativeRequest& request) {
    if (request.lion_lr_scale_override >= 0.0f) return request.lion_lr_scale_override;
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_LION_LR_SCALE");
        return (e && e[0]) ? std::max(0.0f, static_cast<float>(std::atof(e))) : 0.1f;
    }();
    return v;
}
static float lion_wd_scale(const NativeRequest& request) {
    if (request.lion_wd_scale_override >= 0.0f) return request.lion_wd_scale_override;
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_LION_WD_SCALE");
        return (e && e[0]) ? std::max(0.0f, static_cast<float>(std::atof(e))) : 3.0f;
    }();
    return v;
}
static float lion_beta1(const NativeRequest& request) {
    if (request.lion_beta1_override >= 0.0f) return request.lion_beta1_override;
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_LION_BETA1");
        return (e && e[0]) ? static_cast<float>(std::atof(e)) : 0.9f;
    }();
    return v;
}
static float lion_beta2(const NativeRequest& request) {
    if (request.lion_beta2_override >= 0.0f) return request.lion_beta2_override;
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_LION_BETA2");
        return (e && e[0]) ? static_cast<float>(std::atof(e)) : 0.99f;
    }();
    return v;
}

// Optimizer Stage 3 v2 (2026-07-16 redesign), scoped to Lion only:
// trust_i = clamp(ema_i / (current_i + eps), lo, hi), reusing PSS Stage
// 1a's per-slot gradient-norm EMA (pss_slot_norm_ema) and the ratio it
// already computes -- free, no new kernel pass, no amortization. v1
// (trust_i = clamp(||w_i||/sqrt(n_i), lo, hi), see
// probe_ledger.jsonl optimizer_stage3_trust_ratio_CORRECTION) compared
// each slot against an architectural constant (~1/sqrt(H)) that was
// nearly identical across every slot regardless of whether THAT slot was
// currently spiking -- it could only act as a uniform global throttle,
// not a selective one. Comparing a slot against its OWN history instead
// is selective by construction: a normal slot's ratio stays near 1 (no
// damping), only a slot that's currently spiking relative to its own
// past gets throttled. Default off; AdamW is untouched regardless of
// this flag.
// Phase 2 (2026-07-22): request.lion_trust_ratio_*_override wins over the
// cached env value -- same sentinel/worker-caching rationale as the four
// lion_* functions above.
static bool trust_ratio_enabled(const NativeRequest& request) {
    if (request.lion_trust_ratio_enabled_override >= 0) {
        return request.lion_trust_ratio_enabled_override != 0;
    }
    static const bool v = [] {
        const char* e = std::getenv("IDA_NATIVE_OPTIMIZER_TRUST_RATIO");
        return e && e[0] == '1';
    }();
    return v;
}
static float trust_ratio_lo(const NativeRequest& request) {
    if (request.lion_trust_ratio_lo_override >= 0.0f) return request.lion_trust_ratio_lo_override;
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_TRUST_RATIO_LO");
        return (e && e[0]) ? static_cast<float>(std::atof(e)) : 0.1f;
    }();
    return v;
}
static float trust_ratio_hi(const NativeRequest& request) {
    if (request.lion_trust_ratio_hi_override >= 0.0f) return request.lion_trust_ratio_hi_override;
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_TRUST_RATIO_HI");
        return (e && e[0]) ? static_cast<float>(std::atof(e)) : 10.0f;
    }();
    return v;
}

// AdamW moment (m/v) buffers are the one weight-adjacent allocation that's
// only touched once per REAL optimizer step (once per grad_accum window),
// not every microstep — unlike weights (read every forward), gradients
// (written every backward), and activations (both, every microstep). That
// makes them the safe target for host offload: allocate in pinned,
// GPU-mapped host RAM instead of the device arena, so adamw_step's kernel
// runs completely unchanged (CUDA's UVA makes the mapped host pointer look
// like an ordinary device pointer to kernel code) but the traffic crosses
// PCIe instead of HBM. Bounded, infrequent cost (~16-32GB of PCIe traffic
// once per optimizer step for a 1B-param model, ~1-2s) in exchange for
// freeing the full m/v footprint (~4-8GB) of device VRAM — the exact
// resource that was the binding constraint behind config_tuner's AI OOM
// backoff (78.97/79.18GiB on the full 16-layer model), not the gradient-
// explosion instability this session's other fixes already resolved.
// 0/unset = disabled (lab-test flag); weights and gradients are NEVER
// offloaded by this flag, only m/v.
static bool optim_state_host_offload_enabled() {
    static const bool v = [] {
        // Era 13 default: ON (recipe-verified; -9.8 GB VRAM on the AI body
        // for ~0.74 s PCIe per optimizer step = 0.14% of a ga=64 window).
        // =0 restores device-resident m/v.
        const char* e = std::getenv("IDA_NATIVE_OPTIM_STATE_HOST_OFFLOAD");
        return !e || e[0] == '1';
    }();
    return v;
}

template <typename T>
static T* alloc_optim_host_mapped(std::size_t n, void** out_host_backing) {
    void* host_ptr = nullptr;
    IDA_CUDA_CHECK(cudaHostAlloc(&host_ptr, n * sizeof(T), cudaHostAllocMapped));
    std::memset(host_ptr, 0, n * sizeof(T));
    T* dev_ptr = nullptr;
    IDA_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dev_ptr), host_ptr, 0));
    *out_host_backing = host_ptr;
    return dev_ptr;
}

LatticeOptState allocate_lattice_opt(const NativeRequest& request, const LatticeWeights& w, NativeArena& arena) {
    const int H = w.hidden_size, I = w.intermediate_size;
    const int L = w.num_layers,  V = w.vocab_size;
    const std::size_t KV = static_cast<std::size_t>(w.kv_heads > 0 ? w.kv_heads : w.heads) *
        static_cast<std::size_t>(H / w.heads);
    LatticeOptState opt{};

    opt.optimizer_state_bf16 = optimizer_state_uses_bf16(request);
    const bool host_offload = optim_state_host_offload_enabled();
    // Stage 2b (2026-07-16): Lion is a single-moment optimizer by
    // construction -- it never reads or writes v (see the use_lion branch
    // in the optimizer loop below). Stage 2a left v allocated-but-unused
    // to validate the update rule first without also risking a memory-
    // layout refactor; now that lr_scale~0.3 gates well against both
    // AdamW arms, skip the v allocation entirely -- halves optimizer
    // state memory (and the host-offload PCIe traffic that goes with it)
    // for every parameter group when Lion is selected.
    const bool skip_v_alloc = optimizer_uses_lion(request);

    auto az = [&](std::size_t n) -> OptStateTensor {
        OptStateTensor t{};
        if (host_offload) {
            if (opt.optimizer_state_bf16) {
                t.bf16 = alloc_optim_host_mapped<__nv_bfloat16>(n, &t.host_backing);
            } else {
                t.f32 = alloc_optim_host_mapped<float>(n, &t.host_backing);
            }
            return t;  // already zeroed by alloc_optim_host_mapped
        }
        if (opt.optimizer_state_bf16) {
            t.bf16 = alloc_bf16(n, arena);
            IDA_CUDA_CHECK(cudaMemsetAsync(t.bf16, 0, n * sizeof(__nv_bfloat16), arena.stream));
        } else {
            t.f32 = alloc_f32(n, arena);
            k_zeros_f32<<<ceildiv(n, 256), 256, 0, arena.stream>>>(t.f32, n);
        }
        return t;
    };
    auto az_v = [&](std::size_t n) -> OptStateTensor {
        return skip_v_alloc ? OptStateTensor{} : az(n);
    };
    if (w.owns_embedding) {
        opt.m_embed = az(V * H); opt.v_embed = az_v(V * H);
    }
        if (w.position_embeddings) opt.m_position_embeddings = az(static_cast<std::size_t>(w.max_position_embeddings) * H), opt.v_position_embeddings = az_v(static_cast<std::size_t>(w.max_position_embeddings) * H);
    if (w.owns_output) {
        opt.m_fnorm   = az(H);     opt.v_fnorm   = az_v(H);
        opt.m_lm_head = az(V * H); opt.v_lm_head = az_v(V * H);
    }
        if (w.final_norm_bias) opt.m_fnorm_bias = az(H), opt.v_fnorm_bias = az_v(H);
    if (w.lrss_enabled) {
        const std::size_t J = w.lrss_scales;
        opt.m_lrss_q  = az(static_cast<std::size_t>(H) * H);     opt.v_lrss_q  = az_v(static_cast<std::size_t>(H) * H);
        opt.m_lrss_k  = az(static_cast<std::size_t>(H) * H);     opt.v_lrss_k  = az_v(static_cast<std::size_t>(H) * H);
        opt.m_lrss_gw = az(static_cast<std::size_t>(H) * 2 * H); opt.v_lrss_gw = az_v(static_cast<std::size_t>(H) * 2 * H);
        opt.m_lrss_gb = az(H);                                   opt.v_lrss_gb = az_v(H);
        opt.m_lrss_lt = az(J);                                   opt.v_lrss_lt = az_v(J);
        opt.m_lrss_sw = az(J);                                   opt.v_lrss_sw = az_v(J);
        if (w.lss_rank > 0) {
            const std::size_t nd = static_cast<std::size_t>(w.lss_rank) *
                (static_cast<std::size_t>(2 * H) + static_cast<std::size_t>(w.pss_spike_joint_dim));
            const std::size_t nu = static_cast<std::size_t>(H) * w.lss_rank;
            opt.m_lss_dn = az(nd); opt.v_lss_dn = az_v(nd);
            opt.m_lss_up = az(nu); opt.v_lss_up = az_v(nu);
        }
    }
    if (w.pss_pred_rank > 0) {
        const std::size_t n = static_cast<std::size_t>(H) * w.pss_pred_rank;
        opt.m_pss_pred_dn = az(n); opt.v_pss_pred_dn = az_v(n);
        opt.m_pss_pred_up = az(n); opt.v_pss_pred_up = az_v(n);
    }

    opt.layers = new LatticeOptLayer[L];
    for (int l = 0; l < L; ++l) {
        auto& lo = opt.layers[l];
        lo.m_anorm = az(H);     lo.v_anorm = az_v(H);
        lo.m_q     = az(H * H); lo.v_q     = az_v(H * H);
        lo.m_k     = az(static_cast<std::size_t>(H) * KV); lo.v_k = az_v(static_cast<std::size_t>(H) * KV);
        lo.m_v     = az(static_cast<std::size_t>(H) * KV); lo.v_v = az_v(static_cast<std::size_t>(H) * KV);
        lo.m_o     = az(H * H); lo.v_o     = az_v(H * H);
        lo.m_fnorm = az(H);     lo.v_fnorm = az_v(H);
        if (w.layers[l].q_bias != nullptr) {
            lo.m_q_bias = az(static_cast<std::size_t>(H));  lo.v_q_bias = az_v(static_cast<std::size_t>(H));
            lo.m_k_bias = az(static_cast<std::size_t>(KV)); lo.v_k_bias = az_v(static_cast<std::size_t>(KV));
            lo.m_v_bias = az(static_cast<std::size_t>(KV)); lo.v_v_bias = az_v(static_cast<std::size_t>(KV));
            if (w.layers[l].o_bias) {
                lo.m_o_bias = az(H); lo.v_o_bias = az_v(H);
                lo.m_attn_norm_bias = az(H); lo.v_attn_norm_bias = az_v(H);
                lo.m_ffn_norm_bias = az(H); lo.v_ffn_norm_bias = az_v(H);
                lo.m_ffn_in_bias = az(I); lo.v_ffn_in_bias = az_v(I);
                lo.m_ffn_out_bias = az(H); lo.v_ffn_out_bias = az_v(H);
            }
        }
        lo.m_gate  = az(H * I); lo.v_gate  = az_v(H * I);
        lo.m_up    = az(H * I); lo.v_up    = az_v(H * I);
        lo.m_down  = az(I * H); lo.v_down  = az_v(I * H);

        // Cognitive-architecture sparse MoE port: mirrors w.layers[l]'s
        // already-resolved sizes 1:1 (allocated only when num_experts>0).
        const auto& lw = w.layers[l];
        if (lw.num_experts > 0 && lw.moe_kind == 0) {
            const std::size_t I_e = static_cast<std::size_t>(lw.expert_intermediate_size);
            if (lw.moe_shared_trunk) {
                const std::size_t n_in  = static_cast<std::size_t>(I) * H;
                const std::size_t n_out = static_cast<std::size_t>(H) * I;
                lo.m_moe_trunk_in  = az(n_in);  lo.v_moe_trunk_in  = az_v(n_in);
                lo.m_moe_trunk_out = az(n_out); lo.v_moe_trunk_out = az_v(n_out);
            }
            const std::size_t n_expert_in  = static_cast<std::size_t>(lw.num_experts) * I_e * H;
            const std::size_t n_expert_out = static_cast<std::size_t>(lw.num_experts) * H * I_e;
            lo.m_moe_expert_in  = az(n_expert_in);  lo.v_moe_expert_in  = az_v(n_expert_in);
            lo.m_moe_expert_out = az(n_expert_out); lo.v_moe_expert_out = az_v(n_expert_out);

            const std::size_t n_pp = static_cast<std::size_t>(lw.num_routes) * H;
            lo.m_pressure_proj = az(n_pp); lo.v_pressure_proj = az_v(n_pp);
            const std::size_t n_pm = static_cast<std::size_t>(H) * lw.num_routes;
            lo.m_pressure_mod  = az(n_pm); lo.v_pressure_mod  = az_v(n_pm);
            const std::size_t n_rs = static_cast<std::size_t>(lw.num_experts) * H;
            lo.m_router_score  = az(n_rs); lo.v_router_score  = az_v(n_rs);
            if (lw.pressure_to_routes_w != nullptr) {
                const std::size_t n_ptr = static_cast<std::size_t>(lw.num_experts) * lw.num_routes;
                lo.m_pressure_to_routes = az(n_ptr); lo.v_pressure_to_routes = az_v(n_ptr);
            }
        } else if (lw.num_experts > 0 && lw.moe_kind == 1) {
            const std::size_t I_e = static_cast<std::size_t>(lw.expert_intermediate_size);
            const std::size_t n_gu = static_cast<std::size_t>(lw.num_experts) * I_e * H;
            const std::size_t n_dn = static_cast<std::size_t>(lw.num_experts) * H * I_e;
            lo.m_generic_gate = az(n_gu); lo.v_generic_gate = az_v(n_gu);
            lo.m_generic_up   = az(n_gu); lo.v_generic_up   = az_v(n_gu);
            lo.m_generic_down = az(n_dn); lo.v_generic_down = az_v(n_dn);
            const std::size_t n_rs = static_cast<std::size_t>(lw.num_experts) * H;
            lo.m_router_score = az(n_rs); lo.v_router_score = az_v(n_rs);
            if (lw.shared_expert_width > 0) {
                const std::size_t Ws = static_cast<std::size_t>(lw.shared_expert_width);
                const std::size_t n_sgu = Ws * H;
                const std::size_t n_sdn = static_cast<std::size_t>(H) * Ws;
                lo.m_generic_shared_gate = az(n_sgu); lo.v_generic_shared_gate = az_v(n_sgu);
                lo.m_generic_shared_up   = az(n_sgu); lo.v_generic_shared_up   = az_v(n_sgu);
                lo.m_generic_shared_down = az(n_sdn); lo.v_generic_shared_down = az_v(n_sdn);
                lo.m_generic_shared_gate_score = az(H); lo.v_generic_shared_gate_score = az_v(H);
            }
        }
    }
    IDA_CUDA_CHECK(cudaStreamSynchronize(arena.stream));
    return opt;
}

void free_lattice_opt(LatticeOptState& opt, const LatticeWeights& w, NativeArena& arena) {
    auto f = [&](OptStateTensor t) {
        if (t.host_backing) {
            // Host-offloaded: f32/bf16 above are the device-mapped ALIAS,
            // not a real device allocation — must free via the original
            // host pointer, never cudaFreeAsync (that would target an
            // address cudaMallocAsync's pool never allocated).
            IDA_CUDA_CHECK(cudaFreeHost(t.host_backing));
            return;
        }
        if (t.f32) IDA_CUDA_CHECK(cudaFreeAsync(t.f32, arena.stream));
        if (t.bf16) IDA_CUDA_CHECK(cudaFreeAsync(t.bf16, arena.stream));
    };
    f(opt.m_embed); f(opt.v_embed); f(opt.m_position_embeddings); f(opt.v_position_embeddings);
    f(opt.m_fnorm); f(opt.v_fnorm); f(opt.m_fnorm_bias); f(opt.v_fnorm_bias);
    f(opt.m_lm_head); f(opt.v_lm_head);
    f(opt.m_lrss_q); f(opt.v_lrss_q); f(opt.m_lrss_k); f(opt.v_lrss_k);
    f(opt.m_lrss_gw); f(opt.v_lrss_gw); f(opt.m_lrss_gb); f(opt.v_lrss_gb);
    f(opt.m_lrss_lt); f(opt.v_lrss_lt); f(opt.m_lrss_sw); f(opt.v_lrss_sw);
    f(opt.m_lss_dn); f(opt.v_lss_dn); f(opt.m_lss_up); f(opt.v_lss_up);
    f(opt.m_pss_pred_dn); f(opt.v_pss_pred_dn); f(opt.m_pss_pred_up); f(opt.v_pss_pred_up);
    for (int l = 0; l < w.num_layers; ++l) {
        auto& lo = opt.layers[l];
        f(lo.m_anorm); f(lo.v_anorm);
        f(lo.m_q); f(lo.v_q); f(lo.m_k); f(lo.v_k);
        f(lo.m_v); f(lo.v_v); f(lo.m_o); f(lo.v_o);
        f(lo.m_fnorm); f(lo.v_fnorm);
        f(lo.m_q_bias); f(lo.v_q_bias); f(lo.m_k_bias); f(lo.v_k_bias); f(lo.m_v_bias); f(lo.v_v_bias);
        f(lo.m_o_bias); f(lo.v_o_bias); f(lo.m_attn_norm_bias); f(lo.v_attn_norm_bias); f(lo.m_ffn_norm_bias); f(lo.v_ffn_norm_bias); f(lo.m_ffn_in_bias); f(lo.v_ffn_in_bias); f(lo.m_ffn_out_bias); f(lo.v_ffn_out_bias);
        f(lo.m_gate); f(lo.v_gate); f(lo.m_up); f(lo.v_up);
        f(lo.m_down); f(lo.v_down);
        f(lo.m_moe_trunk_in); f(lo.v_moe_trunk_in);
        f(lo.m_moe_trunk_out); f(lo.v_moe_trunk_out);
        f(lo.m_moe_expert_in); f(lo.v_moe_expert_in);
        f(lo.m_moe_expert_out); f(lo.v_moe_expert_out);
        f(lo.m_pressure_proj); f(lo.v_pressure_proj);
        f(lo.m_pressure_mod); f(lo.v_pressure_mod);
        f(lo.m_router_score); f(lo.v_router_score);
        f(lo.m_pressure_to_routes); f(lo.v_pressure_to_routes);
        f(lo.m_generic_gate); f(lo.v_generic_gate);
        f(lo.m_generic_up); f(lo.v_generic_up);
        f(lo.m_generic_down); f(lo.v_generic_down);
        f(lo.m_generic_shared_gate); f(lo.v_generic_shared_gate);
        f(lo.m_generic_shared_up); f(lo.v_generic_shared_up);
        f(lo.m_generic_shared_down); f(lo.v_generic_shared_down);
        f(lo.m_generic_shared_gate_score); f(lo.v_generic_shared_gate_score);
    }
    delete[] opt.layers; opt.layers = nullptr;
    IDA_CUDA_CHECK(cudaStreamSynchronize(arena.stream));
}

// ─── persistent FP32 gradient buffers ────────────────────────────────────────
// Full-model gradients accumulate here across micro-steps (gradient
// accumulation), then global-norm clipping and AdamW run once per optimizer
// step.  ~4 bytes/param — trivial next to the activation buffers.

struct LatticeGradLayer {
    __nv_bfloat16* g_anorm{}; __nv_bfloat16* g_q{}; __nv_bfloat16* g_k{}; __nv_bfloat16* g_v{}; __nv_bfloat16* g_o{};
    // Qwen2-family QKV bias (2026-08-23). FP32 accumulator, matching the
    // column-sum-reduction convention (g_lrss_gate_b et al.), not the bf16
    // per-weight-matrix convention g_q/g_k/g_v above use.
    float* g_q_bias{}; float* g_k_bias{}; float* g_v_bias{};
    float* g_o_bias{};
    float* g_attn_norm_bias{};
    float* g_ffn_norm_bias{};
    float* g_ffn_in_bias{};
    float* g_ffn_out_bias{};
    __nv_bfloat16* g_fnorm{}; __nv_bfloat16* g_gate{}; __nv_bfloat16* g_up{}; __nv_bfloat16* g_down{};
    // Cognitive-architecture sparse MoE port (2026-07-21), mirrors
    // LatticeLayerWeights (allocated only when num_experts>0).
    __nv_bfloat16* g_moe_trunk_in{};  __nv_bfloat16* g_moe_trunk_out{};
    __nv_bfloat16* g_moe_expert_in{}; __nv_bfloat16* g_moe_expert_out{};
    __nv_bfloat16* g_pressure_proj{}; __nv_bfloat16* g_pressure_mod{};
    __nv_bfloat16* g_router_score{};  __nv_bfloat16* g_pressure_to_routes{};
    // Generic per-token top-k SwiGLU MoE (2026-08-23), mirrors
    // LatticeLayerWeights' generic_* fields (allocated only when
    // moe_kind==1). g_router_score above is shared by both modes.
    __nv_bfloat16* g_generic_gate{}; __nv_bfloat16* g_generic_up{}; __nv_bfloat16* g_generic_down{};
    // Shared expert (2026-08-23), mirrors LatticeLayerWeights' generic_shared_*
    // fields (allocated only when shared_expert_width>0).
    __nv_bfloat16* g_generic_shared_gate{}; __nv_bfloat16* g_generic_shared_up{};
    __nv_bfloat16* g_generic_shared_down{}; __nv_bfloat16* g_generic_shared_gate_score{};
};

struct LatticeGrads {
    float* g_embed{};
    float* g_position_embeddings{};
    float* g_fnorm{};
    float* g_fnorm_bias{};
    float* g_lm_head{};
    LatticeGradLayer* layers{};
    // LRSS (allocated only when w.lrss_enabled)
    float* g_lrss_query{};
    float* g_lrss_key{};
    float* g_lrss_gate_w{};
    float* g_lrss_gate_b{};
    float* g_lrss_log_tau{};
    float* g_lrss_scale_w{};
    float* g_lss_down{};
    float* g_lss_up{};
    // PSS Stage 2 (allocated only when w.pss_pred_rank > 0)
    float* g_pss_pred_down{};
    float* g_pss_pred_up{};
};

static LatticeGrads allocate_lattice_grads(const LatticeWeights& w, NativeArena& arena) {
    const int H = w.hidden_size, I = w.intermediate_size;
    const int L = w.num_layers,  V = w.vocab_size;
    const std::size_t KV = static_cast<std::size_t>(w.kv_heads > 0 ? w.kv_heads : w.heads) *
        static_cast<std::size_t>(H / w.heads);
    LatticeGrads g{};
    auto az = [&](std::size_t n) -> float* {
        float* p = alloc_f32(n, arena);
        k_zeros_f32<<<ceildiv(n, 256), 256, 0, arena.stream>>>(p, n);
        return p;
    };
    auto abz = [&](std::size_t n) -> __nv_bfloat16* {
        __nv_bfloat16* p = alloc_bf16(n, arena);
        k_zeros_bf16<<<ceildiv(n, 256), 256, 0, arena.stream>>>(p, n);
        return p;
    };
    if (w.owns_embedding) {
        g.g_embed = az(static_cast<std::size_t>(V) * H);
        if (w.position_embeddings) g.g_position_embeddings = az(static_cast<std::size_t>(w.max_position_embeddings) * H);
    }
    if (w.owns_output) {
        g.g_fnorm   = az(H);
        g.g_lm_head = az(static_cast<std::size_t>(V) * H);
        if (w.final_norm_bias) g.g_fnorm_bias = az(H);
    }
    if (w.lrss_enabled) {
        const int J = w.lrss_scales;
        g.g_lrss_query   = az(static_cast<std::size_t>(H) * H);
        g.g_lrss_key     = az(static_cast<std::size_t>(H) * H);
        g.g_lrss_gate_w  = az(static_cast<std::size_t>(H) * 2 * H);
        g.g_lrss_gate_b  = az(H);
        g.g_lrss_log_tau = az(J);
        g.g_lrss_scale_w = az(J);
        if (w.lss_rank > 0) {
            g.g_lss_down = az(static_cast<std::size_t>(w.lss_rank) *
                (static_cast<std::size_t>(2 * H) + static_cast<std::size_t>(w.pss_spike_joint_dim)));
            g.g_lss_up   = az(static_cast<std::size_t>(H) * w.lss_rank);
        }
    }
    if (w.pss_pred_rank > 0) {
        const std::size_t n = static_cast<std::size_t>(H) * w.pss_pred_rank;
        g.g_pss_pred_down = az(n);
        g.g_pss_pred_up   = az(n);
    }
    g.layers = new LatticeGradLayer[L];
    for (int l = 0; l < L; ++l) {
        auto& gl = g.layers[l];
        gl.g_anorm = abz(H);
        gl.g_q     = abz(H * H);
        gl.g_k     = abz(static_cast<std::size_t>(H) * KV);
        gl.g_v     = abz(static_cast<std::size_t>(H) * KV);
        gl.g_o     = abz(H * H);
        gl.g_fnorm = abz(H);
        if (w.layers[l].q_bias != nullptr) {
            gl.g_q_bias = az(static_cast<std::size_t>(H));
            gl.g_k_bias = az(static_cast<std::size_t>(KV));
            gl.g_v_bias = az(static_cast<std::size_t>(KV));
            if (w.layers[l].o_bias) {
                gl.g_o_bias = az(H);
                gl.g_attn_norm_bias = az(H);
                gl.g_ffn_norm_bias = az(H);
                gl.g_ffn_in_bias = az(I);
                gl.g_ffn_out_bias = az(H);
            }
        }
        gl.g_gate  = abz(H * I); gl.g_up = abz(H * I);
        gl.g_down  = abz(I * H);

        // Cognitive-architecture sparse MoE port: mirrors w.layers[l]'s
        // already-resolved sizes 1:1 (allocated only when num_experts>0).
        const auto& lw = w.layers[l];
        if (lw.num_experts > 0 && lw.moe_kind == 0) {
            const std::size_t I_e = static_cast<std::size_t>(lw.expert_intermediate_size);
            if (lw.moe_shared_trunk) {
                gl.g_moe_trunk_in  = abz(static_cast<std::size_t>(I) * H);
                gl.g_moe_trunk_out = abz(static_cast<std::size_t>(H) * I);
            }
            gl.g_moe_expert_in  = abz(static_cast<std::size_t>(lw.num_experts) * I_e * H);
            gl.g_moe_expert_out = abz(static_cast<std::size_t>(lw.num_experts) * H * I_e);
            gl.g_pressure_proj  = abz(static_cast<std::size_t>(lw.num_routes) * H);
            gl.g_pressure_mod   = abz(static_cast<std::size_t>(H) * lw.num_routes);
            gl.g_router_score   = abz(static_cast<std::size_t>(lw.num_experts) * H);
            if (lw.pressure_to_routes_w != nullptr) {
                gl.g_pressure_to_routes = abz(static_cast<std::size_t>(lw.num_experts) * lw.num_routes);
            }
        } else if (lw.num_experts > 0 && lw.moe_kind == 1) {
            const std::size_t I_e = static_cast<std::size_t>(lw.expert_intermediate_size);
            gl.g_generic_gate = abz(static_cast<std::size_t>(lw.num_experts) * I_e * H);
            gl.g_generic_up   = abz(static_cast<std::size_t>(lw.num_experts) * I_e * H);
            gl.g_generic_down = abz(static_cast<std::size_t>(lw.num_experts) * H * I_e);
            gl.g_router_score = abz(static_cast<std::size_t>(lw.num_experts) * H);
            if (lw.shared_expert_width > 0) {
                const std::size_t Ws = static_cast<std::size_t>(lw.shared_expert_width);
                gl.g_generic_shared_gate = abz(Ws * H);
                gl.g_generic_shared_up   = abz(Ws * H);
                gl.g_generic_shared_down = abz(static_cast<std::size_t>(H) * Ws);
                gl.g_generic_shared_gate_score = abz(static_cast<std::size_t>(H));
            }
        }
    }
    IDA_CUDA_CHECK(cudaStreamSynchronize(arena.stream));
    return g;
}

static void free_lattice_grads(LatticeGrads& g, const LatticeWeights& w, NativeArena& arena) {
    auto f = [&](float* p) { if (p) IDA_CUDA_CHECK(cudaFreeAsync(p, arena.stream)); };
    auto b = [&](__nv_bfloat16* p) { if (p) IDA_CUDA_CHECK(cudaFreeAsync(p, arena.stream)); };
    f(g.g_embed); f(g.g_position_embeddings); f(g.g_fnorm); f(g.g_fnorm_bias); f(g.g_lm_head);
    f(g.g_lrss_query); f(g.g_lrss_key); f(g.g_lrss_gate_w);
    f(g.g_lrss_gate_b); f(g.g_lrss_log_tau); f(g.g_lrss_scale_w);
    f(g.g_lss_down); f(g.g_lss_up);
    f(g.g_pss_pred_down); f(g.g_pss_pred_up);
    for (int l = 0; l < w.num_layers; ++l) {
        auto& gl = g.layers[l];
        b(gl.g_anorm); b(gl.g_q); b(gl.g_k); b(gl.g_v); b(gl.g_o);
        b(gl.g_fnorm); b(gl.g_gate); b(gl.g_up); b(gl.g_down);
        f(gl.g_q_bias); f(gl.g_k_bias); f(gl.g_v_bias);
        f(gl.g_o_bias); f(gl.g_attn_norm_bias); f(gl.g_ffn_norm_bias); f(gl.g_ffn_in_bias); f(gl.g_ffn_out_bias);
        b(gl.g_moe_trunk_in); b(gl.g_moe_trunk_out);
        b(gl.g_moe_expert_in); b(gl.g_moe_expert_out);
        b(gl.g_pressure_proj); b(gl.g_pressure_mod);
        b(gl.g_router_score); b(gl.g_pressure_to_routes);
        b(gl.g_generic_gate); b(gl.g_generic_up); b(gl.g_generic_down);
        b(gl.g_generic_shared_gate); b(gl.g_generic_shared_up);
        b(gl.g_generic_shared_down); b(gl.g_generic_shared_gate_score);
    }
    delete[] g.layers; g.layers = nullptr;
    IDA_CUDA_CHECK(cudaStreamSynchronize(arena.stream));
}

// One entry per parameter tensor: pairs weight/opt/grad for the optimizer loop.

// Sum of squares over a BF16 tensor, accumulated into a device scalar.
// Mirrors sq_sum_acc_f32 (which is float-only) so PARAMETER norms can be
// tracked, not just gradient norms.
__global__ void k_sq_sum_acc_bf16(
    const __nv_bfloat16* __restrict__ x, std::size_t n, float* __restrict__ acc
) {
    __shared__ float sm[256 / 32];
    float s = 0.0f;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float v = __bfloat162float(x[i]);
        s += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int w = 0; w < blockDim.x / 32; ++w) t += sm[w];
        atomicAdd(acc, t);
    }
}

static void sq_sum_acc_bf16(const __nv_bfloat16* w, std::size_t n, float* acc, cudaStream_t s) {
    if (!w || n == 0) return;
    const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((n + 255) / 256, 1024));
    k_sq_sum_acc_bf16<<<blocks ? blocks : 1, 256, 0, s>>>(w, n, acc);
}

// Slot trace cadence. Unconditional -- the existing per-slot dump is gated on
// grad_norm > 1e6, i.e. only pathological steps produce per-slot data, so a
// HEALTHY run emits none and cannot be characterised. You need the healthy
// baseline to know what normal looks like.
static bool slot_trace_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("IDA_NATIVE_SLOT_TRACE");
        return e && e[0] == '1';
    }();
    return v;
}
static int slot_trace_every() {
    static const int v = [] {
        const char* e = std::getenv("IDA_NATIVE_SLOT_TRACE_EVERY");
        if (!e || !e[0]) return 10;
        char* end = nullptr; const long p = std::strtol(e, &end, 10);
        return (end == e || p < 1) ? 10 : static_cast<int>(p);
    }();
    return v;
}

// ── #44: per-expert dW norms ────────────────────────────────────────────────
// One block per expert; each block reduces its own [Ie*H] contiguous chunk of
// the shared expert weight-gradient buffer. Mirrors k_moe_expert_util_par's
// shape (blocks = experts) so the two reports line up one-to-one.
static __global__ void k_expert_grad_sumsq(
    const float* __restrict__ g, long long chunk, float* __restrict__ out
) {
    const long long base = static_cast<long long>(blockIdx.x) * chunk;
    double acc = 0.0;
    for (long long i = threadIdx.x; i < chunk; i += blockDim.x) {
        const double v = static_cast<double>(g[base + i]);
        acc += v * v;
    }
    __shared__ double sm[256];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) sm[threadIdx.x] += sm[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = static_cast<float>(sqrt(sm[0]));
}

static bool expert_grad_util_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("IDA_NATIVE_EXPERT_GRAD_UTIL");
        return e && e[0] == '1';
    }();
    return v;
}

// Reports the per-expert L2 norm of the ACCUMULATED dW that is about to update
// the weights. Pair it with [expert-util]'s share line: share ~0 with a large
// norm here means gradient is reaching an expert that never routed.
static void moe_expert_grad_report(
    const char* slot_name, const float* d_g, long long n, int E,
    int opt_step, cudaStream_t s
) {
    if (!expert_grad_util_enabled() || E <= 1 || !d_g || n <= 0) return;
    if (n % E != 0) {
        // Not an even [E, ...] split -- report rather than guess a layout.
        std::fprintf(stderr,
            "[expert-grad] opt_step=%d %s SKIPPED: n=%lld not divisible by E=%d\n",
            opt_step, slot_name, static_cast<long long>(n), E);
        return;
    }
    const long long chunk = n / E;
    float* d_out = nullptr;
    if (cudaMallocAsync(&d_out, static_cast<std::size_t>(E) * sizeof(float), s) != cudaSuccess) return;
    k_expert_grad_sumsq<<<E, 256, 0, s>>>(d_g, chunk, d_out);
    std::vector<float> h(static_cast<std::size_t>(E), 0.0f);
    cudaMemcpyAsync(h.data(), d_out, static_cast<std::size_t>(E) * sizeof(float),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    cudaFreeAsync(d_out, s);
    double tot = 0.0;
    for (int e = 0; e < E; ++e) tot += h[e];
    std::string norms, shares;
    char buf[40];
    for (int e = 0; e < E; ++e) {
        std::snprintf(buf, sizeof(buf), "%.6g", h[e]);
        norms += buf;
        std::snprintf(buf, sizeof(buf), "%.4f", tot > 0.0 ? h[e] / tot : 0.0);
        shares += buf;
        if (e + 1 < E) { norms += ","; shares += ","; }
    }
    std::fprintf(stderr,
        "[expert-grad] opt_step=%d %s E=%d chunk=%lld norms=[%s] grad_share=[%s]\n",
        opt_step, slot_name, E, static_cast<long long>(chunk),
        norms.c_str(), shares.c_str());
}

struct ParamSlot {
    __nv_bfloat16* w;
    OptStateTensor m{};
    OptStateTensor v{};
    float* g{};
    float* fp8_amax{};
    std::size_t n;
    float wd;   // decoupled weight decay: matrices only, norm scales get 0
    __nv_bfloat16* g_bf16{};
};

static void slot_sq_sum_acc(const ParamSlot& slot, float* out, cudaStream_t stream) {
    if (slot.g_bf16) {
        sq_sum_acc_bf16(slot.g_bf16, slot.n, out, stream);
    } else if (slot.g) {
        sq_sum_acc_f32(slot.g, slot.n, out, stream);
    }
}

static void slot_scale_gradient(const ParamSlot& slot, float scale, cudaStream_t stream) {
    if (slot.g_bf16) {
        k_scale_bf16<<<ceildiv(slot.n, 256), 256, 0, stream>>>(slot.g_bf16, slot.n, scale);
    } else if (slot.g) {
        k_scale_f32<<<ceildiv(slot.n, 256), 256, 0, stream>>>(slot.g, slot.n, scale);
    }
}

static std::vector<ParamSlot> build_param_slots(
    LatticeWeights& w, LatticeOptState& opt, LatticeGrads& g, float wd
) {
    const std::size_t H = w.hidden_size, I = w.intermediate_size;
    const std::size_t V = w.vocab_size;
    const std::size_t KV = static_cast<std::size_t>(w.kv_heads > 0 ? w.kv_heads : w.heads) *
        static_cast<std::size_t>(w.hidden_size / w.heads);
        if (weight == nullptr) return;
    std::vector<ParamSlot> slots;
    auto add_bf16 = [&](auto* weight, OptStateTensor m, OptStateTensor v,
                        __nv_bfloat16* grad, std::size_t n, float weight_decay) {
        slots.push_back({weight, m, v, nullptr, nullptr, n, weight_decay});
        slots.back().g_bf16 = grad;
    };
    if (w.owns_embedding) {
        slots.push_back({w.embed, opt.m_embed, opt.v_embed, g.g_embed, nullptr, V * H, wd});
        if (w.position_embeddings) {
            slots.push_back({w.position_embeddings, opt.m_position_embeddings,
                             opt.v_position_embeddings, g.g_position_embeddings,
                             nullptr, static_cast<std::size_t>(w.max_position_embeddings) * H, wd});
        }
    }
    if (w.owns_output) {
        slots.push_back({w.lm_head, opt.m_lm_head, opt.v_lm_head, g.g_lm_head, nullptr, V * H, wd});
        slots.push_back({w.final_norm, opt.m_fnorm, opt.v_fnorm, g.g_fnorm, nullptr, H, 0.0f});
        if (w.final_norm_bias) {
            slots.push_back({w.final_norm_bias, opt.m_fnorm_bias, opt.v_fnorm_bias,
                             g.g_fnorm_bias, nullptr, H, 0.0f});
        }
    }
    if (w.lrss_enabled) {
        const std::size_t J = w.lrss_scales;
        slots.push_back({w.lrss_query,   opt.m_lrss_q,  opt.v_lrss_q,  g.g_lrss_query,   nullptr, H * H,     wd});
        slots.push_back({w.lrss_key,     opt.m_lrss_k,  opt.v_lrss_k,  g.g_lrss_key,     nullptr, H * H,     wd});
        slots.push_back({w.lrss_gate_w,  opt.m_lrss_gw, opt.v_lrss_gw, g.g_lrss_gate_w,  nullptr, H * 2 * H, wd});
        slots.push_back({w.lrss_gate_b,  opt.m_lrss_gb, opt.v_lrss_gb, g.g_lrss_gate_b,  nullptr, H,         0.0f});
        slots.push_back({w.lrss_log_tau, opt.m_lrss_lt, opt.v_lrss_lt, g.g_lrss_log_tau, nullptr, J,         0.0f});
        slots.push_back({w.lrss_scale_w, opt.m_lrss_sw, opt.v_lrss_sw, g.g_lrss_scale_w, nullptr, J,         0.0f});
        if (w.lss_rank > 0) {
            const std::size_t R = w.lss_rank;
            const std::size_t joint_dim = static_cast<std::size_t>(2 * H) +
                static_cast<std::size_t>(w.pss_spike_joint_dim);
            slots.push_back({w.lss_down, opt.m_lss_dn, opt.v_lss_dn, g.g_lss_down, nullptr, R * joint_dim, wd});
            slots.push_back({w.lss_up,   opt.m_lss_up, opt.v_lss_up, g.g_lss_up,   nullptr, H * R,     wd});
        }
    }
    if (w.pss_pred_rank > 0) {
        const std::size_t R = w.pss_pred_rank;
        slots.push_back({w.pss_pred_down, opt.m_pss_pred_dn, opt.v_pss_pred_dn, g.g_pss_pred_down, nullptr, H * R, wd});
        slots.push_back({w.pss_pred_up,   opt.m_pss_pred_up, opt.v_pss_pred_up, g.g_pss_pred_up,   nullptr, H * R, wd});
    }
    for (int l = 0; l < w.num_layers; ++l) {
        auto& lw = w.layers[l]; auto& lo = opt.layers[l]; auto& gl = g.layers[l];
        add_bf16(lw.attn_norm, lo.m_anorm, lo.v_anorm, gl.g_anorm, H, 0.0f);
        if (lw.attn_norm_bias) {
            slots.push_back({lw.attn_norm_bias, lo.m_attn_norm_bias, lo.v_attn_norm_bias,
                             gl.g_attn_norm_bias, nullptr, H, 0.0f});
        }
        add_bf16(lw.q_proj, lo.m_q, lo.v_q, gl.g_q, H * H, wd);
        add_bf16(lw.k_proj, lo.m_k, lo.v_k, gl.g_k, H * KV, wd);
        add_bf16(lw.v_proj, lo.m_v, lo.v_v, gl.g_v, H * KV, wd);
        add_bf16(lw.o_proj, lo.m_o, lo.v_o, gl.g_o, H * H, wd);
        if (lw.q_bias != nullptr) {
            slots.push_back({lw.q_bias, lo.m_q_bias, lo.v_q_bias, gl.g_q_bias, nullptr, static_cast<std::size_t>(H), 0.0f});
            slots.push_back({lw.k_bias, lo.m_k_bias, lo.v_k_bias, gl.g_k_bias, nullptr, static_cast<std::size_t>(KV), 0.0f});
            slots.push_back({lw.v_bias, lo.m_v_bias, lo.v_v_bias, gl.g_v_bias, nullptr, static_cast<std::size_t>(KV), 0.0f});
            if (lw.o_bias) {
                slots.push_back({lw.o_bias, lo.m_o_bias, lo.v_o_bias, gl.g_o_bias,
                                 nullptr, H, 0.0f});
            }
        }
        add_bf16(lw.ffn_norm, lo.m_fnorm, lo.v_fnorm, gl.g_fnorm, H, 0.0f);
        if (lw.ffn_norm_bias) {
            slots.push_back({lw.ffn_norm_bias, lo.m_ffn_norm_bias, lo.v_ffn_norm_bias,
                             gl.g_ffn_norm_bias, nullptr, H, 0.0f});
        }
        if (lw.moe_kind == 0) {
            add_bf16(lw.gate_proj, lo.m_gate, lo.v_gate, gl.g_gate, H * I, wd);
            add_bf16(lw.up_proj, lo.m_up, lo.v_up, gl.g_up, H * I, wd);
            add_bf16(lw.down_proj, lo.m_down, lo.v_down, gl.g_down, I * H, wd);
            if (lw.ffn_in_bias) {
                slots.push_back({lw.ffn_in_bias, lo.m_ffn_in_bias, lo.v_ffn_in_bias,
                                 gl.g_ffn_in_bias, nullptr, I, 0.0f});
                slots.push_back({lw.ffn_out_bias, lo.m_ffn_out_bias, lo.v_ffn_out_bias,
                                 gl.g_ffn_out_bias, nullptr, H, 0.0f});
            }
        }
        // Cognitive-architecture sparse MoE port: one slot per weight TYPE
        // (not per expert) -- expert_fc_in_w/expert_fc_out_w are already
        // one contiguous [num_experts, ...] allocation each, so a single
        // slot covers every expert's share of it.
        if (lw.num_experts > 0 && lw.moe_kind == 0) {
            const std::size_t I_e = static_cast<std::size_t>(lw.expert_intermediate_size);
            if (lw.moe_shared_trunk) {
                add_bf16(lw.trunk_fc_in_w, lo.m_moe_trunk_in, lo.v_moe_trunk_in,
                         gl.g_moe_trunk_in, static_cast<std::size_t>(I) * H, wd);
                add_bf16(lw.trunk_fc_out_w, lo.m_moe_trunk_out, lo.v_moe_trunk_out,
                         gl.g_moe_trunk_out, static_cast<std::size_t>(H) * I, wd);
            }
            add_bf16(lw.expert_fc_in_w, lo.m_moe_expert_in, lo.v_moe_expert_in,
                     gl.g_moe_expert_in, static_cast<std::size_t>(lw.num_experts) * I_e * H, wd);
            add_bf16(lw.expert_fc_out_w, lo.m_moe_expert_out, lo.v_moe_expert_out,
                     gl.g_moe_expert_out, static_cast<std::size_t>(lw.num_experts) * H * I_e, wd);
            add_bf16(lw.pressure_proj_w, lo.m_pressure_proj, lo.v_pressure_proj,
                     gl.g_pressure_proj, static_cast<std::size_t>(lw.num_routes) * H, wd);
            add_bf16(lw.pressure_mod_w, lo.m_pressure_mod, lo.v_pressure_mod,
                     gl.g_pressure_mod, static_cast<std::size_t>(H) * lw.num_routes, wd);
            add_bf16(lw.router_score_w, lo.m_router_score, lo.v_router_score,
                     gl.g_router_score, static_cast<std::size_t>(lw.num_experts) * H, wd);
            if (lw.pressure_to_routes_w != nullptr) {
                add_bf16(lw.pressure_to_routes_w, lo.m_pressure_to_routes,
                         lo.v_pressure_to_routes, gl.g_pressure_to_routes,
                         static_cast<std::size_t>(lw.num_experts) * lw.num_routes, wd);
            }
        } else if (lw.num_experts > 0 && lw.moe_kind == 1) {
            const std::size_t I_e = static_cast<std::size_t>(lw.expert_intermediate_size);
            add_bf16(lw.generic_gate_proj_w, lo.m_generic_gate, lo.v_generic_gate,
                     gl.g_generic_gate, static_cast<std::size_t>(lw.num_experts) * I_e * H, wd);
            add_bf16(lw.generic_up_proj_w, lo.m_generic_up, lo.v_generic_up,
                     gl.g_generic_up, static_cast<std::size_t>(lw.num_experts) * I_e * H, wd);
            add_bf16(lw.generic_down_proj_w, lo.m_generic_down, lo.v_generic_down,
                     gl.g_generic_down, static_cast<std::size_t>(lw.num_experts) * H * I_e, wd);
            add_bf16(lw.router_score_w, lo.m_router_score, lo.v_router_score,
                     gl.g_router_score, static_cast<std::size_t>(lw.num_experts) * H, wd);
            if (lw.shared_expert_width > 0) {
                const std::size_t Ws = static_cast<std::size_t>(lw.shared_expert_width);
                add_bf16(lw.generic_shared_gate_proj_w, lo.m_generic_shared_gate, lo.v_generic_shared_gate,
                         gl.g_generic_shared_gate, Ws * H, wd);
                add_bf16(lw.generic_shared_up_proj_w, lo.m_generic_shared_up, lo.v_generic_shared_up,
                         gl.g_generic_shared_up, Ws * H, wd);
                add_bf16(lw.generic_shared_down_proj_w, lo.m_generic_shared_down, lo.v_generic_shared_down,
                         gl.g_generic_shared_down, static_cast<std::size_t>(H) * Ws, wd);
                add_bf16(lw.generic_shared_gate_score_w, lo.m_generic_shared_gate_score,
                         lo.v_generic_shared_gate_score, gl.g_generic_shared_gate_score,
                         static_cast<std::size_t>(H), wd);
            }
        }
    }
    return slots;
}

// ── data-parallel across two GPUs, without NCCL ─────────────────────────────
//
// The seam is build_param_slots() above: it already enumerates EVERY trainable
// tensor with its gradient pointer and element count. A cross-device gradient
// sum is therefore a loop over that table -- none of the eight
// `for (int l = 0; l < w.num_layers; ++l)` loops in this file need to change,
// which is what makes two-card training tractable here at all.
//
// No NCCL, by requirement. With cudaDeviceEnablePeerAccess and unified virtual
// addressing, a kernel launched on the local device dereferences the peer's
// pointers directly, so the existing k_acc_f32 add kernel does the reduction
// in place with no staging buffer and no host round-trip. `nvidia-smi topo
// -p2p r` reports OK between the two H100s on this box.
//
// DEFAULT OFF. Nothing calls this unless data-parallel is explicitly enabled,
// so the single-card path is byte-identical.

// Idempotent, and tolerant of the "already enabled" case, which is not an
// error. Returns false when the pair genuinely cannot peer, so the caller
// fails closed to single-card rather than silently training on half the data.
bool ida_enable_peer_access(int local_dev, int peer_dev) {
    if (local_dev == peer_dev) return true;
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, local_dev, peer_dev) != cudaSuccess || !can) {
        cudaGetLastError();
        return false;
    }
    int prev = 0;
    if (cudaGetDevice(&prev) != cudaSuccess) return false;
    bool ok = true;
    if (cudaSetDevice(local_dev) != cudaSuccess) return false;
    const cudaError_t e = cudaDeviceEnablePeerAccess(peer_dev, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) ok = false;
    cudaGetLastError();   // clear the benign already-enabled sticky state
    cudaSetDevice(prev);
    return ok;
}

// Sum the peer replica's gradients into this replica's, slot by slot.
//
// Both replicas must have been built by build_param_slots() over models of
// identical shape, so slot i refers to the same tensor on both cards; the size
// equality check below is the guard against that assumption silently breaking
// (a mismatched pair would otherwise scribble across tensors and still produce
// a descending loss -- the failure mode this whole path has to be defended
// against).
//
// Caller contract: `stream` belongs to local_dev, local_dev is current, and the
// peer's backward has already been synchronised. Gradients are SUMMED, not
// averaged -- the caller divides by the replica count wherever it already
// scales by grad-accum, so the two scalings stay in one place.
bool reduce_grads_across_devices(
    const std::vector<ParamSlot>& local_slots,
    const std::vector<ParamSlot>& peer_slots,
    int local_dev, int peer_dev, cudaStream_t stream
) {
    if (local_slots.size() != peer_slots.size()) {
        std::fprintf(stderr,
            "[ida_native_train] data-parallel: slot count mismatch (%zu vs %zu) -- "
            "refusing to reduce\n", local_slots.size(), peer_slots.size());
        return false;
    }
    if (!ida_enable_peer_access(local_dev, peer_dev)) {
        std::fprintf(stderr,
            "[ida_native_train] data-parallel: no peer access %d<-%d -- refusing "
            "to reduce\n", local_dev, peer_dev);
        return false;
    }
    for (std::size_t i = 0; i < local_slots.size(); ++i) {
        const ParamSlot& a = local_slots[i];
        const ParamSlot& b = peer_slots[i];
        if (a.n != b.n) {
            std::fprintf(stderr,
                "[ida_native_train] data-parallel: slot %zu size mismatch "
                "(%zu vs %zu) -- refusing to reduce\n", i, a.n, b.n);
            return false;
        }
        if (a.g == nullptr || b.g == nullptr || a.n == 0) continue;
        k_acc_f32<<<ceildiv(a.n, 256), 256, 0, stream>>>(a.g, b.g, a.n);
    }
    return true;
}


static std::vector<std::string> build_param_slot_names(const LatticeWeights& w) {
    std::vector<std::string> names;
    if (w.owns_embedding) names.push_back("embed");
    if (w.owns_output) {
        names.push_back("lm_head");
        names.push_back("final_norm");
    }
    if (w.lrss_enabled) {
        names.push_back("lrss.query");
        names.push_back("lrss.key");
        names.push_back("lrss.gate_w");
        names.push_back("lrss.gate_b");
        names.push_back("lrss.log_tau");
        names.push_back("lrss.scale_w");
        if (w.lss_rank > 0) {
            names.push_back("lss.down");
            names.push_back("lss.up");
        }
    }
    if (w.pss_pred_rank > 0) {
        names.push_back("pss.pred_down");
        names.push_back("pss.pred_up");
    }
    for (int l = 0; l < w.num_layers; ++l) {
        const std::string p = "L" + std::to_string(w.layer_offset + l) + ".";
        names.push_back(p + "attn_norm");
        names.push_back(p + "q_proj");
        names.push_back(p + "k_proj");
        names.push_back(p + "v_proj");
        names.push_back(p + "o_proj");
        names.push_back(p + "ffn_norm");
        names.push_back(p + "gate_proj");
        names.push_back(p + "up_proj");
        names.push_back(p + "down_proj");
        if (w.layers[l].num_experts > 0) {
            if (w.layers[l].moe_shared_trunk) {
                names.push_back(p + "moe.trunk_fc_in");
                names.push_back(p + "moe.trunk_fc_out");
            }
            names.push_back(p + "moe.expert_fc_in");
            names.push_back(p + "moe.expert_fc_out");
            names.push_back(p + "moe.pressure_proj");
            names.push_back(p + "moe.pressure_mod");
            names.push_back(p + "moe.router_score");
            if (w.layers[l].pressure_to_routes_w != nullptr) {
                names.push_back(p + "moe.pressure_to_routes");
            }
        }
    }
    return names;
}

// ─── dataset loader ──────────────────────────────────────────────────────────

struct HostDataset {
    std::vector<uint32_t> tokens;
    std::vector<int32_t>  labels;
    std::vector<uint16_t> segs;   // per-position sample-start offset within its row
    int num_sequences{0};
    int seq_len{0};
};

static HostDataset load_host_dataset(const NativeRequest& req) {
    const auto& tp = req.input.token_blocks;
    const auto& lp = req.input.label_blocks;
    const auto& sp = req.input.seg_blocks;
    if (!std::filesystem::exists(tp))
        throw std::runtime_error("token_blocks not found: " + tp.string());
    if (!std::filesystem::exists(lp))
        throw std::runtime_error("label_blocks not found: " + lp.string());

    const std::size_t tb = std::filesystem::file_size(tp);
    const std::size_t lb = std::filesystem::file_size(lp);
    if (tb != lb) throw std::runtime_error("token/label block size mismatch");

    const int S = req.input.sequence_length > 0 ? req.input.sequence_length : 2048;
    const std::size_t total = tb / sizeof(uint32_t);
    HostDataset ds{};
    ds.seq_len       = S;
    ds.num_sequences = static_cast<int>(total / static_cast<std::size_t>(S));
    ds.tokens.resize(total);
    ds.labels.resize(total);
    {
        std::ifstream f(tp, std::ios::binary);
        f.read(reinterpret_cast<char*>(ds.tokens.data()), static_cast<std::streamsize>(tb));
        if (!f) throw std::runtime_error("read failed: " + tp.string());
    }
    {
        std::ifstream f(lp, std::ios::binary);
        f.read(reinterpret_cast<char*>(ds.labels.data()), static_cast<std::streamsize>(lb));
        if (!f) throw std::runtime_error("read failed: " + lp.string());
    }
    // Sample-boundary starts (optional).  Absent or size-mismatched → zeros,
    // i.e. one segment per row = the old full-causal behavior.
    ds.segs.assign(total, 0);
    if (!sp.empty() && std::filesystem::exists(sp)) {
        const std::size_t sb_bytes = std::filesystem::file_size(sp);
        if (sb_bytes == total * sizeof(uint16_t)) {
            std::ifstream f(sp, std::ios::binary);
            f.read(reinterpret_cast<char*>(ds.segs.data()), static_cast<std::streamsize>(sb_bytes));
            if (!f) throw std::runtime_error("read failed: " + sp.string());
        } else {
            std::fprintf(stderr,
                "[ida_native_train] seg_blocks size mismatch (%zu vs %zu) — "
                "falling back to full-causal attention\n",
                sb_bytes, total * sizeof(uint16_t));
        }
    }
    return ds;
}

// ─── cuBLAS GEMM wrappers (row-major convention) ─────────────────────────────
// Row-major A[M,K] @ B[K,N] = C[M,N];  lda/ldb/ldc are row strides.
static void gemm_bf16(
    cublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A, int lda,   // A[M,K]
    const __nv_bfloat16* B, int ldb,   // B[K,N]
    float beta,
    __nv_bfloat16* C, int ldc          // C[M,N]
) {
    ::ida_native::gemm_trace::record_current(static_cast<long long>(M), static_cast<long long>(N), static_cast<long long>(K));
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, ldb,
        A, CUDA_R_16BF, lda,
        &beta,
        C, CUDA_R_16BF, ldc,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// Row-major A[M,K] @ B^T = C[M,N] where B is stored [N,K].
static void gemm_bf16_nt(
    cublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A, int lda,   // A[M,K]
    const __nv_bfloat16* B, int ldb,   // B[N,K]
    float beta,
    __nv_bfloat16* C, int ldc          // C[M,N]
) {
    ::ida_native::gemm_trace::record_current(static_cast<long long>(M), static_cast<long long>(N), static_cast<long long>(K));
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, ldb,
        A, CUDA_R_16BF, lda,
        &beta,
        C, CUDA_R_16BF, ldc,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// Weight-gradient GEMM: C_f32[M,N] (β)= A^T @ B with BF16 activations.
// A row-major [K,M] (lda = row stride), B row-major [K,N] (ldb = row stride).
// β=1 accumulates into the persistent FP32 grad buffer (gradient accumulation).
static void gemm_bf16_tn_f32(
    cublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A, int lda,   // A[K,M]
    const __nv_bfloat16* B, int ldb,   // B[K,N]
    float beta,
    float* C, int ldc                  // C[M,N] FP32
) {
    ::ida_native::gemm_trace::record_current(static_cast<long long>(M), static_cast<long long>(N), static_cast<long long>(K));
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, ldb,
        A, CUDA_R_16BF, lda,
        &beta,
        C, CUDA_R_32F, ldc,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// Same weight-gradient contraction with BF16 durable storage.  cuBLAS still
// accumulates in FP32; only the persistent C buffer rounds to BF16, which is
// the memory-saving boundary for the explicit 1F1B experiment.
static void gemm_bf16_tn_f32(
    cublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A, int lda,
    const __nv_bfloat16* B, int ldb,
    float beta,
    __nv_bfloat16* C, int ldc
) {
    ::ida_native::gemm_trace::record_current(static_cast<long long>(M), static_cast<long long>(N), static_cast<long long>(K));
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, ldb,
        A, CUDA_R_16BF, lda,
        &beta,
        C, CUDA_R_16BF, ldc,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// ─── FP8 path (Hopper E4M3/E5M2 via cuBLASLt) ────────────────────────────────
// Per-tensor scaled FP8 for the projection GEMMs: forward X@Wᵀ and backward
// dY@Wᵀ run in FP8 tensor cores; weight-gradient GEMMs stay BF16→FP32.
// cuBLASLt's FP8 TN requirement maps exactly onto row-major A[M,K] @ B[N,K]ᵀ,
// so each weight is cached in both K-major layouts, requantized after every
// optimizer update.  Gate with IDA_NATIVE_FP8=0.

static constexpr float kE4M3Max = 448.0f;

// ── FP8 delayed-scaling saturation instrumentation (IDA_NATIVE_FP8_CLIP_DEBUG)
// Delayed scaling quantizes with the PREVIOUS call's scale, so a surge past
// that window clamps silently (__NV_SATFINITE: no inf/nan). Nothing in this
// engine measured that. One shared [3] device buffer accumulates
// {clipped_calls, max_overflow_ratio, total_calls} across every activation
// slot; the probe is a 1-thread kernel and only runs when the flag is set, so
// the default path is byte-identical to before.
static bool fp8_clip_debug_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("IDA_NATIVE_FP8_CLIP_DEBUG");
        return e && e[0] == '1';
    }();
    return v;
}
static float* g_fp8_clip_stats = nullptr;
static float* g_fp8_elem_stats = nullptr;
// Row-clip ladder instrumentation. 4 sites x 4 floats:
// [clipped_rows, total_rows, max_row_norm, ceiling_used].
// Three of these rungs were converted from mean-relative to ABSOLUTE
// ceilings on 2026-07-09 to kill the self-referential-outlier bug -- but an
// absolute ceiling cannot track legitimate activation growth, so the
// opposite failure (clamping healthy grown rows) became possible and was
// never measured. max_row_norm vs ceiling_used is what distinguishes them.
static float* g_rowclip_stats = nullptr;
static bool rowclip_debug_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("IDA_NATIVE_ROWCLIP_DEBUG");
        return e && e[0] == '1';
    }();
    return v;
}
static constexpr float kE5M2Max = 57344.0f;
static constexpr int kActHistoryLen = 8;

// Cached cuBLASLt plan: descriptors are created once per unique shape and
// reused; only the scale pointers change between calls (host-side attribute
// set, no create/destroy per GEMM).
struct LtPlan {
    cublasLtMatmulDesc_t   op{};
    cublasLtMatrixLayout_t la{}, lb{}, lc{};
    cublasLtMatmulHeuristicResult_t heuristic{};
    bool ready{false};
};
using LtPlanKey = std::array<long long, 6>;   // M, N, K, lda, ldd, a_type

struct Fp8Ctx;   // fwd decl; lt_gemm_fp8_nt uses the ctx's plan cache

enum LayerActSlotKind {
    ACT_QKV_IN = 0,
    ACT_O_IN,
    ACT_FFN_IN,
    ACT_DOWN_IN,
    ACT_DHIDDEN_DOWN,
    ACT_GATE_GRAD,
    ACT_UP_GRAD,
    ACT_DHIDDEN_O,
    ACT_QKV_GRAD,
    ACT_LAYER_SLOT_COUNT
};

enum GlobalActSlotKind {
    ACT_LMHEAD_IN = 0,
    ACT_LOGITS_GRAD,
    ACT_GLOBAL_SLOT_COUNT
};

// Per-tensor delayed-scaling state for one activation/gradient call site.
struct ActSlot {
    float* amax{};
    float* scale{};
    float* descale{};
    float* history{};
    int*   history_cursor{};
    int*   history_count{};
    // Snapshot of `scale`/`descale` taken right before the forward pass
    // advances them — the backward recompute reads these instead of
    // `scale`/`descale` so it quantizes (and dequantizes in the following
    // GEMM) with the exact same pair forward used (delayed-scaling mutates
    // both on every call, so without this the recompute call would see an
    // already-advanced scale and produce genuinely different, not just
    // rounding-noise-different, quantized values).
    float* scale_snapshot{};
    float* descale_snapshot{};
    float  fp8_max{kE4M3Max};
    bool   init{false};
    // Per-slot format calibration.  The first calib_remaining true-forward
    // calls quantize with an exact per-call amax (no stale-scale hazard) while
    // recording the amax spread; at the end the slot commits to E4M3
    // (precision-first, tight range) or E5M2 (range-first) based on how much
    // the surface's dynamic range moves between calls.  Mixed cognitive
    // surfaces routed through one slot are exactly the case a single delayed
    // scale cannot follow — the spread test detects them.  The commit is
    // deferred to the NEXT true-forward call so the final calibration micro's
    // backward recompute still quantizes with the format its forward used.
    int    calib_remaining{0};
    bool   calib_commit_pending{false};
    float* calib_min{};   // device scalar, running min of per-call amax
    float* calib_max{};   // device scalar, running max of per-call amax
};

// One FP8-cached weight tensor.  fwd8 = [out,in] (forward A@Bᵀ), bwd8 = [in,out]
// (backward dX A@Bᵀ).  Weights stored [in,out] transpose into fwd8; lm_head is
// stored [out,in] and transposes into bwd8.
struct Fp8Slot {
    const __nv_bfloat16* w{};
    void* fwd8{};
    void* bwd8{};
    float* amax{};
    float* scale{};
    float* descale{};
    int in_dim{0};
    int out_dim{0};
    bool stored_out_major{false};
};

struct Fp8Ctx {
    bool on{false};
    cublasLtHandle_t lt{};
    bool heuristics_cache_configured{false};
    void* ws{};
    std::size_t ws_size{32u << 20};
    // Local layer slots are [l*7 + {q,k,v,o,gate,up,down}]. The LM-head
    // cache exists only on the output-owning pipeline stage.
    std::vector<Fp8Slot> slots;
    int lm_head_weight_slot{-1};
    float* scalars{};              // backing store for weight amax/scale/descale
    std::vector<ActSlot> layer_act_slots;
    std::vector<ActSlot> global_act_slots;
    // Cognitive-architecture sparse MoE port (Step 4b): expert-bank FP8.
    // Weight slots for trunk_fc_in/out (Mode B only) + expert_fc_in/out are
    // appended to `slots` above (refresh_fp8_weights already loops that
    // whole vector generically, no separate refresh path needed) --
    // moe_weight_base[l] is the index into `slots` where layer l's block
    // starts, -1 if that layer has no MoE. Router projections
    // (pressure_proj/pressure_mod/router_score/pressure_to_routes) are
    // deliberately NOT quantized -- they're tiny (out_dim <= ~11) custom
    // GEMV kernels by design (see moe_small_proj_f32), never cuBLASLt GEMMs
    // to begin with; forcing them through the FP8 tensor-core path would
    // add real complexity for matrices too small to benefit.
    // NOTE: these appended weight slots are not yet wired into
    // bind_fp8_weight_amax_slots' optimizer-fused amax path -- MoE weight
    // FP8 requantization across multiple optimizer steps is unverified
    // (Step 5/6 follow-up); this pass only proves single-forward-pass
    // correctness (matching Step 4's own "forward-only" scope), which
    // build_fp8_ctx's own fresh recompute_amax=true call already covers
    // for the initial quantization used here.
    std::vector<int> moe_weight_base;  // [L], -1 if no MoE that layer
    // Activation slots: [0] = shared fc_in-input (hidden_gated -- ONE
    // quantization reused for the trunk's AND every expert's fc_in GEMM,
    // exactly like ACT_FFN_IN is shared by gate_proj/up_proj today, since
    // they all read the identical input tensor); [1..] = one per (trunk if
    // present, then each expert) for the post-GELU fc_out input, which IS
    // genuinely distinct per expert.
    std::vector<ActSlot> moe_act_slots;
    std::vector<int> moe_act_base;     // [L], -1 if no MoE that layer
    std::map<LtPlanKey, LtPlan> plans;
    std::map<LtPlanKey, LtPlan> plans_f32;  // TN F32-output dW plan cache
    std::map<LtPlanKey, LtPlan> plans_bf16; // TN BF16-output dW plan cache
    // activation quantization scratch (one tensor at a time, stream-ordered)
    void* act8{};
    // FP8 E4M3 activation snapshots for dW GEMMs, stored TRANSPOSED [dim, BS].
    // cuBLASLt FP8 only supports the TN form (A=OP_T, B=OP_N); landing dW
    // row-major requires both operands pre-transposed in memory (same approach
    // as TE's fp8 transpose cache).
    void* snap_qkv{};    // [H, max_bs] E4M3 — normed^T from layer recompute
    void* snap_o{};      // [H, max_bs] E4M3 — attn_out^T from layer recompute
    void* snap_ffn{};    // [H, max_bs] E4M3 — normed2^T from layer recompute
    void* snap_down{};   // [I, max_bs] E4M3 — swiglu_out^T from layer recompute
    // Transposed E5M2 gradient scratch [out_dim, BS] for the dW GEMM A operand.
    void* grad8_t{};     // [max(3H, I), max_bs]
};

// ─── Ampere FP8-packed weight cache ─────────────────────────────────────────
// Ampere sm_86 (RTX 3050/3090) has no validated native FP8 GEMM path in this trainer.
// These entries are therefore a storage cache only: BF16 master weights are
// packed to raw E4M3 bytes with one scale per tensor while the same pass fills
// reusable BF16 staging buffers. Existing cublasGemmEx calls consume those
// staging buffers and keep FP32 accumulation unchanged.
struct AmperePackedWeight {
    const __nv_bfloat16* master{};
    std::uint8_t* packed{};
    __nv_bfloat16* dequant{};
    float* amax{};
    float* scale{};
    float* descale{};
    int* bad_count{};
    std::size_t n{0};
};

struct AmperePackedWeights {
    bool on{false};
    std::vector<AmperePackedWeight> entries;
};

thread_local const AmperePackedWeights* g_ampere_packed_weights = nullptr;

static bool precision_profile_uses_ampere_packed(const NativeRequest& request) {
    return request.precision_profile == "ampere_fp8_packed";
}

static const __nv_bfloat16* ampere_compute_weight(const __nv_bfloat16* master) {
    if (!g_ampere_packed_weights || !g_ampere_packed_weights->on) return master;
    for (const auto& entry : g_ampere_packed_weights->entries) {
        if (entry.master == master) return entry.dequant;
    }
    return master;
}

static void refresh_ampere_packed_weights(AmperePackedWeights& cache, cudaStream_t s) {
    if (!cache.on) return;
    std::vector<int> bad(cache.entries.size(), 0);
    for (auto& entry : cache.entries) {
        fp8_pack_bf16_e4m3_raw_and_dequant(
            entry.master, entry.packed, entry.dequant, entry.amax,
            entry.scale, entry.descale, entry.bad_count, entry.n, s);
        IDA_CUDA_CHECK(cudaMemcpyAsync(
            &bad[&entry - cache.entries.data()], entry.bad_count, sizeof(int),
            cudaMemcpyDeviceToHost, s));
    }
    IDA_CUDA_CHECK(cudaStreamSynchronize(s));
    for (std::size_t i = 0; i < bad.size(); ++i) {
        if (bad[i] != 0) {
            throw std::runtime_error(
                "ampere_fp8_packed rejected nonfinite BF16 master weight at cache entry " +
                std::to_string(i));
        }
    }
}

static AmperePackedWeights build_ampere_packed_weights(
    const NativeRequest& request, const LatticeWeights& w, NativeArena& arena
) {
    AmperePackedWeights cache{};
    cache.on = precision_profile_uses_ampere_packed(request);
    if (!cache.on) return cache;
    const int H = w.hidden_size;
    const int I = w.intermediate_size;
    const int KV = (w.kv_heads > 0 ? w.kv_heads : w.heads) * (H / w.heads);
    auto add = [&](const __nv_bfloat16* master, std::size_t n) {
        AmperePackedWeight entry{};
        entry.master = master;
        entry.n = n;
        entry.packed = alloc_u8(n, arena);
        entry.dequant = alloc_bf16(n, arena);
        entry.amax = alloc_f32(1, arena);
        entry.scale = alloc_f32(1, arena);
        entry.descale = alloc_f32(1, arena);
        entry.bad_count = alloc_i32(1, arena);
        cache.entries.push_back(entry);
    };
    for (int l = 0; l < w.num_layers; ++l) {
        const auto& lw = w.layers[l];
        add(lw.q_proj, static_cast<std::size_t>(H) * H);
        add(lw.k_proj, static_cast<std::size_t>(H) * KV);
        add(lw.v_proj, static_cast<std::size_t>(H) * KV);
        add(lw.o_proj, static_cast<std::size_t>(H) * H);
        add(lw.gate_proj, static_cast<std::size_t>(H) * I);
        add(lw.up_proj, static_cast<std::size_t>(H) * I);
        add(lw.down_proj, static_cast<std::size_t>(I) * H);
        // Routed bodies keep the same BF16 master weights and use the same
        // storage-only cache for their additional linear projections.  The
        // expert bank is one contiguous tensor per weight type, so a single
        // per-tensor E4M3 scale is deterministic and matches the durable
        // ParamSlot layout.
        if (lw.num_experts > 0 && lw.moe_kind == 0) {
            if (lw.moe_shared_trunk) {
                add(lw.trunk_fc_in_w, static_cast<std::size_t>(I) * H);
                add(lw.trunk_fc_out_w, static_cast<std::size_t>(H) * I);
            }
            const std::size_t expert_i =
                static_cast<std::size_t>(lw.num_experts) * lw.expert_intermediate_size * H;
            const std::size_t expert_o =
                static_cast<std::size_t>(lw.num_experts) * H * lw.expert_intermediate_size;
            add(lw.expert_fc_in_w, expert_i);
            add(lw.expert_fc_out_w, expert_o);
            add(lw.pressure_proj_w, static_cast<std::size_t>(lw.num_routes) * H);
            add(lw.pressure_mod_w, static_cast<std::size_t>(H) * lw.num_routes);
            add(lw.router_score_w, static_cast<std::size_t>(lw.num_experts) * H);
            if (lw.pressure_to_routes_w != nullptr) {
                add(lw.pressure_to_routes_w,
                    static_cast<std::size_t>(lw.num_experts) * lw.num_routes);
            }
        }
    }
    if (w.owns_output) add(w.lm_head, static_cast<std::size_t>(w.vocab_size) * H);
    refresh_ampere_packed_weights(cache, arena.stream);
    return cache;
}

static void free_ampere_packed_weights(AmperePackedWeights& cache, NativeArena& arena) {
    if (!cache.on) return;
    for (auto& entry : cache.entries) {
        if (entry.packed) IDA_CUDA_CHECK(cudaFreeAsync(entry.packed, arena.stream));
        if (entry.dequant) IDA_CUDA_CHECK(cudaFreeAsync(entry.dequant, arena.stream));
        if (entry.amax) IDA_CUDA_CHECK(cudaFreeAsync(entry.amax, arena.stream));
        if (entry.scale) IDA_CUDA_CHECK(cudaFreeAsync(entry.scale, arena.stream));
        if (entry.descale) IDA_CUDA_CHECK(cudaFreeAsync(entry.descale, arena.stream));
        if (entry.bad_count) IDA_CUDA_CHECK(cudaFreeAsync(entry.bad_count, arena.stream));
    }
    cache.entries.clear();
    cache.on = false;
}

struct StepBuffers;

// q_amax/q_scale/q_descale/k_amax/k_scale/k_descale are PER-LAYER (arrays of
// num_layers floats, indexed by layer_idx) — the delayed-scaling history for
// layer N must stay layer N's own, independent of every other layer's. An
// earlier version shared one scale slot across all layers: forward visits
// 0..L-1 in order, backward's recompute visits L-1..0 in reverse, so a
// shared slot's "previous value" meant a different layer's leftover state
// depending on sweep direction — no snapshot trick can fix a fundamentally
// shared resource like that. The big scratch buffers below (qk_packed,
// q_unpack_f32, etc.) are still fine shared: each layer's pack+attend
// completes before the next layer starts, same stream, no cross-layer
// state carried in them.
struct PackedFp4AttentionCtx {
    bool on{false};
    std::uint8_t* qk_packed{};
    std::uint8_t* q_saved_e4m3{};  // [num_layers * packed_elems] optional ablation cache
    std::uint8_t* k_saved_e4m3{};  // [num_layers * packed_elems] optional ablation cache
    int num_layers{0};
    float* q_amax{};    // [num_layers]
    float* q_scale{};   // [num_layers]
    float* q_descale{}; // [num_layers]
    float* k_amax{};    // [num_layers]
    float* k_scale{};   // [num_layers]
    float* k_descale{}; // [num_layers]
    // Mean-centered (IDA_NATIVE_FP4_CENTERED) delayed-scaling state: q_mean/
    // k_mean are the persistent per-layer origin (like q_scale/k_scale); the
    // sum/max_abs_dev pairs are per-call scratch that this call's record
    // fills in, which the NEXT call's fp4_scale_from_stats_centered turns
    // into the following mean/scale — same delayed-scaling shape as amax.
    float* q_mean{};         // [num_layers]
    float* k_mean{};         // [num_layers]
    float* q_sum{};          // [num_layers] scratch
    float* k_sum{};          // [num_layers] scratch
    float* q_max_abs_dev{};  // [num_layers] scratch
    float* k_max_abs_dev{};  // [num_layers] scratch
    // IDA_NATIVE_FP4_CENTERED_FIX=1 only: frozen copies of mean/scale/descale
    // taken on the true-forward call, before that call's own record/update
    // advances the live state above. The backward recompute (and the
    // decode step that follows any pack, forward or recompute) reads these
    // instead of the live arrays, so both calls encode/decode against the
    // identical values the true forward actually used — same shape as
    // ActSlot's scale_snapshot/descale_snapshot fix for FP8, applied here to
    // the mean too since a mismatched mean is a much bigger absolute error.
    float* q_mean_snapshot{};     // [num_layers]
    float* k_mean_snapshot{};     // [num_layers]
    float* q_scale_snapshot{};    // [num_layers]
    float* k_scale_snapshot{};    // [num_layers]
    float* q_descale_snapshot{};  // [num_layers]
    float* k_descale_snapshot{};  // [num_layers]
    float* q_unpack_f32{};
    float* k_unpack_f32{};
    float* q_tile_stage_f32{};
    float* k_tile_stage_f32{};
    float* v_tile_stage_f32{};
    std::size_t tile_stage_elems{0};
    int tma_stage_depth{0};
    std::size_t packed_elems{0};
};

// Forward-declared here (defined near prepare_packed_fp4_attention_forward)
// so packed_fp4_operands, defined earlier in the file, can route to the
// snapshot arrays whenever the packed-FP4 delayed-state timing fix is active.
static bool fp4_centered_enabled();
static bool fp4_centered_int2_enabled();
static bool fp4_delayed_state_fix_enabled();
static bool packed_fp4_backward_replay_saved_e4m3_enabled();

static PackedFp4AttentionOperands packed_fp4_operands(
    const PackedFp4AttentionCtx& ctx, int layer_idx
) {
    PackedFp4AttentionOperands operands{};
    operands.qk_packed = ctx.qk_packed;
    if (ctx.q_saved_e4m3 && ctx.k_saved_e4m3) {
        const std::size_t off = static_cast<std::size_t>(layer_idx) * ctx.packed_elems;
        operands.q_saved_e4m3 = ctx.q_saved_e4m3 + off;
        operands.k_saved_e4m3 = ctx.k_saved_e4m3 + off;
    }
    if (fp4_delayed_state_fix_enabled()) {
        // The live arrays hold NEXT step's values by the time any decode
        // reads them (record+update already ran inside prepare_packed_fp4_
        // attention_forward); the snapshot arrays hold what was actually
        // used to pack ctx.qk_packed this call.
        operands.q_scale = ctx.q_scale_snapshot + layer_idx;
        operands.q_descale = ctx.q_descale_snapshot + layer_idx;
        operands.k_scale = ctx.k_scale_snapshot + layer_idx;
        operands.k_descale = ctx.k_descale_snapshot + layer_idx;
        operands.q_mean = ctx.q_mean_snapshot + layer_idx;
        operands.k_mean = ctx.k_mean_snapshot + layer_idx;
    } else {
        operands.q_scale = ctx.q_scale + layer_idx;
        operands.q_descale = ctx.q_descale + layer_idx;
        operands.k_scale = ctx.k_scale + layer_idx;
        operands.k_descale = ctx.k_descale + layer_idx;
        operands.q_mean = ctx.q_mean + layer_idx;
        operands.k_mean = ctx.k_mean + layer_idx;
    }
    operands.q_unpack_f32 = ctx.q_unpack_f32;
    operands.k_unpack_f32 = ctx.k_unpack_f32;
    operands.q_tile_stage_f32 = ctx.q_tile_stage_f32;
    operands.k_tile_stage_f32 = ctx.k_tile_stage_f32;
    operands.v_tile_stage_f32 = ctx.v_tile_stage_f32;
    operands.tile_stage_elems = ctx.tile_stage_elems;
    operands.tma_stage_depth = ctx.tma_stage_depth;
    operands.packed_elems = ctx.packed_elems;
    return operands;
}

static bool precision_profile_uses_fp8(const NativeRequest& request) {
    // FP8 cuBLASLt projection GEMMs (Q/K/V/O/FFN).
    // hopper_fp8_packed_fp4: AI-family combined profile — FP8 linear projections
    // + packed-FP4 WGMMA attention in one pass.
    return request.precision_profile == "legacy_fp8"
        || request.precision_profile == "hopper_fp8_packed_fp4";
}

static bool precision_profile_uses_packed_fp4(const NativeRequest& request) {
    return request.precision_profile == "hopper_bf16_packed_fp4"
        || request.precision_profile == "hopper_fp8_packed_fp4";
}

static bool precision_profile_uses_wgmma_fp8(const NativeRequest& request) {
    return request.precision_profile == "hopper_bf16_fp8";
}

// Both Hopper WGMMA attention backends reuse the same q/k scratch shape
// (byte-carved e4m3 storage sized to the f32-equivalent element count), so
// they share one allocation gate here.
static bool precision_profile_uses_wgmma_attention_scratch(const NativeRequest& request) {
    return precision_profile_uses_packed_fp4(request) || precision_profile_uses_wgmma_fp8(request);
}

static const char* runtime_precision_label(const NativeRequest& request) {
    if (request.precision_profile == "hopper_bf16_packed_fp4") {
        return "hopper_bf16_packed_fp4";
    }
    if (request.precision_profile == "hopper_bf16_fp8") {
        return "hopper_bf16_fp8";
    }
    if (request.precision_profile == "legacy_bf16") {
        return "legacy_bf16";
    }
    if (request.precision_profile == "ampere_fp8_packed") {
        return "ampere_fp8_packed";
    }
    return "legacy_fp8";
}

static void validate_precision_state_request(const NativeRequest& request) {
    const auto require_supported = [](
        const std::string& value,
        const char* field,
        std::initializer_list<const char*> supported
    ) {
        for (const char* candidate : supported) {
            if (value == candidate) {
                return;
            }
        }
        std::string supported_values;
        bool first = true;
        for (const char* candidate : supported) {
            if (!first) {
                supported_values += ", ";
            }
            supported_values += candidate;
            first = false;
        }
        throw std::runtime_error(
            std::string("native precision policy ") + field + "=" + value +
            " is unsupported; supported values: " + supported_values
        );
    };
    require_supported(
        request.optimizer_state_precision, "optimizer_state_precision", {"fp32", "bf16"}
    );
    require_supported(
        request.optimizer_type, "optimizer_type", {"lion"}
    );
    require_supported(
        request.gradient_buffer_precision, "gradient_buffer_precision", {"fp32", "bf16"}
    );
    require_supported(
        request.gemm_accumulator_precision, "gemm_accumulator_precision", {"fp32"}
    );
}

static ActSlot& layer_act_slot(Fp8Ctx& f, int layer_idx, LayerActSlotKind kind) {
    return f.layer_act_slots[static_cast<std::size_t>(layer_idx) * ACT_LAYER_SLOT_COUNT + kind];
}

// Weight slot accessors mirroring build_fp8_ctx's push order for MoE:
// shared_trunk ? [trunk_in, trunk_out, expert_in, expert_out] : [expert_in, expert_out].
static Fp8Slot& moe_trunk_in_slot(Fp8Ctx& f, int layer_idx) {
    return f.slots[static_cast<std::size_t>(f.moe_weight_base[static_cast<std::size_t>(layer_idx)])];
}
static Fp8Slot& moe_trunk_out_slot(Fp8Ctx& f, int layer_idx) {
    return f.slots[static_cast<std::size_t>(f.moe_weight_base[static_cast<std::size_t>(layer_idx)]) + 1];
}
static Fp8Slot& moe_expert_in_slot(Fp8Ctx& f, int layer_idx, bool shared_trunk) {
    return f.slots[static_cast<std::size_t>(f.moe_weight_base[static_cast<std::size_t>(layer_idx)]) + (shared_trunk ? 2 : 0)];
}
static Fp8Slot& moe_expert_out_slot(Fp8Ctx& f, int layer_idx, bool shared_trunk) {
    return f.slots[static_cast<std::size_t>(f.moe_weight_base[static_cast<std::size_t>(layer_idx)]) + (shared_trunk ? 3 : 1)];
}

static ActSlot& moe_fcin_act_slot(Fp8Ctx& f, int layer_idx) {
    return f.moe_act_slots[static_cast<std::size_t>(f.moe_act_base[static_cast<std::size_t>(layer_idx)])];
}

// which: 0 = trunk (only valid when the layer has moe_shared_trunk), else
// 1-based expert index (expert e -> which = e + 1, or e + 2 if shared_trunk
// also occupies slot 1 -- callers pass the already-offset `which` computed
// from moe_shared_trunk, see moe_expert_bank_forward).
static ActSlot& moe_fcout_act_slot(Fp8Ctx& f, int layer_idx, int which) {
    return f.moe_act_slots[static_cast<std::size_t>(f.moe_act_base[static_cast<std::size_t>(layer_idx)] + 1 + which)];
}

static ActSlot& global_act_slot(Fp8Ctx& f, GlobalActSlotKind kind) {
    return f.global_act_slots[static_cast<std::size_t>(kind)];
}

static PackedFp4AttentionCtx build_packed_fp4_attention_ctx(
    const NativeRequest& request,
    const StepBuffers& sb,
    int num_layers,
    NativeArena& arena
);

static void free_packed_fp4_attention_ctx(
    PackedFp4AttentionCtx& ctx,
    NativeArena& arena
) {
    auto f = [&](auto* p) { if (p) IDA_CUDA_CHECK(cudaFreeAsync(p, arena.stream)); };
    f(ctx.qk_packed);
    f(ctx.q_saved_e4m3);
    f(ctx.k_saved_e4m3);
    f(ctx.q_amax);
    f(ctx.q_scale);
    f(ctx.q_descale);
    f(ctx.k_amax);
    f(ctx.k_scale);
    f(ctx.k_descale);
    f(ctx.q_mean);
    f(ctx.k_mean);
    f(ctx.q_sum);
    f(ctx.k_sum);
    f(ctx.q_max_abs_dev);
    f(ctx.k_max_abs_dev);
    f(ctx.q_mean_snapshot);
    f(ctx.k_mean_snapshot);
    f(ctx.q_scale_snapshot);
    f(ctx.k_scale_snapshot);
    f(ctx.q_descale_snapshot);
    f(ctx.k_descale_snapshot);
    f(ctx.q_unpack_f32);
    f(ctx.k_unpack_f32);
    f(ctx.q_tile_stage_f32);
    f(ctx.k_tile_stage_f32);
    f(ctx.v_tile_stage_f32);
    ctx = PackedFp4AttentionCtx{};
}

static void prepare_packed_fp4_attention_forward(
    PackedFp4AttentionCtx& ctx,
    const StepBuffers& sb,
    int layer_idx,
    cudaStream_t s,
    bool is_recompute = false
);

static void destroy_lt_plan(LtPlan& plan) {
    if (plan.la) cublasLtMatrixLayoutDestroy(plan.la);
    if (plan.lb) cublasLtMatrixLayoutDestroy(plan.lb);
    if (plan.lc) cublasLtMatrixLayoutDestroy(plan.lc);
    if (plan.op) cublasLtMatmulDescDestroy(plan.op);
    plan = LtPlan{};
}

// Number of exact-amax calibration calls each activation slot runs before
// committing to a per-slot FP8 format.  0 disables (old fixed-type behavior).
static int fp8_calib_calls() {
    static const int v = [] {
        const char* e = std::getenv("IDA_FP8_CALIB_CALLS");
        if (!e || !e[0]) return 8;
        char* end = nullptr;
        const long parsed = std::strtol(e, &end, 10);
        return (end == e || parsed < 0) ? 8 : static_cast<int>(parsed);
    }();
    return v;
}

// Amax spread (max/min across the calibration window) above which a slot's
// dynamic range is judged too mobile for E4M3 and flips to E5M2.
static float fp8_type_spread_threshold() {
    static const float v = [] {
        const char* e = std::getenv("IDA_FP8_TYPE_SPREAD");
        if (!e || !e[0]) return 4.0f;
        char* end = nullptr;
        const float parsed = std::strtof(e, &end);
        return (end == e || parsed <= 1.0f) ? 4.0f : parsed;
    }();
    return v;
}

__global__ void k_calib_minmax_update(
    const float* __restrict__ amax, float* __restrict__ cmin, float* __restrict__ cmax
) {
    const float a = fmaxf(*amax, 1e-12f);
    *cmin = fminf(*cmin, a);
    *cmax = fmaxf(*cmax, a);
}

// ─── activation row clip (the semantic clip's activation-stream sibling) ─────
// High-frequency low-semantic tokens (punctuation, delimiters) become
// attention sinks with massive per-token activation rows in the UNNORMALIZED
// surfaces (attn_out, swiglu_out).  One outlier row sets the per-tensor amax,
// the shared FP8 scale collapses every other row toward zero, and the dW that
// consumes the tensor blows up (observed: L*.o_proj = inf on the AI body).
// The post-RMSNorm surfaces don't need this — the norm equalizes their rows.
// Clip token rows whose L2 norm exceeds mult × mean row norm — data-adaptive,
// so healthy rows are untouched at any width.
static float act_row_clip_mult(const NativeRequest& request) {
    // request.act_row_clip_override (per-burn, wrapper-supplied) wins over
    // the cached env var -- see request.hpp for why: a shared multi-tenant
    // process can't represent a per-family ceiling via its own fixed env,
    // same fix already applied to global_clip_override/lion_*_override.
    if (request.act_row_clip_override >= 0.0f) return request.act_row_clip_override;
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_ACT_ROW_CLIP");
        if (!e || !e[0]) return 1000.0f;
        char* end = nullptr;
        const float parsed = std::strtof(e, &end);
        return (end == e) ? 1000.0f : parsed;
    }();
    return v;
}

__global__ void k_act_row_normsq(
    const __nv_bfloat16* __restrict__ x, int cols, int row_stride,
    float* __restrict__ row_normsq, float* __restrict__ total
) {
    __shared__ float sm[256 / 32];
    const std::size_t row = blockIdx.x;
    const __nv_bfloat16* xr = x + row * row_stride;
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = __bfloat162float(xr[c]);
        s += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int w = 0; w < blockDim.x / 32; ++w) t += sm[w];
        row_normsq[row] = t;
        if (total) atomicAdd(total, sqrtf(t));
    }
}

__device__ float* g_rowclip_dev_stats = nullptr;

__global__ void k_act_row_clip_apply(
    __nv_bfloat16* __restrict__ x, int cols, int row_stride,
    const float* __restrict__ row_normsq, const float* __restrict__ total,
    int rows, float mult
) {
    const std::size_t row = blockIdx.x;
    const float mean_norm = *total / static_cast<float>(rows);
    const float ceiling = mult * fmaxf(mean_norm, 1e-6f);
    const float norm = sqrtf(row_normsq[row]);
    if (g_rowclip_dev_stats != nullptr && threadIdx.x == 0) {
        atomicAdd(&g_rowclip_dev_stats[13], 1.0f);
        atomicMax(reinterpret_cast<int*>(&g_rowclip_dev_stats[14]), __float_as_int(norm));
        g_rowclip_dev_stats[15] = ceiling;
        if (norm > ceiling) atomicAdd(&g_rowclip_dev_stats[12], 1.0f);
    }
    if (norm <= ceiling) return;
    const float scalef = ceiling / norm;
    __nv_bfloat16* xr = x + row * row_stride;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        xr[c] = __float2bfloat16(__bfloat162float(xr[c]) * scalef);
}

// Absolute-ceiling variant: mean-relative clipping (above) is
// self-referential when the outlier itself dominates the mean it's being
// compared against (one row at 1e12 among 262144 rows still pulls the
// mean to ~3.8e6, so an 8x-mean ceiling sits at ~3e7 — astronomically
// loose).  A fixed, data-INdependent ceiling has no such failure mode.
static int act_row_clip_mode() {   // 0 = fixed (legacy), 1 = EMA-adaptive
    static const int v = [] {
        const char* e = std::getenv("IDA_NATIVE_ACT_ROW_CLIP_MODE");
        return (e && e[0] == '1') ? 1 : 0;
    }();
    return v;
}
static float act_row_clip_ema_mult() {
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_ACT_ROW_CLIP_EMA_MULT");
        if (!e || !e[0]) return 16.0f;
        char* end = nullptr; const float p = std::strtof(e, &end);
        return (end == e || p <= 0.0f) ? 16.0f : p;
    }();
    return v;
}
static float act_row_clip_ema_beta() {
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_ACT_ROW_CLIP_EMA_BETA");
        if (!e || !e[0]) return 0.99f;
        char* end = nullptr; const float p = std::strtof(e, &end);
        return (end == e || p <= 0.0f || p >= 1.0f) ? 0.99f : p;
    }();
    return v;
}

// EMA state, one slot per call site (see kActRowSite*). Persistent across the
// whole burn -- that persistence is the point: it is what makes a one-step
// outlier unable to move the ceiling.
static float* g_act_row_ema = nullptr;

// ceiling = mult * ema(mean_row_norm). Single thread: updates the EMA from
// this call's mean, then writes the ceiling the apply kernel will read.
__global__ void k_act_row_ema_ceiling(
    const float* __restrict__ total, int rows, float* __restrict__ ema,
    float beta, float mult, float* __restrict__ out_ceiling
) {
    const float mean = *total / fmaxf(static_cast<float>(rows), 1.0f);
    float e = *ema;
    e = (e <= 0.0f) ? mean : (beta * e + (1.0f - beta) * mean);
    *ema = e;
    *out_ceiling = mult * fmaxf(e, 1e-6f);
}

// Absolute-ceiling apply, ceiling supplied by device pointer (EMA mode).
__global__ void k_act_row_clip_apply_dev(
    __nv_bfloat16* __restrict__ x, int cols, int row_stride,
    const float* __restrict__ row_normsq, const float* __restrict__ ceiling_p,
    float* __restrict__ stats
) {
    const std::size_t row = blockIdx.x;
    const float ceiling = *ceiling_p;
    const float norm = sqrtf(row_normsq[row]);
    if (stats != nullptr && threadIdx.x == 0) {
        atomicAdd(&stats[1], 1.0f);
        atomicMax(reinterpret_cast<int*>(&stats[2]), __float_as_int(norm));
        stats[3] = ceiling;
        if (norm > ceiling) atomicAdd(&stats[0], 1.0f);
    }
    if (norm <= ceiling) return;
    const float scalef = ceiling / norm;
    __nv_bfloat16* xr = x + row * row_stride;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        xr[c] = __float2bfloat16(__bfloat162float(xr[c]) * scalef);
}

__global__ void k_act_row_clip_apply_abs(
    __nv_bfloat16* __restrict__ x, int cols, int row_stride,
    const float* __restrict__ row_normsq,
    float ceiling, float* __restrict__ stats
) {
    const std::size_t row = blockIdx.x;
    const float norm = sqrtf(row_normsq[row]);
    if (stats != nullptr && threadIdx.x == 0) {
        atomicAdd(&stats[1], 1.0f);
        atomicMax(reinterpret_cast<int*>(&stats[2]), __float_as_int(norm));
        stats[3] = ceiling;
        if (norm > ceiling) atomicAdd(&stats[0], 1.0f);
    }
    if (norm <= ceiling) return;
    const float scalef = ceiling / norm;
    __nv_bfloat16* xr = x + row * row_stride;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        xr[c] = __float2bfloat16(__bfloat162float(xr[c]) * scalef);
}

// In-place row clip over x [rows, cols].  normsq_scratch holds rows+1 floats.
// Converted to an absolute ceiling 2026-07-09 (was: mult * mean_row_norm,
// self-referential once an outlier row dominates the mean it's compared
// against — same flaw diagnosed and fixed in dqkv_row_clip/interlayer_clip
// the same day).  IDA_NATIVE_ACT_ROW_CLIP now reads as a fixed ceiling,
// not a multiplier — default kept generous (1000) to match the sibling
// clips; existing deployments overriding this env var for the OLD
// multiplier semantics will need to switch to an absolute value.
static void act_row_clip(
    const NativeRequest& request,
    __nv_bfloat16* x, int rows, int cols,
    float* normsq_scratch, cudaStream_t s
) {
    const float ceiling = act_row_clip_mult(request);
    if (ceiling <= 0.0f && act_row_clip_mode() == 0) return;
    if (act_row_clip_mode() == 1 && g_act_row_ema != nullptr) {
        // EMA mode needs the row-norm SUM, so pass the accumulator through.
        IDA_CUDA_CHECK(cudaMemsetAsync(normsq_scratch + rows, 0, sizeof(float), s));
        k_act_row_normsq<<<rows, 256, 0, s>>>(x, cols, cols, normsq_scratch,
                                              normsq_scratch + rows);
        k_act_row_ema_ceiling<<<1, 1, 0, s>>>(
            normsq_scratch + rows, rows, g_act_row_ema,
            act_row_clip_ema_beta(), act_row_clip_ema_mult(), g_act_row_ema + 1);
        k_act_row_clip_apply_dev<<<rows, 256, 0, s>>>(
            x, cols, cols, normsq_scratch, g_act_row_ema + 1,
            g_rowclip_stats ? g_rowclip_stats + 0 : nullptr);
        return;
    }
    k_act_row_normsq<<<rows, 256, 0, s>>>(x, cols, cols, normsq_scratch, nullptr);
    k_act_row_clip_apply_abs<<<rows, 256, 0, s>>>(x, cols, cols, normsq_scratch, ceiling,
        g_rowclip_stats ? g_rowclip_stats + 0 : nullptr);
}

// Inter-layer backward-sweep clip — same mechanism as act_row_clip, own
// ablation knob so it can be tuned/disabled independently.  See the call
// site's comment (layer_backward_body, right after the residual fork) for
// the mechanism this targets: gradient-magnitude compounding through
// depth in the backward sweep, root-caused 2026-07-09.
// FIXED absolute ceiling, not a mean-relative multiplier — see
// k_act_row_clip_apply_abs's comment (dqkv_row_clip's mate) for why
// relative clipping is self-referentially broken once the outlier itself
// dominates the mean it's compared against.  d_hidden is read by O-proj's
// gradient GEMM at the TOP of each layer's attention backward, before
// qk_row_clip/dqkv_row_clip get any chance to run — found 2026-07-09 when
// the mean-relative version fixed q/k/v but left L0.o_proj as the next
// dominant slot (4.8e11), a whack-a-mole signature of the same
// self-referential-ceiling bug, not a new mechanism.
static float interlayer_clip_ceiling() {
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_INTERLAYER_CLIP");
        if (!e || !e[0]) return 1000.0f;
        char* end = nullptr;
        const float parsed = std::strtof(e, &end);
        return (end == e) ? 1000.0f : parsed;
    }();
    return v;
}

static void interlayer_clip(
    __nv_bfloat16* x, int rows, int cols,
    float* normsq_scratch, cudaStream_t s
) {
    const float ceiling = interlayer_clip_ceiling();
    if (ceiling <= 0.0f) return;
    k_act_row_normsq<<<rows, 256, 0, s>>>(x, cols, cols, normsq_scratch, nullptr);
    k_act_row_clip_apply_abs<<<rows, 256, 0, s>>>(x, cols, cols, normsq_scratch, ceiling,
        g_rowclip_stats ? g_rowclip_stats + 4 : nullptr);
}

// Q/K-specific row clip: same mechanism, applied BEFORE packed-FP4 Q/K
// quantization instead of after attention.  Q/K live interleaved in
// sb.qkv [BS, 3H] (row_stride=3H; Q at col-offset 0, K at col-offset H) —
// the strided kernels above address that view directly via x=(qkv+offset),
// row_stride=3H, cols=H.
//
// Why: prepare_packed_fp4_attention_forward's delayed-scale calibration
// (k_fp4_pack_pair_gauss_record et al.) reads a scale calibrated from the
// PREVIOUS call's amax while recording THIS call's amax as a side effect
// for the call after — a one-call lag.  Extreme-value statistics say the
// max of the microbatch's token population grows with population size, so
// a larger mb raises the odds that this call's true max exceeds what the
// prior (smaller-draw) calibration anticipated, independent of any timing
// bug.  L0 is uniquely exposed: it consumes the RAW embedding lookup
// before any residual-stream averaging has regularized the distribution,
// same root cause the embed-table sink-token clip already targets on the
// gradient side — this is the forward-path analog, backing the Q/K
// projection specifically, and (being a fresh-per-call adaptive ceiling,
// not a persistent/delayed state) it scales its own protection with
// whatever population size mb draws each call, with no separate state to
// go stale across recompute.
static float qk_row_clip_mult() {
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_QK_ROW_CLIP");
        if (!e || !e[0]) return 8.0f;
        char* end = nullptr;
        const float parsed = std::strtof(e, &end);
        return (end == e) ? 8.0f : parsed;
    }();
    return v;
}

static void qk_row_clip(
    __nv_bfloat16* qkv, int rows, int H,
    float* normsq_scratch, cudaStream_t s
) {
    const float mult = qk_row_clip_mult();
    if (mult <= 0.0f) return;
    const int row_stride = 3 * H;
    float* total = normsq_scratch + rows;
    IDA_CUDA_CHECK(cudaMemsetAsync(total, 0, sizeof(float), s));
    k_act_row_normsq<<<rows, 256, 0, s>>>(qkv, H, row_stride, normsq_scratch, total);
    k_act_row_clip_apply<<<rows, 256, 0, s>>>(qkv, H, row_stride, normsq_scratch, total, rows, mult);
    IDA_CUDA_CHECK(cudaMemsetAsync(total, 0, sizeof(float), s));
    k_act_row_normsq<<<rows, 256, 0, s>>>(qkv + H, H, row_stride, normsq_scratch, total);
    k_act_row_clip_apply<<<rows, 256, 0, s>>>(qkv + H, H, row_stride, normsq_scratch, total, rows, mult);
}

// dQ/dK/dV row clip — same interleaved [BS,3H] layout as qk_row_clip, but
// applied to the ATTENTION BACKWARD'S OUTPUT (dQ/dK/dV, freshly reshaped
// into sb.qkv), right before the weight-gradient GEMMs (dW_q/k/v) consume
// them.  This is the earliest point that can actually contain the
// explosion at its source, rather than downstream of it:
//
// Root cause (2026-07-09, three-probe bisection on the AI mb=128
// L0.attn_norm/k_proj/q_proj gradient explosion):
//   1. pre-layer d_hidden (LM-head/CE-loss gradient): ~1e-11, healthy —
//      rules out the loss gradient as the origin.
//   2. post-FFN-backward d_hidden (= dO into attention): ~2.4, healthy —
//      rules out FFN backward; the amplification is not inherited.
//   3. dS-cancellation probe inside the attention backward itself: from
//      that healthy ~2.4 dO, dp/D reach ~2e6 on the FIRST layer touched
//      (no prior compounding possible) and grow to ~6e12 by the last —
//      the ~1e17x amplification happens ENTIRELY inside the attention
//      backward's own dp/D/dS math, not accumulated across layers.
// interlayer_clip (applied to d_hidden AFTER a full layer's FFN+attention
// backward) was already too late: by that point THIS layer's own
// g_q/g_k/g_anorm were already corrupted from the exploded dQ/dK/dV.
// Clipping dQ/dK/dV directly, before the dW GEMMs read them, is the
// earliest point that actually contains the source.
// FIXED absolute row-L2-norm ceiling (not a multiplier) — see
// k_act_row_clip_apply_abs's comment for why relative/mean-based clipping
// fails here.  Default 1000: generous headroom above legitimate healthy
// dQ/dK/dV row norms (observed low tens at Edge scale, low hundreds at AI
// scale with self-LSE), and 1000+ orders of magnitude below every
// catastrophic value measured in the dS-cancellation probe (1e6-1e13).
static float dqkv_row_clip_ceiling() {
    static const float v = [] {
        const char* e = std::getenv("IDA_NATIVE_DQKV_ROW_CLIP");
        if (!e || !e[0]) return 1000.0f;
        char* end = nullptr;
        const float parsed = std::strtof(e, &end);
        return (end == e) ? 1000.0f : parsed;
    }();
    return v;
}

static void dqkv_row_clip(
    __nv_bfloat16* qkv, int rows, int H,
    float* normsq_scratch, cudaStream_t s, int kv_cols = -1
) {
    const float ceiling = dqkv_row_clip_ceiling();
    if (ceiling <= 0.0f) return;
    const int row_stride = 3 * H;
    for (int seg = 0; seg < 3; ++seg) {
        __nv_bfloat16* x = qkv + seg * H;
        const int cols = (seg == 0 || kv_cols <= 0) ? H : kv_cols;
        k_act_row_normsq<<<rows, 256, 0, s>>>(x, cols, row_stride, normsq_scratch, nullptr);
        k_act_row_clip_apply_abs<<<rows, 256, 0, s>>>(x, cols, row_stride, normsq_scratch, ceiling,
            g_rowclip_stats ? g_rowclip_stats + 8 : nullptr);
    }
}

// Direct per-weight-slot absolute clip on the persistent gradient
// accumulator (gl.g_q, gl.g_o, ...), applied immediately after each
// micro-step's contribution.  Targets what row-level clipping cannot:
// dW += X^T @ Y sums across ALL BS rows (262144 at mb=128), so even with
// every row individually bounded, the aggregate can still land far above
// any sane weight-gradient magnitude once enough rows are near their own
// ceiling — found 2026-07-09 when L2.o_proj stayed dominant at ~1.3e11
// even after every row-level operand clip in the layer was fixed and
// verified active.  Fully device-side (no host readback — this runs many
// times per micro-step, once per weight slot per layer).
static float slot_grad_abs_clip_ceiling() {
    static const float v = [] {
        // Era 13 default: OFF — never fired post-init-fix (ablation GN
        // bit-identical with it disabled).  IDA_NATIVE_SLOT_GRAD_ABS_CLIP
        // re-arms for ablation.
        const char* e = std::getenv("IDA_NATIVE_SLOT_GRAD_ABS_CLIP");
        if (!e || !e[0]) return 0.0f;
        char* end = nullptr;
        const float parsed = std::strtof(e, &end);
        return (end == e) ? 0.0f : parsed;
    }();
    return v;
}

static void slot_grad_abs_clip(
    float* g, std::size_t n, float* normsq_scratch, cudaStream_t s
) {
    const float ceiling = slot_grad_abs_clip_ceiling();
    if (ceiling <= 0.0f) return;
    IDA_CUDA_CHECK(cudaMemsetAsync(normsq_scratch, 0, sizeof(float), s));
    sq_sum_acc_f32(g, n, normsq_scratch, s);
    k_slot_grad_abs_clip_apply<<<ceildiv(n, 256), 256, 0, s>>>(g, n, normsq_scratch, ceiling);
}

static void slot_grad_abs_clip(
    __nv_bfloat16* g, std::size_t n, float* normsq_scratch, cudaStream_t s
) {
    const float ceiling = slot_grad_abs_clip_ceiling();
    if (ceiling <= 0.0f) return;
    IDA_CUDA_CHECK(cudaMemsetAsync(normsq_scratch, 0, sizeof(float), s));
    sq_sum_acc_bf16(g, n, normsq_scratch, s);
    k_slot_grad_abs_clip_apply_bf16<<<ceildiv(n, 256), 256, 0, s>>>(g, n, normsq_scratch, ceiling);
}

static void alloc_act_slot(
    ActSlot& slot,
    float fp8_max,
    NativeArena& arena
) {
    slot.fp8_max = fp8_max;
    IDA_CUDA_CHECK(ida_malloc_async(&slot.amax, sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&slot.scale, sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&slot.descale, sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&slot.scale_snapshot, sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&slot.descale_snapshot, sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&slot.history, kActHistoryLen * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&slot.history_cursor, sizeof(int), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&slot.history_count, sizeof(int), arena.pool, arena.stream));
    IDA_CUDA_CHECK(cudaMemsetAsync(slot.history, 0, kActHistoryLen * sizeof(float), arena.stream));
    IDA_CUDA_CHECK(cudaMemsetAsync(slot.history_cursor, 0, sizeof(int), arena.stream));
    IDA_CUDA_CHECK(cudaMemsetAsync(slot.history_count, 0, sizeof(int), arena.stream));
    slot.calib_remaining = fp8_calib_calls();
    if (slot.calib_remaining > 0) {
        IDA_CUDA_CHECK(ida_malloc_async(&slot.calib_min, sizeof(float), arena.pool, arena.stream));
        IDA_CUDA_CHECK(ida_malloc_async(&slot.calib_max, sizeof(float), arena.pool, arena.stream));
        // 0x7F byte-fill ≈ 3.39e38 — a valid float sentinel for "min so far".
        IDA_CUDA_CHECK(cudaMemsetAsync(slot.calib_min, 0x7F, sizeof(float), arena.stream));
        IDA_CUDA_CHECK(cudaMemsetAsync(slot.calib_max, 0, sizeof(float), arena.stream));
    }
}

enum Fp8SlotKind { F8_Q = 0, F8_K, F8_V, F8_O, F8_GATE, F8_UP, F8_DOWN };

static Fp8Ctx build_fp8_ctx(
    const NativeRequest& request,
    const LatticeWeights& w,
    std::size_t max_act_elems,
    int max_bs,
    NativeArena& arena
) {
    Fp8Ctx f{};
    const char* env = std::getenv("IDA_NATIVE_FP8");
    f.on = precision_profile_uses_fp8(request) && !(env && std::string(env) == "0");
    if (!f.on) return f;

    CUBLAS_CHECK(cublasLtCreate(&f.lt));
    IDA_CUDA_CHECK(ida_malloc_async(&f.ws, f.ws_size, arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&f.act8, max_act_elems, arena.pool, arena.stream));

    const int H = w.hidden_size, I = w.intermediate_size;
    const int L = w.num_layers,  V = w.vocab_size;
    // Cognitive-architecture MoE port: each MoE-enabled layer appends extra
    // mk() calls below (trunk_fc_in/out when moe_shared_trunk, always
    // expert_fc_in/out) beyond the base 7-per-layer+lm_head count. Without
    // counting them here, f.scalars is undersized and idx runs past its end
    // once any MoE layer's slots are built -- every later slot's amax/scale/
    // descale pointer (f.scalars + idx*12 + ...) then points past the
    // allocation, a real out-of-bounds atomic write (confirmed via
    // compute-sanitizer 2026-07-22: k_amax_bf16 atomicMax landing 1 byte
    // past an unrelated small allocation, corrupting whatever the arena
    // placed next to it).
    int moe_extra_slots = 0;
    for (int l = 0; l < L; ++l) {
        if (w.layers[l].num_experts > 0) {
            moe_extra_slots += w.layers[l].moe_shared_trunk ? 4 : 2;
        }
    }
    // 12 floats (48 B) per slot: amax/scale/descale each on their own 16-byte
    // boundary — cuBLASLt rejects FP8 scale pointers that are not 16B-aligned.
    const int n_slots = L * 7 + (w.owns_output ? 1 : 0) + moe_extra_slots;
    IDA_CUDA_CHECK(ida_malloc_async(&f.scalars, n_slots * 12 * sizeof(float), arena.pool, arena.stream));

    auto mk = [&](const __nv_bfloat16* wt, int in_d, int out_d, bool out_major, int idx) {
        Fp8Slot slot{};
        slot.w = wt; slot.in_dim = in_d; slot.out_dim = out_d;
        slot.stored_out_major = out_major;
        const std::size_t n = static_cast<std::size_t>(in_d) * out_d;
        IDA_CUDA_CHECK(ida_malloc_async(&slot.fwd8, n, arena.pool, arena.stream));
        IDA_CUDA_CHECK(ida_malloc_async(&slot.bwd8, n, arena.pool, arena.stream));
        slot.amax    = f.scalars + idx * 12;
        slot.scale   = f.scalars + idx * 12 + 4;
        slot.descale = f.scalars + idx * 12 + 8;
        f.slots.push_back(slot);
    };

    int idx = 0;
    for (int l = 0; l < L; ++l) {
        const auto& lw = w.layers[l];
        mk(lw.q_proj,    H, H, false, idx++);
        mk(lw.k_proj,    H, H, false, idx++);
        mk(lw.v_proj,    H, H, false, idx++);
        mk(lw.o_proj,    H, H, false, idx++);
        mk(lw.gate_proj, H, I, false, idx++);
        mk(lw.up_proj,   H, I, false, idx++);
        mk(lw.down_proj, I, H, false, idx++);
    }
    if (w.owns_output) {
        f.lm_head_weight_slot = idx;
        mk(w.lm_head, H, V, true, idx++);
    }

    // Step 4b: expert-bank weight FP8 slots, appended after lm_head.
    f.moe_weight_base.assign(static_cast<std::size_t>(L), -1);
    for (int l = 0; l < L; ++l) {
        const auto& lw = w.layers[l];
        if (lw.num_experts <= 0) continue;
        f.moe_weight_base[static_cast<std::size_t>(l)] = idx;
        const int Ie = lw.expert_intermediate_size;
        if (lw.moe_shared_trunk) {
            mk(lw.trunk_fc_in_w,  H,  I,  true, idx++);   // [I,H]  out-major
            mk(lw.trunk_fc_out_w, I,  H,  true, idx++);   // [H,I]  out-major
        }
        // expert_fc_in_w/expert_fc_out_w are ONE contiguous [num_experts,...]
        // allocation each -- quantize the whole block in one mk() call (n =
        // num_experts * Ie * H), matching build_param_slots' own "one slot
        // per weight TYPE, not per expert" treatment of the same tensors.
        mk(lw.expert_fc_in_w,  H,  lw.num_experts * Ie, true, idx++);  // [num_experts*Ie, H]
        mk(lw.expert_fc_out_w, Ie, lw.num_experts * H,  true, idx++);  // [num_experts*H, Ie] -- see note below
    }

    f.layer_act_slots.resize(static_cast<std::size_t>(L) * ACT_LAYER_SLOT_COUNT);
    for (auto& slot : f.layer_act_slots) {
        alloc_act_slot(slot, kE4M3Max, arena);
    }
    for (int l = 0; l < L; ++l) {
        layer_act_slot(f, l, ACT_DHIDDEN_DOWN).fp8_max = kE5M2Max;
        layer_act_slot(f, l, ACT_GATE_GRAD).fp8_max = kE5M2Max;
        layer_act_slot(f, l, ACT_UP_GRAD).fp8_max = kE5M2Max;
        layer_act_slot(f, l, ACT_DHIDDEN_O).fp8_max = kE5M2Max;
        layer_act_slot(f, l, ACT_QKV_GRAD).fp8_max = kE5M2Max;
    }
    if (w.owns_output) {
        f.global_act_slots.resize(ACT_GLOBAL_SLOT_COUNT);
        alloc_act_slot(global_act_slot(f, ACT_LMHEAD_IN), kE4M3Max, arena);
        alloc_act_slot(global_act_slot(f, ACT_LOGITS_GRAD), kE5M2Max, arena);
    }

    // Step 4b: expert-bank activation FP8 slots. Slot 0 = shared fc_in-input
    // (hidden_gated), reused for the trunk's and every expert's fc_in GEMM
    // (same input tensor). Slot 1 = trunk's post-GELU fc_out-input (Mode B
    // only). Slots [1 or 2 .. +num_experts) = each expert's own post-GELU
    // fc_out-input (genuinely distinct per expert, unlike fc_in's shared
    // input).
    f.moe_act_base.assign(static_cast<std::size_t>(L), -1);
    for (int l = 0; l < L; ++l) {
        const auto& lw = w.layers[l];
        if (lw.num_experts <= 0) continue;
        f.moe_act_base[static_cast<std::size_t>(l)] =
            static_cast<int>(f.moe_act_slots.size());
        const int n_act = 1 + (lw.moe_shared_trunk ? 1 : 0) + lw.num_experts;
        for (int i = 0; i < n_act; ++i) {
            f.moe_act_slots.emplace_back();
            alloc_act_slot(f.moe_act_slots.back(), kE4M3Max, arena);
        }
    }
    // FP8 E4M3 activation snapshots for dW GEMMs — sized for one layer's activations.
    // IDA_FP8_DW=0 leaves the snap pointers null: dW GEMMs fall back to BF16
    // while forward/dx stay FP8 (bisect switch for the snap-symmetry path).
    const char* dw_env = std::getenv("IDA_FP8_DW");
    if (!(dw_env && std::string(dw_env) == "0")) {
        const std::size_t snap_h = static_cast<std::size_t>(max_bs) * H;
        const std::size_t snap_i = static_cast<std::size_t>(max_bs) * I;
        IDA_CUDA_CHECK(ida_malloc_async(&f.snap_qkv,  snap_h, arena.pool, arena.stream));
        IDA_CUDA_CHECK(ida_malloc_async(&f.snap_o,    snap_h, arena.pool, arena.stream));
        IDA_CUDA_CHECK(ida_malloc_async(&f.snap_ffn,  snap_h, arena.pool, arena.stream));
        IDA_CUDA_CHECK(ida_malloc_async(&f.snap_down, snap_i, arena.pool, arena.stream));
        // Transposed gradient scratch — largest dW A-operand is qkv [3H,BS] or FFN [I,BS]
        const std::size_t grad_t_bytes =
            static_cast<std::size_t>(max_bs) * std::max(3 * H, I);
        IDA_CUDA_CHECK(ida_malloc_async(&f.grad8_t, grad_t_bytes, arena.pool, arena.stream));
    }
    return f;
}

static void free_fp8_ctx(Fp8Ctx& f, NativeArena& arena) {
    if (!f.on) return;
    auto fr = [&](void* p) { if (p) IDA_CUDA_CHECK(cudaFreeAsync(p, arena.stream)); };
    for (auto& entry : f.plans)     destroy_lt_plan(entry.second);
    for (auto& entry : f.plans_f32) destroy_lt_plan(entry.second);
    for (auto& entry : f.plans_bf16) destroy_lt_plan(entry.second);
    for (auto& s : f.slots) { fr(s.fwd8); fr(s.bwd8); }
    for (auto& s : f.layer_act_slots) {
        fr(s.amax); fr(s.scale); fr(s.descale);
        fr(s.scale_snapshot); fr(s.descale_snapshot);
        fr(s.history); fr(s.history_cursor); fr(s.history_count);
        fr(s.calib_min); fr(s.calib_max);
    }
    for (auto& s : f.global_act_slots) {
        fr(s.amax); fr(s.scale); fr(s.descale);
        fr(s.scale_snapshot); fr(s.descale_snapshot);
        fr(s.history); fr(s.history_cursor); fr(s.history_count);
        fr(s.calib_min); fr(s.calib_max);
    }
    for (auto& s : f.moe_act_slots) {
        fr(s.amax); fr(s.scale); fr(s.descale);
        fr(s.scale_snapshot); fr(s.descale_snapshot);
        fr(s.history); fr(s.history_cursor); fr(s.history_count);
        fr(s.calib_min); fr(s.calib_max);
    }
    fr(f.scalars); fr(f.ws); fr(f.act8);
    fr(f.snap_qkv); fr(f.snap_o); fr(f.snap_ffn); fr(f.snap_down); fr(f.grad8_t);
    cublasLtDestroy(f.lt);
    f.on = false;
}

static void bind_fp8_weight_amax_slots(
    std::vector<ParamSlot>& params,
    Fp8Ctx& f8,
    const LatticeWeights& w
) {
    if (!f8.on) return;
    for (auto& p : params) {
        if (w.owns_output && p.w == w.lm_head && f8.lm_head_weight_slot >= 0) {
            p.fp8_amax = f8.slots[static_cast<std::size_t>(f8.lm_head_weight_slot)].amax;
            continue;
        }
        for (int l = 0; l < w.num_layers; ++l) {
            const auto& lw = w.layers[l];
            const std::size_t fp8_base = static_cast<std::size_t>(l) * 7;
            if (p.w == lw.q_proj)    p.fp8_amax = f8.slots[fp8_base + F8_Q].amax;
            if (p.w == lw.k_proj)    p.fp8_amax = f8.slots[fp8_base + F8_K].amax;
            if (p.w == lw.v_proj)    p.fp8_amax = f8.slots[fp8_base + F8_V].amax;
            if (p.w == lw.o_proj)    p.fp8_amax = f8.slots[fp8_base + F8_O].amax;
            if (p.w == lw.gate_proj) p.fp8_amax = f8.slots[fp8_base + F8_GATE].amax;
            if (p.w == lw.up_proj)   p.fp8_amax = f8.slots[fp8_base + F8_UP].amax;
            if (p.w == lw.down_proj) p.fp8_amax = f8.slots[fp8_base + F8_DOWN].amax;
        }
    }
}

static LtPlan& get_lt_plan(
    Fp8Ctx& f,
    int M, int N, int K,
    int lda, int ldc,
    cudaDataType_t a_type
) {
    if (!f.heuristics_cache_configured) {
        CUBLAS_CHECK(cublasLtHeuristicsCacheSetCapacity(512));
        f.heuristics_cache_configured = true;
    }

    const LtPlanKey key = {
        static_cast<long long>(M),
        static_cast<long long>(N),
        static_cast<long long>(K),
        static_cast<long long>(lda),
        static_cast<long long>(ldc),
        static_cast<long long>(a_type),
    };
    auto it = f.plans.find(key);
    if (it != f.plans.end()) {
        return it->second;
    }

    LtPlan plan{};
    const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.op, compute_type, CUDA_R_32F));
    // FP8 requires the TN form.  Probe-verified mapping: cuBLASLt operand "A"
    // is our WEIGHT ([N,K] row-major = col [K,N], OP_T → [N,K]); operand "B"
    // is our ACTIVATION ([M,K] row-major = col [K,M], OP_N).  D is col [N,M].
    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.op, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.la, CUDA_R_8F_E4M3, K, N, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.lb, a_type, K, M, lda));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.lc, CUDA_R_16BF, N, M, ldc));

    cublasLtMatmulPreference_t pref{};
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &f.ws_size,
        sizeof(f.ws_size)
    ));

    int returned = 0;
    const auto status = cublasLtMatmulAlgoGetHeuristic(
        f.lt,
        plan.op,
        plan.la,
        plan.lb,
        plan.lc,
        plan.lc,
        pref,
        1,
        &plan.heuristic,
        &returned
    );
    cublasLtMatmulPreferenceDestroy(pref);
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0) {
        destroy_lt_plan(plan);
        throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic failed for FP8 NT plan");
    }
    plan.ready = true;
    // One-time-per-unique-shape diagnostic: which cuBLASLt algorithm actually
    // got selected. Added 2026-08-14 after a real nsys profile showed every
    // GEMM kernel name tagged for an older architecture (sm89_xmma_gemm_*,
    // cutlass_80_tensorop_*) with none tagged sm_120 -- ncu's own occupancy
    // metrics were unavailable (ERR_NVGPUCTRPERM, ncu needs host-level admin
    // rights this rented instance doesn't grant). algoId + wavesCount are
    // free: cuBLASLt already computes them inside AlgoGetHeuristic, so this
    // costs nothing extra and needs no hardware performance-counter access.
    if (::ida_native::gemm_trace::enabled()) {
        int algo_id = -1;
        std::size_t written = 0;
        cublasLtMatmulAlgoConfigGetAttribute(
            &plan.heuristic.algo, CUBLASLT_ALGO_CONFIG_ID,
            &algo_id, sizeof(algo_id), &written);
        std::fprintf(
            ::ida_native::ontology::sink(),
            "{\"type\":\"gemm_algo\",\"M\":%d,\"N\":%d,\"K\":%d,"
            "\"algo_id\":%d,\"waves_count\":%.3f,\"workspace_bytes\":%zu}\n",
            M, N, K, algo_id,
            plan.heuristic.wavesCount,
            plan.heuristic.workspaceSize);
        std::fflush(::ida_native::ontology::sink());
    }
    auto [inserted, _ok] = f.plans.emplace(key, std::move(plan));
    return inserted->second;
}

static void lt_gemm_fp8_nt(
    Fp8Ctx& f,
    int M, int N, int K,
    const void* A, int lda, cudaDataType_t a_type, const float* a_descale,
    const void* B, const float* b_descale,
    __nv_bfloat16* C, int ldc,
    cudaStream_t s
) {
    ::ida_native::gemm_trace::record_current(static_cast<long long>(M), static_cast<long long>(N), static_cast<long long>(K));
    LtPlan& plan = get_lt_plan(f, M, N, K, lda, ldc, a_type);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // cuBLASLt "A" is the weight (la), "B" is the activation (lb): the scale
    // attributes and operand order follow that mapping.
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &b_descale, sizeof(b_descale)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a_descale, sizeof(a_descale)));
    CUBLAS_CHECK(cublasLtMatmul(
        f.lt,
        plan.op,
        &alpha,
        B,
        plan.la,
        A,
        plan.lb,
        &beta,
        C,
        plan.lc,
        C,
        plan.lc,
        &plan.heuristic.algo,
        f.ws,
        f.ws_size,
        s
    ));
}

// Weight-gradient GEMM: dW[M,N] (β=1 accum) += X^T @ dY with FP8 inputs → F32 grad.
// X: E4M3 [K,M] row-major (activation snap), dY: E5M2 [K,N] row-major (gradient).
// Uses the cuBLAS swap trick: cuBLAS-A=dY (OP_N), cuBLAS-B=X (OP_T), C=dW [N,M] col.
static LtPlan& get_lt_plan_f32(
    Fp8Ctx& f,
    int M, int N, int K,
    int lda_x, int ldb_dy, int ldc_dw
) {
    if (!f.heuristics_cache_configured) {
        CUBLAS_CHECK(cublasLtHeuristicsCacheSetCapacity(512));
        f.heuristics_cache_configured = true;
    }
    const LtPlanKey key = {
        static_cast<long long>(M),
        static_cast<long long>(N),
        static_cast<long long>(K),
        static_cast<long long>(lda_x),
        static_cast<long long>(ldb_dy),
        static_cast<long long>(ldc_dw),
    };
    auto it = f.plans_f32.find(key);
    if (it != f.plans_f32.end()) return it->second;

    LtPlan plan{};
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    // cuBLASLt FP8 only supports TN: A=OP_T, B=OP_N.  Both operands arrive
    // pre-transposed in memory (dY^T [N,K] and X^T [M,K], row-major, ld=K):
    //   op(A) = (col[K,N])^T = dY^T [N,K];  op(B) = col[K,M] = X [K,M]
    //   C[N,M] col = dY^T @ X = dW^T col = dW[M,N] row-major.
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    // la = dY^T[N,K] row → col [K,N,ld=K]  E5M2
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.la, CUDA_R_8F_E5M2, K, N, K));
    // lb = X^T[M,K] row → col [K,M,ld=K]  E4M3
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.lb, CUDA_R_8F_E4M3, K, M, K));
    // lc = dW[M,N] row → col [N,M,ld=ldc_dw]; beta=1 accumulates
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.lc, CUDA_R_32F, N, M, ldc_dw));

    cublasLtMatmulPreference_t pref{};
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &f.ws_size, sizeof(f.ws_size)));
    int returned = 0;
    const auto status = cublasLtMatmulAlgoGetHeuristic(
        f.lt, plan.op, plan.la, plan.lb, plan.lc, plan.lc,
        pref, 1, &plan.heuristic, &returned);
    cublasLtMatmulPreferenceDestroy(pref);
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0) {
        destroy_lt_plan(plan);
        throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic failed for FP8 TN F32 dW plan");
    }
    plan.ready = true;
    auto [inserted, _ok] = f.plans_f32.emplace(key, std::move(plan));
    return inserted->second;
}

static void lt_gemm_fp8_tn_accum_f32(
    Fp8Ctx& f,
    int M, int N, int K,
    const void* Xt_e4m3,   // X^T [M,K] E4M3 row-major contiguous (pre-transposed snap)
    const float* ds_x,     // E4M3 descale (GPU ptr)
    const void* dYt_e5m2,  // dY^T [N,K] E5M2 row-major contiguous (pre-transposed grad)
    const float* ds_dy,    // E5M2 descale (GPU ptr)
    float* dW, int ldc_dw, // dW[M,N] F32 row-major, row stride ldc_dw
    cudaStream_t s
) {
    ::ida_native::gemm_trace::record_current(static_cast<long long>(M), static_cast<long long>(N), static_cast<long long>(K));
    LtPlan& plan = get_lt_plan_f32(f, M, N, K, K, K, ldc_dw);
    const float alpha = 1.0f;
    const float beta  = 1.0f;  // accumulate into persistent grad buffer
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ds_dy, sizeof(ds_dy)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &ds_x,  sizeof(ds_x)));
    CUBLAS_CHECK(cublasLtMatmul(
        f.lt, plan.op, &alpha,
        dYt_e5m2, plan.la,  // cuBLAS-A = dY^T (OP_T applied → dY^T [N,K])
        Xt_e4m3,  plan.lb,  // cuBLAS-B = X^T (OP_N → X [K,M] col view)
        &beta,
        dW, plan.lc,        // C (read for accumulation)
        dW, plan.lc,        // D (write)
        &plan.heuristic.algo,
        f.ws, f.ws_size,
        s
    ));
}

static LtPlan& get_lt_plan_bf16(
    Fp8Ctx& f, int M, int N, int K, int lda_x, int ldb_dy, int ldc_dw
) {
    if (!f.heuristics_cache_configured) {
        CUBLAS_CHECK(cublasLtHeuristicsCacheSetCapacity(512));
        f.heuristics_cache_configured = true;
    }
    const LtPlanKey key = {
        static_cast<long long>(M), static_cast<long long>(N), static_cast<long long>(K),
        static_cast<long long>(lda_x), static_cast<long long>(ldb_dy),
        static_cast<long long>(ldc_dw),
    };
    auto it = f.plans_bf16.find(key);
    if (it != f.plans_bf16.end()) return it->second;
    LtPlan plan{};
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.la, CUDA_R_8F_E5M2, K, N, ldb_dy));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.lb, CUDA_R_8F_E4M3, K, M, lda_x));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.lc, CUDA_R_16BF, N, M, ldc_dw));
    cublasLtMatmulPreference_t pref{};
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &f.ws_size, sizeof(f.ws_size)));
    int returned = 0;
    const auto status = cublasLtMatmulAlgoGetHeuristic(
        f.lt, plan.op, plan.la, plan.lb, plan.lc, plan.lc,
        pref, 1, &plan.heuristic, &returned);
    cublasLtMatmulPreferenceDestroy(pref);
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0) {
        destroy_lt_plan(plan);
        throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic failed for FP8 TN BF16 dW plan");
    }
    plan.ready = true;
    auto [inserted, _ok] = f.plans_bf16.emplace(key, std::move(plan));
    return inserted->second;
}

static void lt_gemm_fp8_tn_accum_f32(
    Fp8Ctx& f,
    int M, int N, int K,
    const void* Xt_e4m3, const float* ds_x,
    const void* dYt_e5m2, const float* ds_dy,
    __nv_bfloat16* dW, int ldc_dw,
    cudaStream_t s
) {
    ::ida_native::gemm_trace::record_current(static_cast<long long>(M), static_cast<long long>(N), static_cast<long long>(K));
    LtPlan& plan = get_lt_plan_bf16(f, M, N, K, K, K, ldc_dw);
    const float alpha = 1.0f;
    const float beta = 1.0f;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ds_dy, sizeof(ds_dy)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &ds_x, sizeof(ds_x)));
    CUBLAS_CHECK(cublasLtMatmul(
        f.lt, plan.op, &alpha,
        dYt_e5m2, plan.la, Xt_e4m3, plan.lb,
        &beta, dW, plan.lc, dW, plan.lc,
        &plan.heuristic.algo, f.ws, f.ws_size, s));
}

// Requantize every cached weight (called after each optimizer update).
static void refresh_fp8_weights(Fp8Ctx& f, cudaStream_t s, bool recompute_amax) {
    if (!f.on) return;
    for (auto& sl : f.slots) {
        const std::size_t n = static_cast<std::size_t>(sl.in_dim) * sl.out_dim;
        if (recompute_amax) {
            fp8_amax_bf16(sl.w, n, sl.amax, s);
        }
        fp8_scale_from_amax(sl.amax, kE4M3Max, sl.scale, sl.descale, s);
        if (sl.stored_out_major) {
            // stored [out,in]: direct quant IS the forward layout
            fp8_quantize_e4m3(sl.w, sl.fwd8, sl.scale, n, s);
            fp8_quantize_transpose_e4m3(sl.w, sl.bwd8, sl.scale, sl.out_dim, sl.in_dim, s);
        } else {
            // stored [in,out]: direct quant IS the backward layout
            fp8_quantize_e4m3(sl.w, sl.bwd8, sl.scale, n, s);
            fp8_quantize_transpose_e4m3(sl.w, sl.fwd8, sl.scale, sl.in_dim, sl.out_dim, s);
        }
    }
}

// Quantize an activation/gradient tensor into the shared scratch buffer.
// First use seeds the delayed-scaling slot with an exact amax pass; every
// later call uses the previous scale and records the next amax as a side
// effect, then derives the next scale from a short history window.
//
// is_recompute=true is for the backward pass's activation-checkpoint
// recompute of this same layer within the same micro-step: it must quantize
// with the EXACT scale the original forward call used, read-only, with no
// amax-record or history-advance side effect. Without this, the recompute
// call (which runs after forward has already advanced slot.scale) would
// quantize the same data with a different scale than forward did — a real,
// deterministic mismatch, not floating-point noise.
// The cuBLASLt operand type matching what fp8_quant_act just wrote for this
// slot (calibration may have committed the slot to E5M2).
static cudaDataType_t act_fp8_type(const ActSlot& slot) {
    return slot.fp8_max == kE5M2Max ? CUDA_R_8F_E5M2 : CUDA_R_8F_E4M3;
}

static void fp8_quant_act(
    Fp8Ctx& f, ActSlot& slot, const __nv_bfloat16* x, std::size_t n, cudaStream_t s,
    bool is_recompute = false
) {
    if (is_recompute) {
        if (slot.fp8_max == kE5M2Max) {
            fp8_quantize_e5m2(x, f.act8, slot.scale_snapshot, n, s);
        } else {
            fp8_quantize_e4m3(x, f.act8, slot.scale_snapshot, n, s);
        }
        return;
    }

    // Commit a pending calibration decision before this call quantizes —
    // never between a forward and its recompute (see ActSlot comment).
    if (slot.calib_commit_pending) {
        slot.calib_commit_pending = false;
        float mn = 0.0f, mx = 0.0f;
        IDA_CUDA_CHECK(cudaStreamSynchronize(s));
        IDA_CUDA_CHECK(cudaMemcpy(&mn, slot.calib_min, sizeof(float), cudaMemcpyDeviceToHost));
        IDA_CUDA_CHECK(cudaMemcpy(&mx, slot.calib_max, sizeof(float), cudaMemcpyDeviceToHost));
        const float spread = (mn > 0.0f) ? (mx / mn) : 0.0f;
        if (slot.fp8_max == kE4M3Max && spread > fp8_type_spread_threshold()) {
            // Range moves too much for E4M3's window — commit to E5M2 and
            // reseed the delayed state from the widest observed amax.
            slot.fp8_max = kE5M2Max;
            IDA_CUDA_CHECK(cudaMemcpyAsync(slot.amax, slot.calib_max, sizeof(float),
                                           cudaMemcpyDeviceToDevice, s));
            IDA_CUDA_CHECK(cudaMemsetAsync(slot.history_cursor, 0, sizeof(int), s));
            IDA_CUDA_CHECK(cudaMemsetAsync(slot.history_count, 0, sizeof(int), s));
            fp8_scale_from_amax(slot.amax, slot.fp8_max, slot.scale, slot.descale, s);
            fp8_update_delayed_scale(
                slot.amax, slot.fp8_max, slot.scale, slot.descale,
                slot.history, kActHistoryLen, slot.history_cursor,
                slot.history_count, s);
        }
    }

    IDA_CUDA_CHECK(cudaMemsetAsync(slot.amax, 0, sizeof(float), s));
    if (slot.calib_remaining > 0) {
        // Calibration: exact per-call amax (no stale scale) + spread recording.
        fp8_amax_bf16(x, n, slot.amax, s);
        fp8_scale_from_amax(slot.amax, slot.fp8_max, slot.scale, slot.descale, s);
        IDA_CUDA_CHECK(cudaMemcpyAsync(slot.scale_snapshot, slot.scale, sizeof(float),
                                       cudaMemcpyDeviceToDevice, s));
        IDA_CUDA_CHECK(cudaMemcpyAsync(slot.descale_snapshot, slot.descale, sizeof(float),
                                       cudaMemcpyDeviceToDevice, s));
        if (slot.fp8_max == kE5M2Max) {
            fp8_quantize_e5m2(x, f.act8, slot.scale, n, s);
        } else {
            fp8_quantize_e4m3(x, f.act8, slot.scale, n, s);
        }
        k_calib_minmax_update<<<1, 1, 0, s>>>(slot.amax, slot.calib_min, slot.calib_max);
        fp8_update_delayed_scale(
            slot.amax, slot.fp8_max, slot.scale, slot.descale,
            slot.history, kActHistoryLen, slot.history_cursor,
            slot.history_count, s);
        slot.init = true;
        if (--slot.calib_remaining == 0) slot.calib_commit_pending = true;
        return;
    }
    if (!slot.init) {
        fp8_amax_bf16(x, n, slot.amax, s);
        fp8_scale_from_amax(slot.amax, slot.fp8_max, slot.scale, slot.descale, s);
        IDA_CUDA_CHECK(cudaMemcpyAsync(slot.scale_snapshot, slot.scale, sizeof(float),
                                       cudaMemcpyDeviceToDevice, s));
        IDA_CUDA_CHECK(cudaMemcpyAsync(slot.descale_snapshot, slot.descale, sizeof(float),
                                       cudaMemcpyDeviceToDevice, s));
        if (slot.fp8_max == kE5M2Max) {
            fp8_quantize_e5m2(x, f.act8, slot.scale, n, s);
        } else {
            fp8_quantize_e4m3(x, f.act8, slot.scale, n, s);
        }
        fp8_update_delayed_scale(
            slot.amax,
            slot.fp8_max,
            slot.scale,
            slot.descale,
            slot.history,
            kActHistoryLen,
            slot.history_cursor,
            slot.history_count,
            s
        );
        slot.init = true;
        return;
    }

    // Snapshot the scale/descale forward is about to use BEFORE they
    // advance, so the recompute call (later, same micro-step) can reproduce
    // both exactly.
    IDA_CUDA_CHECK(cudaMemcpyAsync(slot.scale_snapshot, slot.scale, sizeof(float),
                                   cudaMemcpyDeviceToDevice, s));
    IDA_CUDA_CHECK(cudaMemcpyAsync(slot.descale_snapshot, slot.descale, sizeof(float),
                                   cudaMemcpyDeviceToDevice, s));
    if (slot.fp8_max == kE5M2Max) {
        fp8_quantize_e5m2_record(x, f.act8, slot.scale, slot.amax, n, s);
    } else {
        fp8_quantize_e4m3_record(x, f.act8, slot.scale, slot.amax, n, s);
    }
    // Probe BEFORE the scale advances: slot.scale is still the value the
    // quantize above actually used, slot.amax is this call's true amax.
    if (fp8_clip_debug_enabled() && g_fp8_clip_stats != nullptr) {
        fp8_clip_probe(slot.amax, slot.scale, slot.fp8_max, g_fp8_clip_stats, s);
        if (g_fp8_elem_stats != nullptr) {
            fp8_elem_clip_count(x, slot.scale, slot.fp8_max, n, g_fp8_elem_stats, s);
        }
    }
    fp8_update_delayed_scale(
        slot.amax,
        slot.fp8_max,
        slot.scale,
        slot.descale,
        slot.history,
        kActHistoryLen,
        slot.history_cursor,
        slot.history_count,
        s
    );
}

// ─── per-step activation buffers ─────────────────────────────────────────────

struct PssFenceMetrics {
    float err_sq{};
    float target_sq{};
    std::uint32_t covered{};
    std::uint32_t int2_matched{};
    std::uint32_t int2_scored{};
};
struct PssFenceEvidence {
    PssFenceMetrics metrics{};
    float int2_inv_rms{};
    float confidence_min{};
    float confidence_max{};
};
static_assert(sizeof(PssFenceMetrics) == 5 * sizeof(float));
static_assert(sizeof(PssFenceEvidence) == 8 * sizeof(float));

struct StepBuffers {
    // Shapes: B = batch, S = seq_len, H = hidden, I = intermediate, V = vocab
    // nH = num_heads, Hd = head_dim = H/nH, kvH = physical K/V width.
    int B, S, H, I, V, nH, Hd, kvH;
    int mb_alloc;   // allocated batch (B may be clamped below this per step)

    // Residual stream + norm scratch
    __nv_bfloat16* hidden{};       // [B*S, H]
    __nv_bfloat16* normed{};       // [B*S, H]  attention-norm output (kept for dW_qkv)
    __nv_bfloat16* normed2{};      // [B*S, H]  ffn-norm output (kept for dW_gate/up)
    float*         rms_save{};     // [B*S]     ffn / final norm rms
    float*         rms_a{};        // [B*S]     attention norm rms

    // Saved layer inputs for recompute-based backward: [L+1][mb*S, H]
    // slot l = input to layer l; slot L = input to the final norm.
    // With IDA_NATIVE_SAVED_ACT_HOST_OFFLOAD=1, `saved` holds only a
    // saved_ring-deep device ring and the full [L+1] set lives in pinned
    // DDR5 (`saved_host`): forward drains each slice D2H behind the compute,
    // backward prefetches H2D ahead of the recompute.  Slices still resident
    // from the forward tail (the last `saved_ring` layers) skip the
    // round-trip.  Device cost drops from (L+1)·mb·S·H to ring·mb·S·H —
    // 9.3 GiB → 2.2 GiB on the 16-layer AI body at mb=64.
    __nv_bfloat16* saved{};
    __nv_bfloat16* saved_host{};        // pinned [L+1][mb*S,H]; null = off
    int            saved_ring{0};       // device ring depth (0 = offload off)
    int            saved_slots{0};      // L+1 (pinned slot count)
    cudaStream_t   saved_copy_stream{};
    // ev_release[r]: compute-stream record — slot r's content is ready for
    // the copy stream (fwd: D2D landed) or no longer needed (bwd: last read
    // enqueued).  ev_copied[r]: copy-stream record — the copy involving slot
    // r (fwd D2H / bwd H2D) is complete.  Waits and records are paired in
    // host program order, single-threaded, so each wait latches its
    // intended record.
    cudaEvent_t*   saved_ev_release{};  // [saved_ring]
    cudaEvent_t*   saved_ev_copied{};   // [saved_ring]

    // Attention scratch — BF16 (flash-style; no [S,S] buffer). LSE/rowdot are
    // the only FP32 survivors here: online-softmax accumulation needs the
    // range, everything Q/K/V/O-shaped is BF16 in and out.
    __nv_bfloat16* qkv{};          // [B*S, 3H]
    __nv_bfloat16* q_f32{};        // [B, nH, S, Hd]
    __nv_bfloat16* k_f32{};        // [B, nH, S, Hd]
    __nv_bfloat16* v_f32{};        // [B, nH, S, Hd]
    __nv_bfloat16* o_f32{};        // [B, nH, S, Hd]  attention context
    __nv_bfloat16* do_f32{};       // [B, nH, S, Hd]  d(context) in backward
    __nv_bfloat16* dq_f32{};       // [B, nH, S, Hd]
    __nv_bfloat16* dk_f32{};       // [B, nH, S, Hd]
    __nv_bfloat16* dv_f32{};       // [B, nH, S, Hd]
    float*         lse{};          // [B, nH, S]  log-sum-exp per query row
    float*         rowdot{};       // [B, nH, S]  D = rowsum(dO ∘ O) scratch
    __nv_bfloat16* attn_out{};     // [B*S, H]
    __nv_bfloat16* o_out{};        // [B*S, H]
    const uint16_t* segs{};        // [B, S] sample-start offsets (owned by caller)

    // FFN scratch
    __nv_bfloat16* gate_out{};     // [B*S, I]
    __nv_bfloat16* up_out{};       // [B*S, I]
    __nv_bfloat16* swiglu_out{};   // [B*S, I]
    __nv_bfloat16* d_ffn_i{};      // [B*S, I]  d(swiglu) in backward
    __nv_bfloat16* ffn_out{};      // [B*S, H]

    // Cognitive-architecture sparse MoE port (null when the whole layer's
    // num_experts==0 -- shared across layers within one burn since all
    // MoE-enabled layers use the same num_experts/num_routes, see
    // allocate_lattice_weights). Sized for this request's max num_routes/
    // num_experts across layers (uniform in practice today).
    // Reused across both pooling passes (pool of hidden, then pool of
    // hidden_gated) -- only meaningful at each sample's start position
    // until gathered; see moe_pool_by_sample's contract.
    float* moe_pool_scratch{};      // [B*S, H]   pooling accumulator/scratch
    float* moe_count_scratch{};     // [B*S]      pooling count scratch
    float* moe_pooled{};            // [B*S, H]   gathered pool(hidden), pass 1
    float* moe_pressure{};          // [B*S, num_routes]  tanh(proj(pooled)) -- already per-position (moe_pooled was gathered)
    float* moe_modulation{};        // [B*S, H]   sigmoid(mod(pressure)) -- already per-position
    __nv_bfloat16* moe_hidden_gated{}; // [B*S, H]  hidden * modulation (bf16, feeds pass-2 pooling + Step 4's expert input)
    float* moe_pooled2{};           // [B*S, H]   gathered pool(hidden_gated), pass 2
    // Only allocated/used when pressure_to_routes_w is present (num_routes
    // != num_experts); when they're equal (Identity in the reference),
    // moe_pressure is used directly as the "routed pressure" term instead.
    float* moe_routed_pressure{};   // [B*S, num_experts]
    float* moe_logits{};            // [B*S, num_experts]: router_score(pooled2) -> += routed_pressure -> softmax+topk (in place) -> lateral inhibition (in place) = FINAL route weights, per-position, ready for Step 4
    float* moe_route_weights{};     // [B*S, num_experts]  final, gathered per-position (post lateral-inhibition)

    // Step 4: CognitiveCircuitBank dispatch scratch. moe_expert_scratch holds
    // one expert's (or the trunk's) fc_in output, pre- and then post-GELU in
    // place -- sized to the wider of trunk width (I) and per-expert width
    // (expert_intermediate_size) since Mode B's residual experts are
    // typically narrower than the trunk but nothing enforces that.
    // moe_expert_out holds that same pass's fc_out output before it's
    // scaled by its route weight and accumulated into sb.ffn_out (which
    // moe_expert_bank_forward zeroes and reuses as the mixed accumulator).
    __nv_bfloat16* moe_expert_scratch{}; // [B*S, max(I, expert_intermediate_size)]
    __nv_bfloat16* moe_expert_out{};     // [B*S, H]
    int* moe_dispatch_counts{};          // [num_experts], selected-row ablation
    int* moe_dispatch_indices{};         // [num_experts, B*S], selected rows
    int* moe_dispatch_offsets{};         // [num_experts], device-side prefix offsets for grouped scheduler
    int* moe_dispatch_counts_saved{};    // [layers, num_experts], forward-owned selected rows
    int* moe_dispatch_indices_saved{};   // [layers, num_experts, B*S], forward-owned selected rows
    std::vector<int> moe_dispatch_counts_host;
    std::vector<int> moe_dispatch_counts_host_saved;

    // Step 5: backward scratch/saves. Small per-row scalars (T/c1/S) are
    // SAVED (populated every moe_router_forward call, including backward's
    // recompute pass -- cheap enough per the plan's own "small saved masks"
    // guidance); large intermediates (hidden_gated, per-expert pre/post-GELU)
    // are RECOMPUTED during backward instead, matching this engine's
    // existing recompute-based-backward philosophy everywhere else.
    float* moe_count{};               // [B*S] sample counts (segs-derived, shared by both pool backwards)
    float* moe_start_grad_scratch{};  // [B*S,H] scratch for moe_pool_by_sample_backward (both calls, sequential)
    __nv_bfloat16* moe_d_hidden_gated{}; // [B*S,H] accumulator: total gradient into hidden_gated
    float* moe_d_pooled{};             // [B*S,H] scratch: d(pooled) before its pool-backward reduction
    float* moe_d_pooled2{};            // [B*S,H] scratch: d(pooled2) before its pool-backward reduction
    float* moe_d_pressure{};           // [B*S,num_routes] accumulator: d(pressure), both paths summed
    float* moe_d_pressure_contrib2{};  // [B*S,num_routes] scratch: pressure_mod's own contribution,
                                        // computed separately then added into moe_d_pressure
    float* moe_d_modulation{};         // [B*S,H] scratch: d(modulation), fresh per call
    float* moe_d_logits{};             // [B*S,num_experts] scratch: reused through the softmax/topk/inhibition backward chain
    float* moe_d_routed_pressure{};    // [B*S,num_experts] scratch (only when pressure_to_routes_w present)
    float* moe_topk_row_sum{};         // [B*S] T, saved by moe_topk_route_f32
    float* moe_scores_save{};          // [B*S,num_experts] pure softmax output, pre-top-k --
                                        // needed by attn_softmax_backward_f32
    float* moe_topk_p_save{};          // [B*S,num_experts] p (post top-k renorm, pre-inhibition) --
                                        // snapshotted before moe_lateral_inhibition_f32 overwrites
                                        // sb.moe_logits further; needed as the "y" in the top-k
                                        // renormalize's own norm-by-sum backward
    float* moe_inhib_c1{};             // [B*S,num_experts] bounded_pre_floor, saved by moe_lateral_inhibition_f32
    float* moe_inhib_row_sum{};        // [B*S] S, saved by moe_lateral_inhibition_f32
    __nv_bfloat16* moe_expert_scratch2{}; // [B*S,max(I,Ie)] second scratch: post-GELU, kept separate from
                                          // moe_expert_scratch's pre-GELU value during backward recompute
    __nv_bfloat16* moe_d_expert_scratch{}; // [B*S,max(I,Ie)] d(post_gelu), then overwritten in place
                                            // with d(pre_gelu) via gelu_backward (safe: elementwise, same index)
    float* moe_probe_amax{};               // [1] scratch for moe_fake_quant_roundtrip's amax
                                            // (quantization-quality probe, IDA_NATIVE_MOE_ACT_PRECISION) --
                                            // dedicated, NOT shared with any real-data buffer

    // Generic per-token top-k SwiGLU MoE (2026-08-23). Deliberately its OWN
    // scratch, not aliased onto any of the legacy-mode buffers above, even
    // though several are similarly shaped -- the two modes never run in the
    // same layer (moe_kind is per-layer, mutually exclusive), but keeping
    // them unaliased removes any chance of a read/write hazard between this
    // new path and the existing, validated one. sb.moe_logits/moe_topk_row_sum
    // ARE reused (routing math is identical either way: softmax->top-k->
    // renormalize, moe_topk_route_f32 as-is, no pooling/pressure/lateral-
    // inhibition inputs for this mode).
    float* generic_moe_hidden_f32{};    // [B*S, H] cast of the FFN-norm'd hidden (sb.normed2), router+expert input
    float* generic_moe_gate_f32{};      // [B*S, max(1,Ie)] per-expert gate_proj output, reused across the expert loop
    float* generic_moe_up_f32{};        // [B*S, max(1,Ie)] per-expert up_proj output, reused across the expert loop
    __nv_bfloat16* generic_moe_gate_bf16{}; // [B*S, max(1,Ie)] cast of generic_moe_gate_f32
    __nv_bfloat16* generic_moe_up_bf16{};   // [B*S, max(1,Ie)] cast of generic_moe_up_f32
    __nv_bfloat16* generic_moe_h_bf16{};    // [B*S, max(1,Ie)] swiglu_forward(gate,up) output
    float* generic_moe_h_f32{};          // [B*S, max(1,Ie)] cast of generic_moe_h_bf16, feeds down_proj
    float* generic_moe_down_f32{};       // [B*S, H] per-expert down_proj output, before route-weight scaling
    __nv_bfloat16* generic_moe_down_bf16{}; // [B*S, H] cast of generic_moe_down_f32; accumulated directly
                                             // into sb.ffn_out via moe_scale_accumulate_bf16, same as legacy's
                                             // moe_expert_out -- no separate accumulator needed
    // Backward-only recompute scratch (mirrors the forward intermediates
    // above 1:1 so backward can recompute rather than save every expert's
    // activations -- same recompute-based-backward convention as the rest
    // of this engine).
    float* generic_moe_d_hidden_f32{};  // [B*S, H] accumulator: d(hidden) summed over all experts + router
    float* generic_moe_d_gate_f32{};    // [B*S, max(1,Ie)] scratch
    float* generic_moe_d_up_f32{};      // [B*S, max(1,Ie)] scratch
    __nv_bfloat16* generic_moe_d_gate_bf16{}; // [B*S, max(1,Ie)] scratch (swiglu_backward output)
    __nv_bfloat16* generic_moe_d_up_bf16{};   // [B*S, max(1,Ie)] scratch (swiglu_backward output)
    __nv_bfloat16* generic_moe_d_h_bf16{};    // [B*S, max(1,Ie)] scratch: d(swiglu_out), cast target for d_h_f32
    float* generic_moe_d_h_f32{};        // [B*S, max(1,Ie)] scratch: d(down_proj input)
    float* generic_moe_d_logits{};       // [B*S, num_experts] scratch: d(route weights) -> through topk/softmax backward -> d(router logits)
    float* generic_moe_d_hidden_contrib_f32{}; // [B*S, H] transient: moe_small_proj_backward_input OVERWRITES (does not
                                                // accumulate, unlike its _backward_weight sibling), so each expert's
                                                // gate/up/router contribution lands here first, then add_inplace_f32
                                                // sums it into generic_moe_d_hidden_f32

    // Shared expert (2026-08-23). Reuses every generic_moe_gate/up/h/down_*
    // buffer above -- they're sized to max(expert_intermediate_size,
    // shared_expert_width) precisely so the shared expert's single (no-loop)
    // FFN pass can run through the same scratch right after (forward) or
    // as part of (backward) the routed-expert loop, same "already scratch,
    // reused per iteration" convention those buffers use for N>1 experts.
    // Only the per-token scalar gate needs its own buffers.
    float* generic_moe_shared_gate_scalar_f32{}; // [B*S] sigmoid(gate_score(hidden)) -- saved for backward
    float* generic_moe_shared_dgate_scalar_f32{}; // [B*S] transient: d(post-sigmoid) from row-dot, then
                                                   // overwritten in place with d(pre-sigmoid) via sigmoid_backward_f32

    // Native packed-FP4 activation path (real storage reduction on top of
    // the existing FP8 GEMM, IDA_NATIVE_MOE_ACT_PRECISION=fp4_native).
    // Packed storage is HALF the bytes of the fp8 scratch it replaces;
    // decode scratch is fp8-sized (matches what lt_gemm_fp8_nt already
    // expects, so no GEMM-side change needed). Own scratch, NOT f8.act8 --
    // this mode must work even when the surrounding profile has its own
    // separate f8.act8 usage mid-flight for other tensors in the same step.
    // Calibration state (amax/scale/descale/scale_snapshot/descale_snapshot)
    // is NOT duplicated here -- reuses the SAME Fp8Ctx ActSlot instances
    // (moe_fcin_act_slot/moe_fcout_act_slot) the direct-FP8 path already
    // has, since the two modes are mutually exclusive per run (env-var
    // selected) and never touch a slot in the same run.
    std::uint8_t* moe_fp4_packed_fcin{};     // [B*S*H/2] bytes
    std::uint8_t* moe_fp4_packed_fcout{};    // [B*S*max(I,Ie)/2] bytes
    __nv_fp8_e4m3* moe_fp4_decode_scratch{}; // [B*S*max(H,I,Ie)] fp8 -- feeds lt_gemm_fp8_nt directly

    // LRSS hook points (null = LRSS off).  Owned by run_lattice_training;
    // forward()/backward_accumulate() only consume.
    const LrssParams* lrss_p{};
    LrssScratch*      lrss_s{};
    LrssGrads*        lrss_g{};
    std::vector<std::uint8_t> lss_feedback_skip_layer; // [L], true = feedback-scaled tail layer
    int lss_feedback_skip_tail{0};
    int lss_feedback_completed_optimizer_steps{0};

    // Fused chunked lm_head + CE (never materializes [B*S, V])
    __nv_bfloat16* logits_chunk{}; // [ce_chunk, V]  logits → grad in place
    int*           n_valid{};      // [1]
    float*         loss{};         // [1]
    int            ce_chunk{0};

    // Gradient backflow
    __nv_bfloat16* d_hidden{};     // [B*S, H]
    __nv_bfloat16* d_hidden_feedback{}; // [B*S, H] scaled FFN branch for LSS feedback
    __nv_bfloat16* d_normed{};     // [B*S, H]

    // FP32 scratch
    float*         g_scale{};      // [H]  per-call norm-scale gradient
    float*         norm_bias_scale{}; // [H] LayerNorm beta-gradient scratch
    float*         norm_acc{};     // [1]  global grad-norm accumulator
    float*         grad_update_scratch{}; // BF16 gradient -> FP32 optimizer staging
    std::size_t    grad_update_scratch_n{0};
    float*         embed_clip_stats{}; // [2]  clipped_rows, max_preclip_row_norm

    // PSS Stage 2 (null = IDA_NATIVE_PSS_PRED off). Shadow-mode low-rank
    // predictor of the tail layer's real FFN output (sb.ffn_out at
    // l==num_layers-1), scored but never blended into the body -- Stage 4
    // owns any actual body-consumption path.
    __nv_bfloat16* pss_pred_hidden{};    // [B*S, R]  relu(pred_down · pre_tail_hidden)
    __nv_bfloat16* pss_pred_ffn{};       // [B*S, H]  predicted tail FFN output
    __nv_bfloat16* d_pss_pred_ffn{};     // [B*S, H]  aux-loss gradient seed
    __nv_bfloat16* d_pss_pred_hidden{};  // [B*S, R]  backward scratch
    float*         pss_fence_metrics{};  // [8] contiguous scalar readback surface
    float*         pss_pred_err_sq{};    // alias: pss_fence_metrics[0]
    float*         pss_pred_target_sq{}; // alias: pss_fence_metrics[1]
    float*         pss_pred_covered{};   // alias: pss_fence_metrics[2]
    float*         pss_int2_inv_rms{};   // alias: pss_fence_metrics[5], delayed real-target scale
    float*         pss_int2_matched{};   // alias: pss_fence_metrics[3], uint32 count
    float*         pss_int2_scored{};    // alias: pss_fence_metrics[4], uint32 count
    float*         pss_confidence_minmax{}; // alias: pss_fence_metrics[6:8], current window range

    // Per-block int2 symbol scale (2026-08-13, ablation instrument -- see
    // k_pss_pred_score/k_pss_update_int2_inv_rms). SEPARATE allocation from
    // pss_pred_target_sq deliberately: that scalar also feeds aux-loss
    // normalization (a training-relevant consumer), so it stays untouched;
    // these two arrays exist only to let int2_agreement be scored against a
    // per-block scale instead of one global scale, borrowing MXFP4/NVFP4's
    // block-scaled-quantization convention as the hypothesis under test, not
    // an assumed-correct design. pss_int2_num_blocks == 1 (block_size == 0,
    // the default) reproduces the prior single-global-scale behavior exactly.
    float*         pss_int2_target_sq_blocks{};
    float*         pss_int2_inv_rms_blocks{};
    int            pss_int2_num_blocks{1};

    // Magnitude-bucketed err/hit-rate diagnostic (2026-08-15) -- see
    // pss_mag_bucket/k_pss_pred_score. Separate allocation from
    // pss_int2_target_sq_blocks for the same reason that one is separate
    // from pss_pred_target_sq: different question (is error uniform across
    // the |real| distribution?), fixed kPssMagBuckets size regardless of H.
    float*         pss_mag_bucket_err_sq{};
    float*         pss_mag_bucket_target_sq{};
    float*         pss_mag_bucket_covered{};
    float*         pss_mag_bucket_count{};

    // Decoupled PSS optimizer cadence (2026-08-14, IDA_NATIVE_PSS_ACCUM_STEPS).
    // The trunk's grad-accum window (accum_at()/--grad-accum-override) is a
    // process-wide single value shared by every slot; there is no way to give
    // PSS a LARGER effective accumulation window than the trunk without this.
    // Design: each trunk optimizer step's ALREADY-AVERAGED PSS gradient
    // (g_pss_pred_down/up, post scale_slots) gets added into these persistent
    // accumulators BEFORE the normal per-slot optimizer loop runs; the PSS
    // Lion update itself is applied only every pss_accum_steps() trunk steps,
    // using accumulator/window_count (a second, independent average on top of
    // the trunk's own per-window average) -- skipped trunk-steps leave PSS's
    // weights and Lion momentum completely untouched. This does NOT touch the
    // existing zero-gradients-every-window loop or scale_slots at all: p.g
    // itself is consumed into the accumulator before the normal window ends,
    // so the trunk's own per-window averaging is never double-applied.
    // pss_accum_steps()==1 (default) reproduces today's behavior exactly:
    // window_count reaches 1 every single trunk step, so the "accumulate
    // then immediately apply" path fires every time, identical to applying
    // p.g directly.
    float*         pss_window_accum_down{};
    float*         pss_window_accum_up{};
    int            pss_window_count{0};

    // PSS Stage 4 (IDA_NATIVE_PSS_GOVERNOR, default off): governor-owned
    // engagement fraction in [0,1]. Run-scoped, not checkpointed -- a
    // resumed run re-earns engagement from 0, same as the LSS feedback
    // controller's completed_optimizer_steps gate. Set once per optimizer
    // step at the fence (mirrors sb.lss_feedback_skip_layer's pattern: host
    // state computed in the main loop, read by forward()/backward_
    // accumulate() every micro-step of the following window).
    float pss_engaged_frac{0.0f};
    float support_transition_scale{1.0f};
    // Host-side evidence set by the true-forward PSS hook / backward hook.
    // These are deliberately not checkpoint state.
    int pss_scored_micros{0};
    float pss_last_blend_frac{0.0f};
    float pss_aux_weight_used{0.0f};
    bool pss_aux_normalize_used{false};
    bool pss_conditioning_active{false};
};

static bool moe_native_fp4_enabled();

static StepBuffers alloc_step_buffers(
    const LatticeWeights& w, int B, int S, NativeArena& arena
) {
    const int H  = w.hidden_size;
    const int I  = w.intermediate_size;
    const int V  = w.vocab_size;
    const int nH = w.heads;
    const int Hd = H / nH;
    const int L  = w.num_layers;
    const std::size_t BS = static_cast<std::size_t>(B) * S;

    StepBuffers sb{};
    sb.B = B; sb.S = S; sb.H = H; sb.I = I; sb.V = V; sb.nH = nH; sb.Hd = Hd;
    sb.kvH = (w.kv_heads > 0 ? w.kv_heads : w.heads) * Hd;
    sb.mb_alloc = B;
    sb.lss_feedback_skip_layer.assign(static_cast<std::size_t>(L), 0);

    sb.hidden      = alloc_bf16(BS * H,          arena);
    sb.normed      = alloc_bf16(BS * H,          arena);
    sb.normed2     = alloc_bf16(BS * H,          arena);
    sb.rms_save    = alloc_f32 (BS,              arena);
    sb.rms_a       = alloc_f32 (BS,              arena);
    sb.saved_slots = L + 1;
    {
        // Era 13 default: ON (-15.8 GB on the AI body at zero measured tps
        // cost).  =0 restores full device residency.
        const char* off_env = std::getenv("IDA_NATIVE_SAVED_ACT_HOST_OFFLOAD");
        if (!off_env || off_env[0] == '1') {
            const char* ring_env = std::getenv("IDA_NATIVE_SAVED_ACT_RING");
            int ring = ring_env ? std::atoi(ring_env) : 4;
            ring = std::min(std::max(ring, 3), L + 1);
            sb.saved_ring = ring;
            IDA_CUDA_CHECK(cudaHostAlloc(
                &sb.saved_host,
                static_cast<std::size_t>(L + 1) * BS * H * sizeof(__nv_bfloat16),
                cudaHostAllocDefault));
            IDA_CUDA_CHECK(cudaStreamCreateWithFlags(
                &sb.saved_copy_stream, cudaStreamNonBlocking));
            sb.saved_ev_release = new cudaEvent_t[ring];
            sb.saved_ev_copied  = new cudaEvent_t[ring];
            for (int r = 0; r < ring; ++r) {
                IDA_CUDA_CHECK(cudaEventCreateWithFlags(
                    &sb.saved_ev_release[r], cudaEventDisableTiming));
                IDA_CUDA_CHECK(cudaEventCreateWithFlags(
                    &sb.saved_ev_copied[r], cudaEventDisableTiming));
            }
            std::fprintf(stderr,
                "[ida_native_train] saved-activation host offload: ring=%d, "
                "pinned %.1f MiB, device %.1f MiB (was %.1f MiB)\n",
                ring,
                static_cast<double>(L + 1) * BS * H * 2.0 / (1024.0 * 1024.0),
                static_cast<double>(ring) * BS * H * 2.0 / (1024.0 * 1024.0),
                static_cast<double>(L + 1) * BS * H * 2.0 / (1024.0 * 1024.0));
        }
    }
    sb.saved = alloc_bf16(
        static_cast<std::size_t>(sb.saved_ring > 0 ? sb.saved_ring : L + 1)
            * BS * H, arena);
    sb.qkv         = alloc_bf16(BS * 3 * H,      arena);
    sb.q_f32       = alloc_bf16(BS * H,          arena);
    sb.k_f32       = alloc_bf16(BS * H,          arena);
    sb.v_f32       = alloc_bf16(BS * H,          arena);
    sb.o_f32       = alloc_bf16(BS * H,          arena);
    sb.do_f32      = alloc_bf16(BS * H,          arena);
    sb.dq_f32      = alloc_bf16(BS * H,          arena);
    sb.dk_f32      = alloc_bf16(BS * H,          arena);
    sb.dv_f32      = alloc_bf16(BS * H,          arena);
    sb.lse         = alloc_f32 (BS * nH,         arena);
    sb.rowdot      = alloc_f32 (BS * nH,         arena);
    sb.attn_out    = alloc_bf16(BS * H,          arena);
    sb.o_out       = alloc_bf16(BS * H,          arena);
    bool has_dense_ffn = false;
    for (int l = 0; l < L; ++l) {
        if (w.layers[l].num_experts <= 0) {
            has_dense_ffn = true;
            break;
        }
    }
    // A routed MoE layer returns from layer_forward_body before gate/up/
    // SwiGLU and takes the MoE-specific backward branch. Pure MoE shards
    // therefore do not need these four [B*S,I] workspaces; retaining them
    // doubled a 256 MiB dead footprint for the 1F1B stage-0 slot.
    if (has_dense_ffn) {
        sb.gate_out   = alloc_bf16(BS * I,          arena);
        sb.up_out     = alloc_bf16(BS * I,          arena);
        sb.swiglu_out = alloc_bf16(BS * I,          arena);
        sb.d_ffn_i    = alloc_bf16(BS * I,          arena);
    }
    sb.ffn_out     = alloc_bf16(BS * H,          arena);
    // Cognitive-architecture sparse MoE port: gated on num_experts>0, taken
    // from layer 0 since allocate_lattice_weights sizes every MoE-enabled
    // layer uniformly within one request (not a per-layer-varying config).
    if (L > 0 && w.layers[0].num_experts > 0 && w.layers[0].moe_kind == 0) {
        const int num_routes = w.layers[0].num_routes;
        const int num_experts = w.layers[0].num_experts;
        sb.moe_pool_scratch  = alloc_f32(BS * H, arena);
        sb.moe_count_scratch = alloc_f32(BS, arena);
        sb.moe_pooled        = alloc_f32(BS * H, arena);
        sb.moe_pressure      = alloc_f32(BS * static_cast<std::size_t>(num_routes), arena);
        sb.moe_modulation    = alloc_f32(BS * H, arena);
        sb.moe_hidden_gated  = alloc_bf16(BS * H, arena);
        sb.moe_pooled2       = alloc_f32(BS * H, arena);
        if (num_routes != num_experts) {
            sb.moe_routed_pressure = alloc_f32(BS * static_cast<std::size_t>(num_experts), arena);
        }
        sb.moe_logits = alloc_f32(BS * static_cast<std::size_t>(num_experts), arena);
        const int expert_scratch_w = std::max(I, w.layers[0].expert_intermediate_size);
        sb.moe_expert_scratch = alloc_bf16(BS * static_cast<std::size_t>(expert_scratch_w), arena);
        sb.moe_expert_out     = alloc_bf16(BS * H, arena);
        sb.moe_dispatch_counts = alloc_i32(num_experts, arena);
        sb.moe_dispatch_indices = alloc_i32(BS * static_cast<std::size_t>(num_experts), arena);
        sb.moe_dispatch_offsets = alloc_i32(num_experts, arena);
        sb.moe_dispatch_counts_saved = alloc_i32(static_cast<std::size_t>(L) * num_experts, arena);
        sb.moe_dispatch_indices_saved = alloc_i32(static_cast<std::size_t>(L) * num_experts * BS, arena);
        sb.moe_dispatch_counts_host.assign(static_cast<std::size_t>(num_experts), 0);
        sb.moe_dispatch_counts_host_saved.assign(static_cast<std::size_t>(L) * num_experts, 0);

        sb.moe_count              = alloc_f32(BS, arena);
        sb.moe_start_grad_scratch = alloc_f32(BS * H, arena);
        sb.moe_d_hidden_gated     = alloc_bf16(BS * H, arena);
        sb.moe_d_pooled           = alloc_f32(BS * H, arena);
        sb.moe_d_pooled2          = alloc_f32(BS * H, arena);
        sb.moe_d_pressure         = alloc_f32(BS * static_cast<std::size_t>(num_routes), arena);
        sb.moe_d_pressure_contrib2 = alloc_f32(BS * static_cast<std::size_t>(num_routes), arena);
        sb.moe_d_modulation       = alloc_f32(BS * H, arena);
        sb.moe_d_logits           = alloc_f32(BS * static_cast<std::size_t>(num_experts), arena);
        if (num_routes != num_experts) {
            sb.moe_d_routed_pressure = alloc_f32(BS * static_cast<std::size_t>(num_experts), arena);
        }
        sb.moe_topk_row_sum  = alloc_f32(BS, arena);
        sb.moe_scores_save   = alloc_f32(BS * static_cast<std::size_t>(num_experts), arena);
        sb.moe_topk_p_save   = alloc_f32(BS * static_cast<std::size_t>(num_experts), arena);
        sb.moe_inhib_c1      = alloc_f32(BS * static_cast<std::size_t>(num_experts), arena);
        sb.moe_inhib_row_sum = alloc_f32(BS, arena);
        sb.moe_expert_scratch2 = alloc_bf16(BS * static_cast<std::size_t>(expert_scratch_w), arena);
        sb.moe_d_expert_scratch = alloc_bf16(BS * static_cast<std::size_t>(expert_scratch_w), arena);
        sb.moe_probe_amax = alloc_f32(1, arena);

        // The native FP4 activation path is opt-in.  Do not reserve its
        // packed/decode workspaces for the normal FP8-weight/BF16-activation
        // profile; on a dual 3090 the second 1F1B slot needs every free MiB.
        if (moe_native_fp4_enabled()) {
            const std::size_t fp4_scratch_w = static_cast<std::size_t>(std::max(H, expert_scratch_w));
            sb.moe_fp4_packed_fcin    = alloc_u8(BS * static_cast<std::size_t>(H) / 2, arena);
            sb.moe_fp4_packed_fcout   = alloc_u8(BS * static_cast<std::size_t>(expert_scratch_w) / 2, arena);
            sb.moe_fp4_decode_scratch = alloc_fp8_e4m3(BS * fp4_scratch_w, arena);
        }
    }
    // Generic per-token top-k SwiGLU MoE (2026-08-23). Own dedicated scratch,
    // never allocated alongside the legacy block above (moe_kind is uniform
    // per body, same assumption the legacy block already makes about
    // num_experts/num_routes at layer 0).
    if (L > 0 && w.layers[0].num_experts > 0 && w.layers[0].moe_kind == 1) {
        const int num_experts = w.layers[0].num_experts;
        const std::size_t Ie = static_cast<std::size_t>(
            std::max(w.layers[0].expert_intermediate_size, w.layers[0].shared_expert_width));
        sb.moe_logits = alloc_f32(BS * static_cast<std::size_t>(num_experts), arena);
        sb.moe_topk_row_sum = alloc_f32(BS, arena);
        sb.moe_scores_save = alloc_f32(BS * static_cast<std::size_t>(num_experts), arena);
        sb.generic_moe_hidden_f32 = alloc_f32(BS * static_cast<std::size_t>(H), arena);
        sb.generic_moe_gate_f32 = alloc_f32(BS * Ie, arena);
        sb.generic_moe_up_f32 = alloc_f32(BS * Ie, arena);
        sb.generic_moe_gate_bf16 = alloc_bf16(BS * Ie, arena);
        sb.generic_moe_up_bf16 = alloc_bf16(BS * Ie, arena);
        sb.generic_moe_h_bf16 = alloc_bf16(BS * Ie, arena);
        sb.generic_moe_h_f32 = alloc_f32(BS * Ie, arena);
        sb.generic_moe_down_f32 = alloc_f32(BS * static_cast<std::size_t>(H), arena);
        sb.generic_moe_down_bf16 = alloc_bf16(BS * static_cast<std::size_t>(H), arena);
        sb.generic_moe_d_hidden_f32 = alloc_f32(BS * static_cast<std::size_t>(H), arena);
        sb.generic_moe_d_gate_f32 = alloc_f32(BS * Ie, arena);
        sb.generic_moe_d_up_f32 = alloc_f32(BS * Ie, arena);
        sb.generic_moe_d_gate_bf16 = alloc_bf16(BS * Ie, arena);
        sb.generic_moe_d_up_bf16 = alloc_bf16(BS * Ie, arena);
        sb.generic_moe_d_h_bf16 = alloc_bf16(BS * Ie, arena);
        sb.generic_moe_d_h_f32 = alloc_f32(BS * Ie, arena);
        sb.generic_moe_d_logits = alloc_f32(BS * static_cast<std::size_t>(num_experts), arena);
        sb.generic_moe_d_hidden_contrib_f32 = alloc_f32(BS * static_cast<std::size_t>(H), arena);
        if (w.layers[0].shared_expert_width > 0) {
            sb.generic_moe_shared_gate_scalar_f32 = alloc_f32(BS, arena);
            sb.generic_moe_shared_dgate_scalar_f32 = alloc_f32(BS, arena);
        }
    }
    // The lower peer stage never evaluates the final norm, LM head, or
    // cross-entropy. Keeping its CE chunk would needlessly duplicate a
    // vocab-sized buffer and defeat part of the model-parallel memory win.
    if (w.owns_output) {
        sb.ce_chunk = static_cast<int>(std::min<std::size_t>(BS, 4096));
        sb.logits_chunk = alloc_bf16(static_cast<std::size_t>(sb.ce_chunk) * V, arena);
        sb.loss = alloc_f32(1, arena);
        IDA_CUDA_CHECK(ida_malloc_async(&sb.n_valid, sizeof(int), arena.pool, arena.stream));
    }
    sb.d_hidden    = alloc_bf16(BS * H,          arena);
    sb.d_hidden_feedback = alloc_bf16(BS * H,    arena);
    sb.d_normed    = alloc_bf16(BS * H,          arena);
    sb.g_scale     = alloc_f32 (H,               arena);
    sb.norm_acc    = alloc_f32 (1,               arena);
    // BF16 layer gradients are staged into FP32 only at the optimizer
    // boundary.  Keep this a dense-matrix-sized reusable buffer and process
    // larger MoE expert tensors in chunks; allocating an entire expert bank
    // here would erase the memory saved by BF16 gradients on 24 GB cards.
    sb.grad_update_scratch_n = std::max<std::size_t>({
        static_cast<std::size_t>(H) * H,
        static_cast<std::size_t>(H) * I,
        static_cast<std::size_t>(I) * H
    });
    sb.grad_update_scratch = alloc_f32(sb.grad_update_scratch_n, arena);
    sb.embed_clip_stats = alloc_f32(2,           arena);
    if (w.pss_pred_rank > 0) {
        const std::size_t R = w.pss_pred_rank;
        sb.pss_pred_hidden   = alloc_bf16(BS * R, arena);
        sb.pss_pred_ffn      = alloc_bf16(BS * H, arena);
        sb.d_pss_pred_ffn    = alloc_bf16(BS * H, arena);
        sb.d_pss_pred_hidden = alloc_bf16(BS * R, arena);
        sb.pss_fence_metrics  = alloc_f32(8, arena);
        sb.pss_pred_err_sq    = sb.pss_fence_metrics + 0;
        sb.pss_pred_target_sq = sb.pss_fence_metrics + 1;
        sb.pss_pred_covered   = sb.pss_fence_metrics + 2;
        sb.pss_int2_inv_rms   = sb.pss_fence_metrics + 5;
        sb.pss_int2_matched   = sb.pss_fence_metrics + 3;
        sb.pss_int2_scored    = sb.pss_fence_metrics + 4;
        sb.pss_confidence_minmax = sb.pss_fence_metrics + 6;
        IDA_CUDA_CHECK(cudaMemsetAsync(sb.pss_int2_inv_rms, 0, sizeof(float), arena.stream));
        k_pss_reset_confidence_minmax<<<1, 1, 0, arena.stream>>>(
            sb.pss_confidence_minmax);

        const int int2_block_size = pss_int2_block_size();
        sb.pss_int2_num_blocks = int2_block_size > 0
            ? std::max(1, H / int2_block_size) : 1;
        sb.pss_int2_target_sq_blocks = alloc_f32(
            static_cast<std::size_t>(sb.pss_int2_num_blocks), arena);
        sb.pss_int2_inv_rms_blocks = alloc_f32(
            static_cast<std::size_t>(sb.pss_int2_num_blocks), arena);
        IDA_CUDA_CHECK(cudaMemsetAsync(sb.pss_int2_inv_rms_blocks, 0,
            static_cast<std::size_t>(sb.pss_int2_num_blocks) * sizeof(float), arena.stream));

        sb.pss_mag_bucket_err_sq = alloc_f32(kPssMagBuckets, arena);
        sb.pss_mag_bucket_target_sq = alloc_f32(kPssMagBuckets, arena);
        sb.pss_mag_bucket_covered = alloc_f32(kPssMagBuckets, arena);
        sb.pss_mag_bucket_count = alloc_f32(kPssMagBuckets, arena);

        sb.pss_window_accum_down = alloc_f32(H * R, arena);
        sb.pss_window_accum_up   = alloc_f32(H * R, arena);
        IDA_CUDA_CHECK(cudaMemsetAsync(sb.pss_window_accum_down, 0,
            H * R * sizeof(float), arena.stream));
        IDA_CUDA_CHECK(cudaMemsetAsync(sb.pss_window_accum_up, 0,
            H * R * sizeof(float), arena.stream));
        sb.pss_window_count = 0;
    }
    return sb;
}

static void free_step_buffers(StepBuffers& sb, NativeArena& arena) {
    auto f = [&](auto* p) { if (p) IDA_CUDA_CHECK(cudaFreeAsync(p, arena.stream)); };
    f(sb.hidden); f(sb.normed); f(sb.normed2); f(sb.rms_save); f(sb.rms_a);
    f(sb.saved);
    if (sb.saved_ring > 0) {
        IDA_CUDA_CHECK(cudaStreamSynchronize(sb.saved_copy_stream));
        for (int r = 0; r < sb.saved_ring; ++r) {
            IDA_CUDA_CHECK(cudaEventDestroy(sb.saved_ev_release[r]));
            IDA_CUDA_CHECK(cudaEventDestroy(sb.saved_ev_copied[r]));
        }
        delete[] sb.saved_ev_release;
        delete[] sb.saved_ev_copied;
        IDA_CUDA_CHECK(cudaStreamDestroy(sb.saved_copy_stream));
        IDA_CUDA_CHECK(cudaFreeHost(sb.saved_host));
    }
    f(sb.qkv); f(sb.q_f32); f(sb.k_f32); f(sb.v_f32);
    f(sb.o_f32); f(sb.do_f32); f(sb.dq_f32); f(sb.dk_f32); f(sb.dv_f32);
    f(sb.lse); f(sb.rowdot); f(sb.attn_out); f(sb.o_out);
    f(sb.gate_out); f(sb.up_out); f(sb.swiglu_out); f(sb.d_ffn_i); f(sb.ffn_out);
    f(sb.logits_chunk); f(sb.loss); f(sb.n_valid);
    f(sb.d_hidden); f(sb.d_hidden_feedback); f(sb.d_normed);
    f(sb.g_scale); f(sb.norm_bias_scale); f(sb.norm_acc); f(sb.grad_update_scratch); f(sb.embed_clip_stats);
    f(sb.pss_pred_hidden); f(sb.pss_pred_ffn); f(sb.d_pss_pred_ffn); f(sb.d_pss_pred_hidden);
    f(sb.pss_fence_metrics);
}

// Pointer to the saved input of layer l (slot L = final-norm input).  Under
// host offload this is the layer's ring slot — callers must respect the
// ring's event protocol (saved_fwd_stash / saved_bwd_fetch below).
static __nv_bfloat16* saved_slot(StepBuffers& sb, int l) {
    const std::size_t slice = static_cast<std::size_t>(sb.mb_alloc) * sb.S * sb.H;
    if (sb.saved_ring > 0) return sb.saved + static_cast<std::size_t>(l % sb.saved_ring) * slice;
    return sb.saved + static_cast<std::size_t>(l) * slice;
}

static __nv_bfloat16* saved_host_slot(StepBuffers& sb, int l) {
    return sb.saved_host
        + static_cast<std::size_t>(l) * sb.mb_alloc * sb.S * sb.H;
}

// Forward-side ring protocol: call BEFORE the D2D write into saved_slot(l)
// (waits for the slot's previous copy to land), and AFTER it (publishes the
// slice to the copy stream, which drains it to pinned DDR5).
static void saved_fwd_stash_begin(StepBuffers& sb, int l, cudaStream_t s) {
    if (sb.saved_ring <= 0) return;
    const int r = l % sb.saved_ring;
    IDA_CUDA_CHECK(cudaStreamWaitEvent(s, sb.saved_ev_copied[r], 0));
}
static void saved_fwd_stash_end(StepBuffers& sb, int l, cudaStream_t s) {
    if (sb.saved_ring <= 0) return;
    const int r = l % sb.saved_ring;
    const std::size_t bytes = static_cast<std::size_t>(sb.B) * sb.S * sb.H
                              * sizeof(__nv_bfloat16);
    IDA_CUDA_CHECK(cudaEventRecord(sb.saved_ev_release[r], s));
    IDA_CUDA_CHECK(cudaStreamWaitEvent(sb.saved_copy_stream,
                                       sb.saved_ev_release[r], 0));
    IDA_CUDA_CHECK(cudaMemcpyAsync(saved_host_slot(sb, l), saved_slot(sb, l),
                                   bytes, cudaMemcpyDeviceToHost,
                                   sb.saved_copy_stream));
    IDA_CUDA_CHECK(cudaEventRecord(sb.saved_ev_copied[r],
                                   sb.saved_copy_stream));
}

// Backward-side ring protocol: fetch_begin waits until slot l's content is
// present (either still resident from the forward tail or restored by the
// prefetch issued when the slot's previous backward tenant released it).
// fetch_end marks every read of h_in enqueued, then immediately queues the
// H2D prefetch for this slot's NEXT backward tenant (layer l - ring), giving
// the copy stream a ring-1 layer head start over the compute.
static void saved_bwd_fetch_begin(StepBuffers& sb, int l, cudaStream_t s) {
    if (sb.saved_ring <= 0) return;
    const int r = l % sb.saved_ring;
    IDA_CUDA_CHECK(cudaStreamWaitEvent(s, sb.saved_ev_copied[r], 0));
}
static void saved_bwd_fetch_end(StepBuffers& sb, int l, cudaStream_t s) {
    if (sb.saved_ring <= 0) return;
    const int r = l % sb.saved_ring;
    IDA_CUDA_CHECK(cudaEventRecord(sb.saved_ev_release[r], s));
    const int prev = l - sb.saved_ring;
    if (prev < 0) return;
    const std::size_t bytes = static_cast<std::size_t>(sb.B) * sb.S * sb.H
                              * sizeof(__nv_bfloat16);
    IDA_CUDA_CHECK(cudaStreamWaitEvent(sb.saved_copy_stream,
                                       sb.saved_ev_release[r], 0));
    IDA_CUDA_CHECK(cudaMemcpyAsync(saved_slot(sb, prev),
                                   saved_host_slot(sb, prev), bytes,
                                   cudaMemcpyHostToDevice,
                                   sb.saved_copy_stream));
    IDA_CUDA_CHECK(cudaEventRecord(sb.saved_ev_copied[r],
                                   sb.saved_copy_stream));
}

static PackedFp4AttentionCtx build_packed_fp4_attention_ctx(
    const NativeRequest& request,
    const StepBuffers& sb,
    int num_layers,
    NativeArena& arena
) {
    constexpr std::size_t kHopperTmaStageTileElems = 2u * 64u * 64u;
    PackedFp4AttentionCtx ctx{};
    ctx.on = precision_profile_uses_wgmma_attention_scratch(request);
    if (!ctx.on) {
        return ctx;
    }
    ctx.num_layers = num_layers;
    const std::size_t nl = static_cast<std::size_t>(num_layers);

    ctx.packed_elems = static_cast<std::size_t>(sb.B) * sb.S * sb.H;
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.qk_packed, ctx.packed_elems, arena.pool, arena.stream));
    if (packed_fp4_backward_replay_saved_e4m3_enabled()) {
        IDA_CUDA_CHECK(ida_malloc_async(
            &ctx.q_saved_e4m3, nl * ctx.packed_elems, arena.pool, arena.stream));
        IDA_CUDA_CHECK(ida_malloc_async(
            &ctx.k_saved_e4m3, nl * ctx.packed_elems, arena.pool, arena.stream));
    }
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.q_amax, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.q_scale, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.q_descale, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.k_amax, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.k_scale, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.k_descale, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.q_mean, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.k_mean, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.q_sum, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.k_sum, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.q_max_abs_dev, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.k_max_abs_dev, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.q_mean_snapshot, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.k_mean_snapshot, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.q_scale_snapshot, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.k_scale_snapshot, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.q_descale_snapshot, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.k_descale_snapshot, nl * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.q_unpack_f32, ctx.packed_elems * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.k_unpack_f32, ctx.packed_elems * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.q_tile_stage_f32, kHopperTmaStageTileElems * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.k_tile_stage_f32, kHopperTmaStageTileElems * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&ctx.v_tile_stage_f32, kHopperTmaStageTileElems * sizeof(float), arena.pool, arena.stream));
    ctx.tile_stage_elems = kHopperTmaStageTileElems;
    ctx.tma_stage_depth = 2;

    std::vector<float> ones(nl, 1.0f);
    IDA_CUDA_CHECK(cudaMemcpyAsync(
        ctx.q_scale, ones.data(), nl * sizeof(float), cudaMemcpyHostToDevice, arena.stream));
    IDA_CUDA_CHECK(cudaMemcpyAsync(
        ctx.q_descale, ones.data(), nl * sizeof(float), cudaMemcpyHostToDevice, arena.stream));
    IDA_CUDA_CHECK(cudaMemcpyAsync(
        ctx.k_scale, ones.data(), nl * sizeof(float), cudaMemcpyHostToDevice, arena.stream));
    IDA_CUDA_CHECK(cudaMemcpyAsync(
        ctx.k_descale, ones.data(), nl * sizeof(float), cudaMemcpyHostToDevice, arena.stream));
    IDA_CUDA_CHECK(cudaMemsetAsync(ctx.q_amax, 0, nl * sizeof(float), arena.stream));
    IDA_CUDA_CHECK(cudaMemsetAsync(ctx.k_amax, 0, nl * sizeof(float), arena.stream));
    // Centered mode starts at mean=0 (first call behaves like plain int4
    // until its own data informs the next mean); sum/max_abs_dev are pure
    // per-call scratch, zeroed by fp4_pack_pair_centered_record itself.
    IDA_CUDA_CHECK(cudaMemsetAsync(ctx.q_mean, 0, nl * sizeof(float), arena.stream));
    IDA_CUDA_CHECK(cudaMemsetAsync(ctx.k_mean, 0, nl * sizeof(float), arena.stream));
    // Snapshots start matching the live initial state (mean=0, scale/descale=1)
    // so the very first call's decode (before any snapshot has been taken by
    // a true-forward pass) is well-defined.
    IDA_CUDA_CHECK(cudaMemsetAsync(ctx.q_mean_snapshot, 0, nl * sizeof(float), arena.stream));
    IDA_CUDA_CHECK(cudaMemsetAsync(ctx.k_mean_snapshot, 0, nl * sizeof(float), arena.stream));
    IDA_CUDA_CHECK(cudaMemcpyAsync(
        ctx.q_scale_snapshot, ones.data(), nl * sizeof(float), cudaMemcpyHostToDevice, arena.stream));
    IDA_CUDA_CHECK(cudaMemcpyAsync(
        ctx.q_descale_snapshot, ones.data(), nl * sizeof(float), cudaMemcpyHostToDevice, arena.stream));
    IDA_CUDA_CHECK(cudaMemcpyAsync(
        ctx.k_scale_snapshot, ones.data(), nl * sizeof(float), cudaMemcpyHostToDevice, arena.stream));
    IDA_CUDA_CHECK(cudaMemcpyAsync(
        ctx.k_descale_snapshot, ones.data(), nl * sizeof(float), cudaMemcpyHostToDevice, arena.stream));
    return ctx;
}

// Legacy environment switches are retained only for request compatibility.
// The public target never activates private packed-precision implementations.
static bool fp4_e2m1_enabled() {
    const char* e = std::getenv("IDA_NATIVE_FP4_E2M1");
    return e && e[0] == '1';
}

static bool fp4_int2_enabled() {
    const char* e = std::getenv("IDA_NATIVE_FP4_INT2");
    return e && e[0] == '1';
}

static bool fp4_gauss_enabled() {
    // Default ON — strongest FP4 variant per the July 5 bucketed matrix,
    // verified default 2026-07-08.  IDA_NATIVE_FP4_GAUSS=0 for ablation.
    const char* e = std::getenv("IDA_NATIVE_FP4_GAUSS");
    return !(e && e[0] == '0');
}

static bool fp4_centered_enabled() {
    const char* e = std::getenv("IDA_NATIVE_FP4_CENTERED");
    return e && e[0] == '1';
}

// Legacy switch retained for compatibility; private packed-precision code is
// not present in the public target.
static bool fp4_centered_int2_enabled() {
    const char* e = std::getenv("IDA_NATIVE_FP4_CENTERED_INT2");
    return e && e[0] == '1';
}

// Applies the forward/recompute timing fix to every delayed-state packed-FP4
// path (plain amax, gauss, e2m1, centered, centered-int2). Without this,
// prepare_packed_fp4_attention_forward runs twice per micro-step (true
// forward + backward's activation-checkpoint recompute) and each call
// advances the live delayed-scaling state, so the recompute call packs/
// decodes against a DIFFERENT scale/mean than the true forward used.
// Centered mode is the most violent case because a stale mean is an absolute
// error, but the plain amax-based paths still drift multiplicatively unless
// they reuse frozen per-step scale/descale too. The env name stays the same
// to preserve existing bench knobs.
static bool fp4_delayed_state_fix_enabled() {
    // Default ON — this is a correctness fix, verified default 2026-07-08.
    // IDA_NATIVE_FP4_CENTERED_FIX=0 re-exposes the timing bug for ablation.
    const char* e = std::getenv("IDA_NATIVE_FP4_CENTERED_FIX");
    return !(e && e[0] == '0');
}

// Untested lab flag (default OFF) — see fp4_amax_pair's comment in
// kernels.hpp for the mechanism.  Removes the one-call-lag inherent to
// delayed scaling for the gauss Q/K path specifically (an extra amax pass
// per call, ~2x the bandwidth cost of the single-pass encode it replaces —
// cheap relative to attention/FFN, but real).  Candidate fix for the
// L0.attn_norm/k_proj/q_proj gradient explosion that scales with
// microbatch size (2026-07-09); qk_row_clip (a per-row L2-norm clip) was
// tried first and did not resolve it, consistent with the failure being a
// per-element/channel spike a row-norm check can't see.
static bool fp4_exact_calib_enabled() {
    const char* e = std::getenv("IDA_NATIVE_FP4_EXACT_CALIB");
    return e && e[0] == '1';
}

static bool packed_fp4_backward_replay_saved_e4m3_enabled() {
    const char* e = std::getenv("IDA_NATIVE_PACKED_FP4_BWD_REPLAY");
    return e && std::string_view(e) == "saved_e4m3";
}

// Legacy switch retained for compatibility; private packed-precision code is
// not present in the public target.
static bool fp4_centered_smooth_enabled() {
    const char* e = std::getenv("IDA_NATIVE_FP4_CENTERED_SMOOTH");
    return e && e[0] == '1';
}

static void prepare_packed_fp4_attention_forward(
    PackedFp4AttentionCtx& ctx,
    const StepBuffers& sb,
    int layer_idx,
    cudaStream_t s,
    bool is_recompute
) {
    if (!ctx.on) {
        return;
    }

    const std::size_t n = static_cast<std::size_t>(sb.B) * sb.S * sb.H;
    // Per-layer slot: this layer's own delayed-scaling state, independent of
    // every other layer's (see PackedFp4AttentionCtx comment).
    float* q_amax = ctx.q_amax + layer_idx;
    float* q_scale = ctx.q_scale + layer_idx;
    float* q_descale = ctx.q_descale + layer_idx;
    float* k_amax = ctx.k_amax + layer_idx;
    float* k_scale = ctx.k_scale + layer_idx;
    float* k_descale = ctx.k_descale + layer_idx;
    float* q_mean = ctx.q_mean + layer_idx;
    float* k_mean = ctx.k_mean + layer_idx;
    float* q_sum = ctx.q_sum + layer_idx;
    float* k_sum = ctx.k_sum + layer_idx;
    float* q_max_abs_dev = ctx.q_max_abs_dev + layer_idx;
    float* k_max_abs_dev = ctx.k_max_abs_dev + layer_idx;
    float* q_mean_snap = ctx.q_mean_snapshot + layer_idx;
    float* k_mean_snap = ctx.k_mean_snapshot + layer_idx;
    float* q_scale_snap = ctx.q_scale_snapshot + layer_idx;
    float* k_scale_snap = ctx.k_scale_snapshot + layer_idx;
    float* q_descale_snap = ctx.q_descale_snapshot + layer_idx;
    float* k_descale_snap = ctx.k_descale_snapshot + layer_idx;

    const bool centered_int2 = fp4_centered_int2_enabled();
    if (fp4_centered_enabled() || centered_int2) {
        const auto pack_record = centered_int2
            ? fp4_pack_pair_centered_int2_record
            : fp4_pack_pair_centered_record;
        const auto scale_from_stats = centered_int2
            ? fp4_scale_from_stats_centered_int2
            : (fp4_centered_smooth_enabled()
                   ? fp4_scale_from_stats_centered_smooth
                   : fp4_scale_from_stats_centered);

        if (gradnorm_debug_enabled() && layer_idx == 0 && !is_recompute) {
            // q_mean/q_scale (and k_) here are still the CURRENT (pre-advance)
            // values regardless of fix/no-fix — both branches below read them
            // as-is before mutating, so this is exactly the window pack_record
            // is about to encode with.
            debug_print_centered_clip_stats("Q(ctr)", sb.qkv, n, q_mean, q_scale, s);
            debug_print_centered_clip_stats("K(ctr)", sb.qkv + sb.H, n, k_mean, k_scale, s);
        }

        if (!fp4_delayed_state_fix_enabled()) {
            // Original (no-fix) behavior: always the live state, called
            // identically on both the true-forward and recompute passes —
            // whatever forward/recompute mismatch that causes is left in
            // deliberately for the "int2 strict, no fix" comparison bench.
            pack_record(
                sb.qkv, sb.qkv + sb.H, ctx.qk_packed,
                q_mean, k_mean, q_scale, k_scale,
                q_sum, k_sum, q_max_abs_dev, k_max_abs_dev, n, s);
            scale_from_stats(q_sum, q_max_abs_dev, static_cast<float>(n), q_mean, q_scale, q_descale, s);
            scale_from_stats(k_sum, k_max_abs_dev, static_cast<float>(n), k_mean, k_scale, k_descale, s);
            if (gradnorm_debug_enabled() && layer_idx == 0) {
                debug_print_scalar("fp4_L0_q_mean(centered)", q_mean, s);
                debug_print_scalar("fp4_L0_k_mean(centered)", k_mean, s);
            }
            return;
        }

        if (!is_recompute) {
            // True forward: freeze what's about to be used to encode THIS
            // call (mean/scale/descale all consistent from the last call),
            // before record+update advances them to next call's values.
            IDA_CUDA_CHECK(cudaMemcpyAsync(q_mean_snap, q_mean, sizeof(float), cudaMemcpyDeviceToDevice, s));
            IDA_CUDA_CHECK(cudaMemcpyAsync(k_mean_snap, k_mean, sizeof(float), cudaMemcpyDeviceToDevice, s));
            IDA_CUDA_CHECK(cudaMemcpyAsync(q_scale_snap, q_scale, sizeof(float), cudaMemcpyDeviceToDevice, s));
            IDA_CUDA_CHECK(cudaMemcpyAsync(k_scale_snap, k_scale, sizeof(float), cudaMemcpyDeviceToDevice, s));
            IDA_CUDA_CHECK(cudaMemcpyAsync(q_descale_snap, q_descale, sizeof(float), cudaMemcpyDeviceToDevice, s));
            IDA_CUDA_CHECK(cudaMemcpyAsync(k_descale_snap, k_descale, sizeof(float), cudaMemcpyDeviceToDevice, s));

            pack_record(
                sb.qkv, sb.qkv + sb.H, ctx.qk_packed,
                q_mean, k_mean, q_scale, k_scale,
                q_sum, k_sum, q_max_abs_dev, k_max_abs_dev, n, s);
            scale_from_stats(q_sum, q_max_abs_dev, static_cast<float>(n), q_mean, q_scale, q_descale, s);
            scale_from_stats(k_sum, k_max_abs_dev, static_cast<float>(n), k_mean, k_scale, k_descale, s);
        } else {
            // Recompute: re-encode using the FROZEN values the true forward
            // used, so ctx.qk_packed ends up bit-identical both times.
            // sum/max_abs_dev outputs here are throwaway scratch — the live
            // state already advanced once this step and must not move again.
            pack_record(
                sb.qkv, sb.qkv + sb.H, ctx.qk_packed,
                q_mean_snap, k_mean_snap, q_scale_snap, k_scale_snap,
                q_sum, k_sum, q_max_abs_dev, k_max_abs_dev, n, s);
        }
        if (gradnorm_debug_enabled() && layer_idx == 0) {
            debug_print_scalar(is_recompute ? "fp4_L0_q_mean_snap(re)" : "fp4_L0_q_mean(fw)", q_mean_snap, s);
        }
        return;
    }
    if (!is_recompute && fp4_delayed_state_fix_enabled()) {
        IDA_CUDA_CHECK(cudaMemcpyAsync(q_scale_snap, q_scale, sizeof(float), cudaMemcpyDeviceToDevice, s));
        IDA_CUDA_CHECK(cudaMemcpyAsync(k_scale_snap, k_scale, sizeof(float), cudaMemcpyDeviceToDevice, s));
        IDA_CUDA_CHECK(cudaMemcpyAsync(q_descale_snap, q_descale, sizeof(float), cudaMemcpyDeviceToDevice, s));
        IDA_CUDA_CHECK(cudaMemcpyAsync(k_descale_snap, k_descale, sizeof(float), cudaMemcpyDeviceToDevice, s));
    }
    if (fp4_gauss_enabled()) {
        if (fp4_exact_calib_enabled()) {
            // Exact (non-delayed) calibration: this call's own amax is
            // computed FIRST, a scale derived from it immediately, and only
            // THEN is the data encoded — no one-call lag for a larger
            // microbatch's higher max-of-N to outrun.  Run identically on
            // forward and recompute (pure function of current data, no
            // snapshot plumbing needed — see qk_row_clip's comment above
            // for the mechanism this targets: L0.attn_norm/k_proj/q_proj
            // blowup that grows with microbatch size).
            fp4_amax_pair(sb.qkv, sb.qkv + sb.H, q_amax, k_amax, n, s);
            fp4_scale_from_amax_gauss(q_amax, q_scale, q_descale, s);
            fp4_scale_from_amax_gauss(k_amax, k_scale, k_descale, s);
            fp4_pack_pair_gauss_record(
                sb.qkv, sb.qkv + sb.H, ctx.qk_packed,
                q_scale, k_scale, q_amax, k_amax, n, s);
        } else if (is_recompute && fp4_delayed_state_fix_enabled()) {
            fp4_pack_pair_gauss_record(
                sb.qkv, sb.qkv + sb.H, ctx.qk_packed,
                q_scale_snap, k_scale_snap, q_amax, k_amax, n, s);
        } else {
            fp4_pack_pair_gauss_record(
                sb.qkv, sb.qkv + sb.H, ctx.qk_packed,
                q_scale, k_scale, q_amax, k_amax, n, s);
            fp4_scale_from_amax_gauss(q_amax, q_scale, q_descale, s);
            fp4_scale_from_amax_gauss(k_amax, k_scale, k_descale, s);
        }
        if (gradnorm_debug_enabled()) {
            debug_print_scalar(("fp4_L" + std::to_string(layer_idx) + "_q_amax(gs)").c_str(), q_amax, s);
            debug_print_scalar(("fp4_L" + std::to_string(layer_idx) + "_k_amax(gs)").c_str(), k_amax, s);
        }
        return;
    }
    if (fp4_int2_enabled()) {
        if (is_recompute && fp4_delayed_state_fix_enabled()) {
            fp4_pack_pair_int2_record(
                sb.qkv, sb.qkv + sb.H, ctx.qk_packed,
                q_scale_snap, k_scale_snap, q_amax, k_amax, n, s);
        } else {
            fp4_pack_pair_int2_record(
                sb.qkv, sb.qkv + sb.H, ctx.qk_packed,
                q_scale, k_scale, q_amax, k_amax, n, s);
            fp4_scale_from_amax_int2(q_amax, q_scale, q_descale, s);
            fp4_scale_from_amax_int2(k_amax, k_scale, k_descale, s);
        }
        if (gradnorm_debug_enabled() && layer_idx == 0) {
            debug_print_qk_distribution("Q(int2)", sb.qkv, n, q_amax, s);
            debug_print_qk_distribution("K(int2)", sb.qkv + sb.H, n, k_amax, s);
        }
        return;
    }
    if (fp4_e2m1_enabled()) {
        if (is_recompute && fp4_delayed_state_fix_enabled()) {
            fp4_pack_pair_e2m1_true_record(
                sb.qkv, sb.qkv + sb.H, ctx.qk_packed,
                q_scale_snap, k_scale_snap, q_amax, k_amax, n, s);
        } else {
            fp4_pack_pair_e2m1_true_record(
                sb.qkv, sb.qkv + sb.H, ctx.qk_packed,
                q_scale, k_scale, q_amax, k_amax, n, s);
            fp4_scale_from_amax_e2m1(q_amax, q_scale, q_descale, s);
            fp4_scale_from_amax_e2m1(k_amax, k_scale, k_descale, s);
        }
        return;
    }
    if (is_recompute && fp4_delayed_state_fix_enabled()) {
        fp4_pack_pair_e2m1_record(
            sb.qkv,
            sb.qkv + sb.H,
            ctx.qk_packed,
            q_scale_snap,
            k_scale_snap,
            q_amax,
            k_amax,
            n,
            s
        );
    } else {
        fp4_pack_pair_e2m1_record(
            sb.qkv,
            sb.qkv + sb.H,
            ctx.qk_packed,
            q_scale,
            k_scale,
            q_amax,
            k_amax,
            n,
            s
        );
        fp4_scale_from_amax(q_amax, q_scale, q_descale, s);
        fp4_scale_from_amax(k_amax, k_scale, k_descale, s);
    }
    if (gradnorm_debug_enabled() && layer_idx == 0) {
        debug_print_qk_distribution("Q(int4)", sb.qkv, n, q_amax, s);
        debug_print_qk_distribution("K(int4)", sb.qkv + sb.H, n, k_amax, s);
    }
}

// ─── reshape BF16 [B*S,H] ↔ per-head BF16 [B,nH,S,Hd] ────────────────────────
// Pure layout change (interleave/de-interleave heads) — no dtype conversion.
__global__ void k_reshape_to_heads_bf16(
    const __nv_bfloat16* src,  // [B*S, H] (row stride = ld)
    __nv_bfloat16*       dst,  // [B, nH, S, Hd]
    int B, int S, int nH, int Hd, int ld
) {
    const int H  = nH * Hd;
    const int bs = blockIdx.x;
    const int h  = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= H) return;
    const int b   = bs / S;
    const int s   = bs % S;
    const int nh  = h / Hd;
    const int hd  = h % Hd;
    dst[b * (nH * S * Hd) + nh * (S * Hd) + s * Hd + hd] =
        src[static_cast<std::size_t>(bs) * ld + h];
}

__global__ void k_reshape_from_heads_bf16(
    const __nv_bfloat16* src,  // [B, nH, S, Hd]
    __nv_bfloat16*       dst,  // [B*S, H] (row stride = ld)
    int B, int S, int nH, int Hd, int ld
) {
    const int H  = nH * Hd;
    const int bs = blockIdx.x;
    const int h  = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= H) return;
    const int b  = bs / S;
    const int s  = bs % S;
    const int nh = h / Hd;
    const int hd = h % Hd;
    dst[static_cast<std::size_t>(bs) * ld + h] =
        src[b * (nH * S * Hd) + nh * (S * Hd) + s * Hd + hd];
}

// GQA/MQA K/V projection view: raw K/V has [B*S, kv_heads*Hd] columns,
// while the proven scalar attention kernels consume [B, q_heads, S, Hd].
// Materialize the shared K/V head for every Q-head group. This keeps the
// attention kernel's MHA contract untouched; backward reduces the replicated
// dK/dV with the matching k_reduce_heads_to_kv_bf16 kernel below.
__global__ void k_reshape_kv_to_heads_bf16(
    const __nv_bfloat16* src,  // [B*S, kv_heads*Hd] (row stride = ld)
    __nv_bfloat16* dst,        // [B, q_heads, S, Hd]
    int B, int S, int q_heads, int kv_heads, int Hd, int ld
) {
    const int bs = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    const int H = q_heads * Hd;
    if (h >= H) return;
    const int b = bs / S;
    const int s = bs % S;
    const int qh = h / Hd;
    const int hd = h % Hd;
    const int group = q_heads / kv_heads;
    const int kvh = qh / group;
    dst[static_cast<std::size_t>(b) * q_heads * S * Hd +
        static_cast<std::size_t>(qh) * S * Hd + s * Hd + hd] =
        src[static_cast<std::size_t>(bs) * ld + kvh * Hd + hd];
}

// Sum the attention gradients for each replicated Q-head group back into the
// raw K/V projection layout [B*S, kv_heads*Hd]. The fixed group order makes
// this deterministic and avoids atomics before the dW GEMMs consume it.
__global__ void k_reduce_heads_to_kv_bf16(
    const __nv_bfloat16* src,  // [B, q_heads, S, Hd]
    __nv_bfloat16* dst,        // [B*S, kv_heads*Hd] (row stride = ld)
    int B, int S, int q_heads, int kv_heads, int Hd, int ld
) {
    const int bs = blockIdx.x;
    const int kv_col = blockIdx.y * blockDim.x + threadIdx.x;
    const int KV = kv_heads * Hd;
    if (kv_col >= KV) return;
    const int b = bs / S;
    const int s = bs % S;
    const int kvh = kv_col / Hd;
    const int hd = kv_col % Hd;
    const int group = q_heads / kv_heads;
    float total = 0.0f;
    for (int g = 0; g < group; ++g) {
        const int qh = kvh * group + g;
        total += __bfloat162float(
            src[static_cast<std::size_t>(b) * q_heads * S * Hd +
                static_cast<std::size_t>(qh) * S * Hd + s * Hd + hd]);
    }
    dst[static_cast<std::size_t>(bs) * ld + kv_col] = __float2bfloat16(total);
}

// ─── Rotary position embedding (RoPE) ────────────────────────────────────────
// Llama/Qwen/Phi-class "rotate half" convention: split each head's Hd-wide
// vector into two halves [0,Hd/2) and [Hd/2,Hd), rotate each (x1,x2) pair by
// angle = pos * theta_i, theta_i = rope_theta^(-2i/Hd). In-place on the same
// [B, nH, S, Hd] layout k_reshape_to_heads_bf16 already produces -- inserted
// right after that reshape, before attention_forward, so it needs no new
// buffer and touches Q/K only (never V, which carries no positional score
// term). IDA's own Lattice architecture has no positional encoding of any
// kind (see the 2026-08-13 sideload evaluation) and never sets rope_theta,
// so this is dead code for every existing production body -- added only for
// the standard-architecture (Llama/Qwen/Phi-class) portability port.
//
// Backward is the same rotation with the angle negated: a rotation matrix is
// orthogonal, so for y = R(theta) x, dL/dx = R(theta)^T dL/dy = R(-theta)
// dL/dy. angle_sign selects which: +1 for the forward application to Q/K,
// -1 for the backward application to dQ/dK (see the call site after
// attention_backward for why dQ/dK, not dNormed, is the right place -- the
// gradient must be un-rotated before the dW_q/dW_k GEMMs, which expect the
// gradient w.r.t. q_proj/k_proj's raw (pre-rotation) output).
//
// Position is packed-sample-relative, not the flat row index: segs[b*S+s]
// holds the current sample's start offset within the packed window (same
// field attention.cu's block-diagonal culling already reads), so
// pos = s - segs[b*S+s] resets to 0 at each sample boundary instead of
// running across the whole packed sequence. segs may be null (no packing
// in play); pos falls back to the flat row index s.
__global__ void k_rope_apply_bf16(
    __nv_bfloat16* __restrict__ x,          // [B, nH, S, Hd], rotated in place
    const std::uint16_t* __restrict__ segs, // [B, S] sample-start per token, or null
    int B, int nH, int S, int Hd,
    float rope_theta, float angle_sign
) {
    const int half = Hd / 2;
    const long total = static_cast<long>(B) * nH * S * half;
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int i  = static_cast<int>(idx % half); idx /= half;
    const int s  = static_cast<int>(idx % S);    idx /= S;
    const int nh = static_cast<int>(idx % nH);   idx /= nH;
    const int b  = static_cast<int>(idx);

    const int seg_start = segs
        ? static_cast<int>(segs[static_cast<std::size_t>(b) * S + s])
        : 0;
    const int pos = s - seg_start;

    const float theta_i = powf(rope_theta, -2.0f * static_cast<float>(i) / static_cast<float>(Hd));
    const float angle = angle_sign * static_cast<float>(pos) * theta_i;
    const float c = cosf(angle);
    const float sn = sinf(angle);

    __nv_bfloat16* row = x +
        ((static_cast<std::size_t>(b) * nH + nh) * S + s) * Hd;
    const float x1 = __bfloat162float(row[i]);
    const float x2 = __bfloat162float(row[i + half]);
    row[i]        = __float2bfloat16(x1 * c - x2 * sn);
    row[i + half] = __float2bfloat16(x2 * c + x1 * sn);
}

static void rope_apply(
    __nv_bfloat16* x, const std::uint16_t* segs,
    int B, int nH, int S, int Hd, float rope_theta, float angle_sign,
    cudaStream_t stream
) {
    if (rope_theta <= 0.0f) return;  // off for every existing IDA Lattice body
    const long total = static_cast<long>(B) * nH * S * (Hd / 2);
    const int threads = 256;
    const long blocks = (total + threads - 1) / threads;
    k_rope_apply_bf16<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        x, segs, B, nH, S, Hd, rope_theta, angle_sign);
}

// ─── layer forward body (shared by forward pass and backward recompute) ──────
// Reads layer input from `h_in`, leaves:
//   sb.normed      = attention-norm output   (rms in sb.rms_a)
//   sb.q/k/v_f32   = per-head Q, K, V
//   sb.o_f32/lse   = flash-attention context + log-sum-exp (for backward)
//   sb.attn_out    = attention context (input to O projection)
//   sb.o_out       = O projection output
//   sb.hidden      = h_in + o_out  (post-attention residual)
//   sb.normed2     = ffn-norm output          (rms in sb.rms_save)
//   sb.gate/up/swiglu_out = FFN intermediates
// Returns nothing; caller finishes with down-proj + residual when needed.
static bool is_gpt2_contract(const NativeRequest& request) {
    const std::string& contract = request.model.architecture_contract.empty()
        ? request.architecture_contract : request.model.architecture_contract;
    return contract == "hf_gpt2_native_v1";
}

static float model_norm_eps(const NativeRequest& request) {
    return request.model.norm_eps > 0.0f ? request.model.norm_eps : 1e-5f;
}
static void layer_forward_body(
    cublasHandle_t handle,
    AttentionBackendKind attention_backend,
    Fp8Ctx& f8,
    PackedFp4AttentionCtx& fp4_attn,
    const NativeRequest& request,
    int layer_idx,
    const LatticeLayerWeights& lw,
    const __nv_bfloat16* h_in,
    StepBuffers& sb,
    cudaStream_t s,
    bool is_recompute = false
) {
    IDA_GEMM_ROLE(::ida_native::gemm_trace::R_REGION_LAYER_FWD);
    const int BS = sb.B * sb.S;
    const int H  = sb.H;
    const int I  = sb.I;
    const bool gpt2 = is_gpt2_contract(request);
    const int nH = sb.nH;
    const int Hd = sb.Hd;
    const int KV = sb.kvH;
    const bool gqa = KV != H;
    const std::size_t bsh = static_cast<std::size_t>(BS) * H;
    const Fp8Slot* fsl = f8.on ? &f8.slots[layer_idx * 7] : nullptr;
    ActSlot* qkv_act = f8.on ? &layer_act_slot(f8, layer_idx, ACT_QKV_IN) : nullptr;
    ActSlot* o_act = f8.on ? &layer_act_slot(f8, layer_idx, ACT_O_IN) : nullptr;
    ActSlot* ffn_act = f8.on ? &layer_act_slot(f8, layer_idx, ACT_FFN_IN) : nullptr;

    // 1. Attention RMSNorm
    if (gpt2) layernorm_forward(h_in, lw.attn_norm, lw.attn_norm_bias,
                                sb.normed, sb.rms_a, BS, H, model_norm_eps(request), s);
    else rmsnorm_forward(h_in, lw.attn_norm, sb.normed, sb.rms_a, BS, H, 1e-6f, s);
    // Completing the systematic sweep (2026-07-09): sb.normed is the
    // activation operand for gl.g_q/g_k/g_v.  RMSNorm bounds row L2 norm by
    // construction, so this is lower-risk than the others, but the ceiling
    // is cheap insurance given everything else on this list turned out to
    // matter.
    act_row_clip(request, sb.normed, BS, H, sb.rowdot, s);

    // 2. Q/K/V projections → qkv[BS, 3H]
    if (f8.on && !gqa) {
        if (is_recompute && f8.snap_qkv && qkv_act->fp8_max == kE4M3Max &&
                fp8_snap_fuse_enabled()) {
            // Fused: one BF16 read → act8 (non-transposed) + snap_qkv (transposed).
            // Saves the act8 HBM read-back that transpose_u8 requires.
            fp8_quant_and_transpose_e4m3(sb.normed, f8.act8, f8.snap_qkv,
                                         qkv_act->scale_snapshot, BS, H, s);
        } else {
            fp8_quant_act(f8, *qkv_act, sb.normed, bsh, s, is_recompute);
            if (is_recompute && f8.snap_qkv && qkv_act->fp8_max == kE4M3Max)
                transpose_u8(f8.act8, f8.snap_qkv, BS, H, s);  // → [H,BS] for TN dW
        }
        // Always the snapshot: slot.descale itself has already been advanced
        // to the NEXT call's value by the time fp8_quant_act returns (real
        // bug, predates today's recompute work) — the snapshot is the only
        // value that actually matches what was just quantized.
        const float* qkv_descale = qkv_act->descale_snapshot;
        if (gradnorm_debug_enabled() && layer_idx == 0) {
            debug_print_scalar(is_recompute ? "qkv_descale(re)" : "qkv_descale(fw)", qkv_descale, s);
        }
        const cudaDataType_t qkv_t = act_fp8_type(*qkv_act);
        lt_gemm_fp8_nt(f8, BS, H, H, f8.act8, H, qkv_t, qkv_descale,
                       fsl[F8_Q].fwd8, fsl[F8_Q].descale, sb.qkv,         3 * H, s);
        lt_gemm_fp8_nt(f8, BS, H, H, f8.act8, H, qkv_t, qkv_descale,
                       fsl[F8_K].fwd8, fsl[F8_K].descale, sb.qkv + H,     3 * H, s);
        lt_gemm_fp8_nt(f8, BS, H, H, f8.act8, H, qkv_t, qkv_descale,
                       fsl[F8_V].fwd8, fsl[F8_V].descale, sb.qkv + 2 * H, 3 * H, s);
    } else {
        gemm_bf16(handle, BS, H, H, 1.f, sb.normed, H,
                  ampere_compute_weight(lw.q_proj), H, 0.f, sb.qkv, 3 * H);
        gemm_bf16(handle, BS, KV, H, 1.f, sb.normed, H,
                  ampere_compute_weight(lw.k_proj), KV, 0.f,
                  sb.qkv + H, 3 * H);
        gemm_bf16(handle, BS, KV, H, 1.f, sb.normed, H,
                  ampere_compute_weight(lw.v_proj), KV, 0.f,
                  sb.qkv + 2 * H, 3 * H);
    }

    // Qwen2-family QKV bias (2026-08-23): additive per-output-feature bias,
    // applied identically regardless of which GEMM path above wrote sb.qkv
    // (fp8 or bf16) -- both leave the pre-bias projection in the same
    // [BS, 3H]-strided layout (Q at base 0, K at base H, V at base 2H).
    // No o_proj bias (Qwen2Attention doesn't have one either).
    if (lw.q_bias != nullptr) {
        bias_add_strided_bf16(sb.qkv, ampere_compute_weight(lw.q_bias), BS, H, 3 * H, 0, s);
        bias_add_strided_bf16(sb.qkv, ampere_compute_weight(lw.k_bias), BS, KV, 3 * H, H, s);
        bias_add_strided_bf16(sb.qkv, ampere_compute_weight(lw.v_bias), BS, KV, 3 * H, 2 * H, s);
    }

    if (fp4_attn.on && attention_backend == AttentionBackendKind::HopperWgmmaPackedFp4) {
        // Bound Q/K row outliers before the delayed-scale FP4 calibration
        // reads them — see qk_row_clip's comment for the mechanism.  Applied
        // identically on both the true-forward and recompute calls (pure
        // function of current data, no persistent state, so no forward/
        // recompute mismatch risk the way the delayed scale itself has).
        qk_row_clip(sb.qkv, BS, H, sb.rowdot, s);
        // Prepare pre-head row-major Q/K bytes for the future Hopper WGMMA path.
        prepare_packed_fp4_attention_forward(fp4_attn, sb, layer_idx, s, is_recompute);
    }

    // 3. Reshape to heads
    {
        dim3 grid(BS, (H + 127) / 128);
        k_reshape_to_heads_bf16<<<grid, 128, 0, s>>>(sb.qkv, sb.q_f32,
            sb.B, sb.S, nH, Hd, 3 * H);
        if (!gqa) {
            k_reshape_to_heads_bf16<<<grid, 128, 0, s>>>(sb.qkv + H, sb.k_f32,
                sb.B, sb.S, nH, Hd, 3 * H);
            k_reshape_to_heads_bf16<<<grid, 128, 0, s>>>(sb.qkv + 2 * H, sb.v_f32,
                sb.B, sb.S, nH, Hd, 3 * H);
        } else {
            const int kv_heads = KV / Hd;
            k_reshape_kv_to_heads_bf16<<<grid, 128, 0, s>>>(sb.qkv + H, sb.k_f32,
                sb.B, sb.S, nH, kv_heads, Hd, 3 * H);
            k_reshape_kv_to_heads_bf16<<<grid, 128, 0, s>>>(sb.qkv + 2 * H, sb.v_f32,
                sb.B, sb.S, nH, kv_heads, Hd, 3 * H);
        }
    }

    // Rotary position embedding: Q/K only, V untouched. No-op (rope_apply
    // returns immediately) for every existing IDA Lattice body -- see
    // rope_apply's comment.
    rope_apply(sb.q_f32, sb.segs, sb.B, nH, sb.S, Hd, request.model.rope_theta, +1.0f, s);
    rope_apply(sb.k_f32, sb.segs, sb.B, nH, sb.S, Hd, request.model.rope_theta, +1.0f, s);

    PackedFp4AttentionOperands packed_fp4_local{};
    const PackedFp4AttentionOperands* packed_fp4 = nullptr;
    if (fp4_attn.on) {
        packed_fp4_local = packed_fp4_operands(fp4_attn, layer_idx);
        packed_fp4 = &packed_fp4_local;
    }

    // 4-6. Flash-style causal attention: context + saved LSE, no [S,S] buffer.
    // Sample-boundary masking (block-diagonal over packed samples) is live on
    // every backend: the WGMMA kernels cull key tiles below the query tile's
    // sample start and mask boundary crossings per element (phase 2).
    const uint16_t* attn_segs = sb.segs;
    attention_forward(attention_backend,
                      sb.q_f32, sb.k_f32, sb.v_f32, sb.o_f32, sb.lse,
                      packed_fp4,
                      attn_segs, nH,
                      sb.B * nH, sb.S, Hd,
                      1.0f / sqrtf(static_cast<float>(Hd)), s,
                      KV / Hd);

    // 7. Reshape back to [B*S, H]
    {
        dim3 grid(BS, (H + 127) / 128);
        k_reshape_from_heads_bf16<<<grid, 128, 0, s>>>(
            sb.o_f32, sb.attn_out, sb.B, sb.S, nH, Hd, H);
    }

    if (gradnorm_debug_enabled() && layer_idx == 0) {
        debug_print_activation_stats("attn_o_f32", sb.o_f32, static_cast<std::size_t>(sb.B) * sb.S * H, s);
        debug_print_activation_stats("attn_out",   sb.attn_out, bsh, s);
    }

    // 8. O projection
    // Row-clip punctuation/sink outlier tokens BEFORE either path consumes
    // attn_out — was gated inside f8.on only (comment below), so the BF16
    // linear profile (hopper_bf16_packed_fp4, used by all the mb=128 AI
    // ablation runs) got ZERO protection on this activation at all.
    // Found 2026-07-09: this is why clipping the GRADIENT operand
    // (d_hidden, via interlayer_clip at the top of attention backward)
    // never touched L0.o_proj's explosion — attn_out itself, an
    // ACTIVATION not a gradient, was the unbounded operand in
    // g_o = attn_out^T @ d_hidden the whole time, and no downstream
    // gradient-side clip can fix an unbounded activation upstream of it.
    // Deterministic function of attn_out's current content, so forward
    // and recompute stay bit-identical regardless of which branch runs.
    act_row_clip(request, sb.attn_out, BS, H, sb.rowdot, s);
    if (f8.on) {
        if (is_recompute && f8.snap_o && o_act->fp8_max == kE4M3Max &&
                fp8_snap_fuse_enabled()) {
            fp8_quant_and_transpose_e4m3(sb.attn_out, f8.act8, f8.snap_o,
                                         o_act->scale_snapshot, BS, H, s);
        } else {
            fp8_quant_act(f8, *o_act, sb.attn_out, bsh, s, is_recompute);
            if (is_recompute && f8.snap_o && o_act->fp8_max == kE4M3Max)
                transpose_u8(f8.act8, f8.snap_o, BS, H, s);  // → [H,BS] for TN dW
        }
        const float* o_descale = o_act->descale_snapshot;  // see qkv_descale comment above
        if (gradnorm_debug_enabled() && layer_idx == 0) {
            debug_print_scalar(is_recompute ? "o_descale(re)" : "o_descale(fw)", o_descale, s);
        }
        lt_gemm_fp8_nt(f8, BS, H, H, f8.act8, H, act_fp8_type(*o_act), o_descale,
                       fsl[F8_O].fwd8, fsl[F8_O].descale, sb.o_out, H, s);
    } else {
        gemm_bf16(handle, BS, H, H, 1.f, sb.attn_out, H,
                  ampere_compute_weight(lw.o_proj), H, 0.f, sb.o_out, H);
    }

    if (lw.o_bias) {
        bias_add_strided_bf16(sb.o_out, ampere_compute_weight(lw.o_bias), BS, H, H, 0, s);
    }
    IDA_LAUNCH(k_add_bf16, "attention.residual_add", ceildiv(bsh, 256), 256, 0, s,
               h_in, sb.o_out, sb.hidden, bsh);
    // Same treatment as attn_out above: sb.hidden feeds ffn_norm's gradient
    // (rmsnorm_backward) exactly the way attn_out fed attn_norm's — an
    // unclipped residual-stream row was the next dominant slot found
    // 2026-07-09 (L1.ffn_norm, 7.6e11) once attn_out's clip closed L0.
    // Deterministic function of current content, forward/recompute stay
    // identical.
    if (gradnorm_debug_enabled() && layer_idx == 0)
        debug_print_activation_stats("hidden_preclip", sb.hidden, bsh, s);
    act_row_clip(request, sb.hidden, BS, H, sb.rowdot, s);

    if (gradnorm_debug_enabled() && layer_idx == 0) {
        debug_print_activation_stats("o_out",  sb.o_out, bsh, s);
        debug_print_activation_stats("hidden", sb.hidden, bsh, s);
    }

    // 10. FFN RMSNorm
    if (gpt2) layernorm_forward(sb.hidden, lw.ffn_norm, lw.ffn_norm_bias,
                                sb.normed2, sb.rms_save, BS, H, model_norm_eps(request), s);
    else rmsnorm_forward(sb.hidden, lw.ffn_norm, sb.normed2, sb.rms_save, BS, H, 1e-6f, s);
    // Same as sb.normed above: gl.g_gate/g_up's activation operand.
    act_row_clip(request, sb.normed2, BS, H, sb.rowdot, s);

    // Routed/expert layers replace the dense FFN with the MoE bank below.
    // The router consumes normed2; gate/up/SwiGLU would be discarded in
    // forward and repeated during checkpoint recompute. Skip it entirely.
    if (lw.num_experts > 0) return;
    if (gpt2) {
        gemm_bf16(handle, BS, I, H, 1.f, sb.normed2, H,
                  ampere_compute_weight(lw.gate_proj), I, 0.f, sb.gate_out, I);
        bias_add_strided_bf16(sb.gate_out, ampere_compute_weight(lw.ffn_in_bias), BS, I, I, 0, s);
        gelu_new_forward(sb.gate_out, sb.swiglu_out, static_cast<std::size_t>(BS) * I, s);
        return;
    }

    // 11. Gate / Up
    if (f8.on) {
        if (is_recompute && f8.snap_ffn && ffn_act->fp8_max == kE4M3Max &&
                fp8_snap_fuse_enabled()) {
            fp8_quant_and_transpose_e4m3(sb.normed2, f8.act8, f8.snap_ffn,
                                         ffn_act->scale_snapshot, BS, H, s);
        } else {
            fp8_quant_act(f8, *ffn_act, sb.normed2, bsh, s, is_recompute);
            if (is_recompute && f8.snap_ffn && ffn_act->fp8_max == kE4M3Max)
                transpose_u8(f8.act8, f8.snap_ffn, BS, H, s);  // → [H,BS] for TN dW
        }
        const float* ffn_descale = ffn_act->descale_snapshot;  // see qkv_descale comment above
        const cudaDataType_t ffn_t = act_fp8_type(*ffn_act);
        lt_gemm_fp8_nt(f8, BS, I, H, f8.act8, H, ffn_t, ffn_descale,
                       fsl[F8_GATE].fwd8, fsl[F8_GATE].descale, sb.gate_out, I, s);
        lt_gemm_fp8_nt(f8, BS, I, H, f8.act8, H, ffn_t, ffn_descale,
                       fsl[F8_UP].fwd8, fsl[F8_UP].descale, sb.up_out, I, s);
    } else {
        gemm_bf16(handle, BS, I, H, 1.f, sb.normed2, H,
                  ampere_compute_weight(lw.gate_proj), I, 0.f, sb.gate_out, I);
        gemm_bf16(handle, BS, I, H, 1.f, sb.normed2, H,
                  ampere_compute_weight(lw.up_proj), I, 0.f, sb.up_out, I);
    }

    // 12. SwiGLU
    swiglu_forward(sb.gate_out, sb.up_out, sb.swiglu_out, BS * I, s);
}

// ─── cognitive-architecture sparse MoE router (Step 3, forward-only) ────────
// PressureField -> ConstitutionalRouter -> LateralInhibition. Operates on
// sb.normed2 (the ffn-norm'd hidden, native's existing FFN input point) --
// same input the dense gate/up/down path would use, so only the FFN's OWN
// input->output mapping changes from dense to sparse-routed (Step 4), not
// what feeds it. Leaves sb.moe_logits holding the final per-position route
// weights; does not yet touch sb.ffn_out (that's Step 4 -- this step is
// forward-only router-pipeline verification, a no-op on the dense path).
// Per-expert token mass from the FINAL post-inhibition route weights
// [BS, E]. counts[e] accumulates each expert's share of routed weight, so a
// collapsed router shows near-all mass on a couple of entries.
// ONE BLOCK PER EXPERT, parallel reduction over BS.
//
// The first version of this used E threads TOTAL (11), each walking BS
// serially. Cost therefore scaled linearly with microbatch (BS = mb*seq), so
// the probe got more expensive exactly as mb grew -- which manufactured a
// fake throughput-saturation curve in the mb ladder and made every tps number
// it touched microbatch-dependently wrong. A diagnostic must not perturb the
// thing it measures. Same shuffle+shared pattern as k_act_row_normsq.
__global__ void k_moe_expert_util_par(
    const float* __restrict__ rw, int BS, int E, float* __restrict__ counts
) {
    __shared__ float sm[256 / 32];
    const int e = blockIdx.x;
    if (e >= E) return;
    float acc = 0.0f;
    for (int t = threadIdx.x; t < BS; t += blockDim.x)
        acc += rw[static_cast<std::size_t>(t) * E + e];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, off);
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int w = 0; w < blockDim.x / 32; ++w) t += sm[w];
        counts[e] = t;
    }
}

static bool expert_util_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("IDA_NATIVE_EXPERT_UTIL");
        return e && e[0] == '1';
    }();
    return v;
}

// Host-side report: share per expert + normalized entropy. Entropy is the
// headline number -- 1.0 means balanced across all E experts, 0.0 means all
// mass on one. A high-loss-quality run with entropy ~0.2 is a dense-2 model
// wearing an 11-expert costume.
static void moe_expert_util_report(
    const float* d_rw, int BS, int E, int layer_idx, int micro_step, cudaStream_t s
) {
    if (!expert_util_enabled() || E <= 0) return;
    float* d_counts = nullptr;
    if (cudaMallocAsync(&d_counts, static_cast<std::size_t>(E) * sizeof(float), s) != cudaSuccess) return;
    IDA_LAUNCH(k_moe_expert_util_par, "moe.expert_utilization", E, 256, 0, s,
               d_rw, BS, E, d_counts);
    std::vector<float> h(static_cast<std::size_t>(E), 0.0f);
    cudaMemcpyAsync(h.data(), d_counts, static_cast<std::size_t>(E) * sizeof(float),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    cudaFreeAsync(d_counts, s);
    double tot = 0.0;
    for (int e = 0; e < E; ++e) tot += h[e];
    if (tot <= 0.0) return;
    double ent = 0.0, mx = 0.0;
    int active = 0;
    for (int e = 0; e < E; ++e) {
        const double p = h[e] / tot;
        if (p > mx) mx = p;
        if (p > 0.01) ++active;          // experts carrying >1% of mass
        if (p > 0.0) ent -= p * std::log(p);
    }
    const double norm_ent = (E > 1) ? (ent / std::log(static_cast<double>(E))) : 0.0;
    std::string shares;
    char buf[32];
    for (int e = 0; e < E; ++e) {
        std::snprintf(buf, sizeof(buf), "%.4f", h[e] / tot);
        shares += buf;
        if (e + 1 < E) shares += ",";
    }
    std::fprintf(stderr,
        "[expert-util] micro_step=%d layer=%d E=%d entropy=%.4f max_share=%.4f "
        "active_gt1pct=%d shares=%s\n",
        micro_step, layer_idx, E, norm_ent, mx, active, shares.c_str());
}



// ── Expert load-balancing auxiliary loss (Switch-Transformer form) ──────────
// `expert_balancing_loss_coef` (0.01) has been present in every MoE config and
// is plumbed all the way into native_request.json, but was NEVER implemented
// -- not in this engine and not in the torch/legacy path either. Nothing has
// ever pushed back on routing concentration. Measured consequence (2026-07-30,
// SENTINEL 0.0.0.1, 300 steps): normalized routing entropy falls 0.44 -> 0.14,
// max_share pins at the 0.9000 lateral-inhibition clamp bound, and only 2 of
// 11 personality experts retain >1% of routed mass. Loss cannot see this -- a
// collapsed router still predicts tokens well -- so it never surfaced.
//
//   L_aux = coef * E * sum_e ( f_e * P_e )
//     f_e = fraction of routed mass dispatched to expert e (post-top-k)
//     P_e = mean pre-top-k softmax probability for expert e
//
// f_e is treated as constant (standard: it comes from a discrete top-k), so
// dL/dP_e = coef * E * f_e, and since P_e = mean_t scores[t][e], the per-token
// gradient is (coef * E * f_e) / BS added into d(scores) BEFORE the softmax
// backward. That is the only injection point where this is correct.
__global__ void k_moe_balance_stats(
    const float* __restrict__ scores,   // [BS,E] pre-top-k softmax
    const float* __restrict__ routed,   // [BS,E] final post-inhibition weights
    int BS, int E,
    float* __restrict__ sumP,           // [E]
    float* __restrict__ sumF            // [E]
) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    float p = 0.0f, f = 0.0f;
    for (int t = 0; t < BS; ++t) {
        const std::size_t i = static_cast<std::size_t>(t) * E + e;
        p += scores[i];
        f += routed[i];
    }
    sumP[e] = p;
    sumF[e] = f;
}

// d_scores[t][e] += coef * E * f_e / BS   (f_e normalized to sum 1 over e)
__global__ void k_moe_balance_grad(
    float* __restrict__ d_scores, const float* __restrict__ sumF,
    int BS, int E, float coef
) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t n = static_cast<std::size_t>(BS) * E;
    if (idx >= n) return;
    const int e = static_cast<int>(idx % E);
    float tot = 0.0f;
    for (int k = 0; k < E; ++k) tot += sumF[k];
    if (tot <= 0.0f) return;
    const float f_e = sumF[e] / tot;
    d_scores[idx] += coef * static_cast<float>(E) * f_e / static_cast<float>(BS);
}

static float expert_balance_coef(float cfg) {
    static const float ov = [] {
        const char* e = std::getenv("IDA_NATIVE_EXPERT_BALANCE_COEF");
        if (!e || !e[0]) return -1.0f;
        char* end = nullptr; const float p = std::strtof(e, &end);
        return (end == e || p < 0.0f) ? -1.0f : p;
    }();
    return (ov >= 0.0f) ? ov : cfg;
}

// Generic per-token top-k SwiGLU MoE forward (2026-08-23; external-model
// portability -- see docs/ "breadth beyond Llama-family" plan). Standard
// Mixtral/Qwen-MoE-shaped routing and experts: softmax(router_logits) ->
// top-k -> renormalize (moe_topk_route_f32 -- IDENTICAL math to the legacy
// mode's own routing formula, just fed straight from sb.normed2 with no
// pooling/pressure/modulation/lateral-inhibition stages), then per expert:
// silu(gate_proj(x)) * up_proj(x) -> down_proj -> scaled by that expert's
// route weight (0 for un-selected rows via moe_topk_route_f32's own
// scatter-zero) and accumulated directly into sb.ffn_out, mirroring
// moe_expert_bank_forward's own accumulation target and
// dense-compute-all-experts strategy (this is a correctness path, not a
// throughput one -- see that function's doc comment for why that's an
// accepted, explicit scope decision in this engine already).
static void moe_generic_forward(const LatticeLayerWeights& lw, StepBuffers& sb, cudaStream_t s) {
    if (lw.num_experts <= 0) return;
    const int BS = sb.B * sb.S;
    const int H = sb.H;
    const int Ie = lw.expert_intermediate_size;
    const int N = lw.num_experts;
    const std::size_t bsh = static_cast<std::size_t>(BS) * H;
    const std::size_t bsi = static_cast<std::size_t>(BS) * Ie;

    cast_bf16_to_f32(sb.normed2, sb.generic_moe_hidden_f32, bsh, s);

    moe_small_proj_f32(sb.generic_moe_hidden_f32, ampere_compute_weight(lw.router_score_w),
                        sb.moe_logits, BS, H, N, s);
    moe_topk_route_f32(sb.moe_logits, BS, N, lw.top_k, s,
                        sb.moe_topk_row_sum, sb.moe_scores_save,
                        lw.generic_moe_normalize_topk);
    // sb.moe_logits now holds final per-position route weights [BS, N].

    bf16_zero(sb.ffn_out, bsh, s);
    for (int e = 0; e < N; ++e) {
        const __nv_bfloat16* gate_w = ampere_compute_weight(
            lw.generic_gate_proj_w + static_cast<std::size_t>(e) * Ie * H);
        const __nv_bfloat16* up_w = ampere_compute_weight(
            lw.generic_up_proj_w + static_cast<std::size_t>(e) * Ie * H);
        const __nv_bfloat16* down_w = ampere_compute_weight(
            lw.generic_down_proj_w + static_cast<std::size_t>(e) * H * Ie);

        moe_small_proj_f32(sb.generic_moe_hidden_f32, gate_w, sb.generic_moe_gate_f32, BS, H, Ie, s);
        moe_small_proj_f32(sb.generic_moe_hidden_f32, up_w, sb.generic_moe_up_f32, BS, H, Ie, s);
        cast_f32_to_bf16(sb.generic_moe_gate_f32, sb.generic_moe_gate_bf16, bsi, s);
        cast_f32_to_bf16(sb.generic_moe_up_f32, sb.generic_moe_up_bf16, bsi, s);
        swiglu_forward(sb.generic_moe_gate_bf16, sb.generic_moe_up_bf16, sb.generic_moe_h_bf16,
                        static_cast<int>(bsi), s);
        cast_bf16_to_f32(sb.generic_moe_h_bf16, sb.generic_moe_h_f32, bsi, s);
        moe_small_proj_f32(sb.generic_moe_h_f32, down_w, sb.generic_moe_down_f32, BS, Ie, H, s);
        cast_f32_to_bf16(sb.generic_moe_down_f32, sb.generic_moe_down_bf16, bsh, s);

        moe_scale_accumulate_bf16(sb.ffn_out, sb.generic_moe_down_bf16, sb.moe_logits, BS, e, N, H, s);
    }

    // Shared expert (Qwen2-MoE-family): an always-on expert applied to every
    // token, additive to sb.ffn_out, scaled by its own per-token sigmoid gate
    // -- NOT part of the N-way softmax route-weight distribution above.
    // Reuses the SAME gate/up/h/down scratch as the routed-expert loop
    // (allocated to max(Ie, shared_expert_width) precisely for this reuse).
    if (lw.shared_expert_width > 0) {
        const int Ws = lw.shared_expert_width;
        const std::size_t bsw = static_cast<std::size_t>(BS) * Ws;

        moe_small_proj_f32(sb.generic_moe_hidden_f32, ampere_compute_weight(lw.generic_shared_gate_score_w),
                            sb.generic_moe_shared_gate_scalar_f32, BS, H, 1, s);
        sigmoid_inplace_f32(sb.generic_moe_shared_gate_scalar_f32, static_cast<std::size_t>(BS), s);

        moe_small_proj_f32(sb.generic_moe_hidden_f32, ampere_compute_weight(lw.generic_shared_gate_proj_w),
                            sb.generic_moe_gate_f32, BS, H, Ws, s);
        moe_small_proj_f32(sb.generic_moe_hidden_f32, ampere_compute_weight(lw.generic_shared_up_proj_w),
                            sb.generic_moe_up_f32, BS, H, Ws, s);
        cast_f32_to_bf16(sb.generic_moe_gate_f32, sb.generic_moe_gate_bf16, bsw, s);
        cast_f32_to_bf16(sb.generic_moe_up_f32, sb.generic_moe_up_bf16, bsw, s);
        swiglu_forward(sb.generic_moe_gate_bf16, sb.generic_moe_up_bf16, sb.generic_moe_h_bf16,
                        static_cast<int>(bsw), s);
        cast_bf16_to_f32(sb.generic_moe_h_bf16, sb.generic_moe_h_f32, bsw, s);
        moe_small_proj_f32(sb.generic_moe_h_f32, ampere_compute_weight(lw.generic_shared_down_proj_w),
                            sb.generic_moe_down_f32, BS, Ws, H, s);
        cast_f32_to_bf16(sb.generic_moe_down_f32, sb.generic_moe_down_bf16, bsh, s);

        // ffn_out[row,:] += gate_scalar[row] * down_bf16[row,:] -- reuses
        // moe_scale_accumulate_bf16 with a [BS,1] "route_weights" tensor
        // (num_experts=1, expert_idx=0), same trick as the router-weighted
        // accumulate above, just with a single always-selected column.
        moe_scale_accumulate_bf16(sb.ffn_out, sb.generic_moe_down_bf16,
                                   sb.generic_moe_shared_gate_scalar_f32, BS, 0, 1, H, s);
    }
}

static void moe_router_forward(const LatticeLayerWeights& lw, StepBuffers& sb, cudaStream_t s) {
    if (lw.num_experts <= 0) return;
    const std::size_t BS = static_cast<std::size_t>(sb.B) * sb.S;
    const int H = sb.H;
    const int num_routes = lw.num_routes;
    const int num_experts = lw.num_experts;

    const bool dbg = gradnorm_debug_enabled();

    // 1. pooled = pool(normed2)  [per packed sample, broadcast per-position]
    moe_pool_by_sample(sb.normed2, sb.segs, sb.moe_pool_scratch, sb.moe_count_scratch, sb.B, sb.S, H, s);
    moe_gather_by_sample(sb.moe_pool_scratch, sb.segs, sb.moe_pooled, sb.B, sb.S, H, s);
    if (dbg) debug_print_f32_activation_stats("moe_pooled", sb.moe_pooled, BS * H, s);
    // Cross-backend parity dump (Phase 4): names/shapes match scripts/
    // native_parity_check.py's fixture keys exactly, so a real trainer.cu
    // burn's dump can be diffed against the SAME PyTorch fixture the
    // standalone moe_router_parity_check binary already verifies -- this
    // is the REAL code path, not the duplicated kernel sequence.
    dump_intermediate_f32("pooled", sb.moe_pooled, BS * H,
                          {static_cast<std::int64_t>(BS), H}, g_debug_micro_step, s);

    // 2. pressure = tanh(pressure_proj(pooled))
    moe_small_proj_f32(sb.moe_pooled, ampere_compute_weight(lw.pressure_proj_w), sb.moe_pressure,
                        static_cast<int>(BS), H, num_routes, s);
    tanh_inplace_f32(sb.moe_pressure, BS * static_cast<std::size_t>(num_routes), s);
    if (dbg) debug_print_f32_activation_stats("moe_pressure", sb.moe_pressure, BS * num_routes, s);
    dump_intermediate_f32("pressure", sb.moe_pressure, BS * num_routes,
                          {static_cast<std::int64_t>(BS), num_routes}, g_debug_micro_step, s);

    // 3. modulation = sigmoid(pressure_mod(pressure))
    moe_small_proj_f32(sb.moe_pressure, ampere_compute_weight(lw.pressure_mod_w), sb.moe_modulation,
                        static_cast<int>(BS), num_routes, H, s);
    sigmoid_inplace_f32(sb.moe_modulation, BS * static_cast<std::size_t>(H), s);
    if (dbg) debug_print_f32_activation_stats("moe_modulation", sb.moe_modulation, BS * H, s);

    // 4. hidden_gated = normed2 * modulation
    moe_modulate_hidden_f32(sb.normed2, sb.moe_modulation, sb.moe_hidden_gated,
                             BS * static_cast<std::size_t>(H), s);
    if (dbg) debug_print_activation_stats("moe_hidden_gated", sb.moe_hidden_gated, BS * H, s);
    dump_intermediate_bf16("hidden_gated", sb.moe_hidden_gated, BS * H,
                           {static_cast<std::int64_t>(BS), H}, g_debug_micro_step, s);

    // 5. pooled2 = pool(hidden_gated)  [second pooling pass, on the gated hidden]
    moe_pool_by_sample(sb.moe_hidden_gated, sb.segs, sb.moe_pool_scratch, sb.moe_count_scratch, sb.B, sb.S, H, s);
    moe_gather_by_sample(sb.moe_pool_scratch, sb.segs, sb.moe_pooled2, sb.B, sb.S, H, s);
    if (dbg) debug_print_f32_activation_stats("moe_pooled2", sb.moe_pooled2, BS * H, s);

    // 6. logits = router_score(pooled2) + routed_pressure
    moe_small_proj_f32(sb.moe_pooled2, ampere_compute_weight(lw.router_score_w), sb.moe_logits,
                        static_cast<int>(BS), H, num_experts, s);
    if (lw.pressure_to_routes_w != nullptr) {
        moe_small_proj_f32(sb.moe_pressure, ampere_compute_weight(lw.pressure_to_routes_w), sb.moe_routed_pressure,
                            static_cast<int>(BS), num_routes, num_experts, s);
        add_inplace_f32(sb.moe_logits, sb.moe_routed_pressure, BS * static_cast<std::size_t>(num_experts), s);
    } else {
        // Identity case (num_routes == num_experts): routed_pressure == pressure itself.
        add_inplace_f32(sb.moe_logits, sb.moe_pressure, BS * static_cast<std::size_t>(num_experts), s);
    }
    if (dbg) debug_print_f32_activation_stats("moe_logits_raw", sb.moe_logits, BS * num_experts, s);

    // 7. softmax -> hard top-k -> scatter -> renormalize (constitutional_router.py)
    // Saves T (row sum pre-renormalize) for backward -- see moe_topk_route_f32's
    // doc comment. Populated on every call, including backward's recompute,
    // since it's cheap (one float per row).
    moe_topk_route_f32(sb.moe_logits, static_cast<int>(BS), num_experts, lw.top_k, s,
                        sb.moe_topk_row_sum, sb.moe_scores_save);
    if (dbg) debug_print_f32_activation_stats("moe_topk_out", sb.moe_logits, BS * num_experts, s);
    // Snapshot p (post top-k renormalize, pre-inhibition) before inhibition
    // overwrites sb.moe_logits further -- needed as backward's "y" for the
    // top-k renormalize's own norm-by-sum backward.
    IDA_CUDA_CHECK(cudaMemcpyAsync(sb.moe_topk_p_save, sb.moe_logits,
        BS * static_cast<std::size_t>(num_experts) * sizeof(float), cudaMemcpyDeviceToDevice, s));

    // 8. lateral inhibition: clamp [minority_floor, dominance_cap] + renormalize
    //    (lateral_inhibition.py's hardcoded reference defaults, no learnable params)
    // Saves c1 (pre-floor clamp) and S (row sum pre-renormalize) for backward.
    moe_lateral_inhibition_f32(sb.moe_logits, static_cast<int>(BS), num_experts, 0.72f, 0.08f, s,
                                sb.moe_inhib_c1, sb.moe_inhib_row_sum);
    if (dbg) debug_print_f32_activation_stats("moe_route_weights", sb.moe_logits, BS * num_experts, s);
    dump_intermediate_f32("route_weights", sb.moe_logits, BS * num_experts,
                          {static_cast<std::int64_t>(BS), num_experts}, g_debug_micro_step, s);
    moe_expert_util_report(sb.moe_logits, static_cast<int>(BS), num_experts,
                           -1, g_debug_micro_step, s);

    // sb.moe_logits now holds the final per-position route weights, ready
    // for Step 4's expert dispatch to consume.
}

// ─── cognitive-architecture sparse MoE expert bank (Step 4, forward-only) ───
// CognitiveCircuitBank: fc_in -> exact-erf GELU -> fc_out per expert (and,
// Mode B, an unconditionally-run shared trunk of the same shape), mixed by
// sb.moe_logits' per-position route weights (moe_router_forward's output)
// into sb.ffn_out -- REPLACES the dense down_proj output for this layer
// when lw.num_experts>0 (caller chooses which path runs, see forward()).
//
// Dense-compute-all-selected-experts, not row-gathered dispatch (explicit
// plan scope decision: per-sequence routing + small expert counts make
// gather/scatter infrastructure not worth it yet). No per-expert zero-
// column skip either -- mathematically identical to skipping (scaling by
// a zero route weight after full computation), just not the perf
// optimization; deferred, tracked as a known follow-up.
// Compatibility hook for a private activation-quality probe. The public
// binary keeps it disabled and does not expose the quantization grid.
static int moe_act_precision_mode() {
    // The public binary never admits private activation quantization through
    // an environment variable. The private package owns that experiment.
    return 0;
}

// Native packed-FP4 activation path: real storage/bandwidth reduction
// layered ON TOP of the existing FP8 GEMM path (Step 4b) -- requires
// f8.on already true, since lt_gemm_fp8_nt needs the cuBLASLt handle/
// workspace that build_fp8_ctx only creates when a real FP8 profile is
// active. Not a standalone precision choice like the bf16-based quality
// probes above; a refinement of the FP8 path, checked only when fp8==true.
//
// thread_local request-set value is the primary mechanism, per-burn safe
// under the persistent worker's multi-tenant dispatch (each concurrently-
// dispatched body gets its own thread; mirrors g_attn_window_request_
// override in attention.cu). The env var fallback below is an ad-hoc
// testing convenience for standalone single-burn invocations (this
// session's manual `train_student_native.py` CLI runs, never routed
// through the persistent multi-tenant worker) -- it carries the SAME
// residual risk the window override's own env fallback already accepted:
// if the persistent worker process's own environment happens to have this
// var set, every thread in that process observes the same value, since
// getenv is process-wide, not per-thread. Not a new hazard introduced
// here -- an existing, accepted tradeoff in this codebase, not silently
// closed. Real per-body control should always go through the request
// field; treat the env var as a manual-testing-only escape hatch.
static bool moe_native_fp4_enabled() {
    return false;
}

static bool moe_fused_expert_fcin_enabled(const LatticeLayerWeights& lw, int trunk_width) {
    const char* e = std::getenv("IDA_NATIVE_MOE_FUSE_EXPERT_FCIN");
    if (!e || e[0] != '1') return false;
    if (lw.num_experts <= 1 || lw.expert_intermediate_size <= 0) return false;
    // v1 is an explicit ablation only. It enables when existing trunk-sized
    // scratch already fits the
    // flattened [BS, E*Ie] output. That gives AI's 4 narrow experts fewer GEMM
    // launches without expanding MoE scratch by multiple GB.
    return lw.num_experts * lw.expert_intermediate_size <= trunk_width;
}

static bool moe_selected_expert_forward_enabled(const LatticeLayerWeights& lw) {
    const char* e = std::getenv("IDA_NATIVE_MOE_SELECTED_EXPERT_FWD");
    if (!e || e[0] != '1') return false;
    return lw.num_experts > 1 && lw.top_k > 0 && lw.top_k < lw.num_experts;
}

static bool moe_selected_expert_backward_enabled(const LatticeLayerWeights& lw) {
    const char* e = std::getenv("IDA_NATIVE_MOE_SELECTED_EXPERT_BWD");
    if (!e || e[0] != '1') return false;
    return lw.num_experts > 1 && lw.top_k > 0 && lw.top_k < lw.num_experts;
}

static bool moe_grouped_device_schedule_enabled(const LatticeLayerWeights& lw) {
    const char* e = std::getenv("IDA_NATIVE_MOE_GROUPED_DEVICE_SCHEDULE");
    if (!e || e[0] != '1') return false;
    return lw.num_experts > 1 && lw.top_k > 0 && lw.top_k < lw.num_experts;
}

static void moe_expert_bank_forward(
    cublasHandle_t handle, Fp8Ctx& f8, int layer_idx,
    const LatticeLayerWeights& lw, StepBuffers& sb, cudaStream_t s
) {
    IDA_GEMM_ROLE(::ida_native::gemm_trace::R_REGION_MOE_FWD);
    if (lw.num_experts <= 0) return;
    const int BS = sb.B * sb.S;
    const int H = sb.H;
    const int Ie = lw.expert_intermediate_size;
    const int probe = moe_act_precision_mode();
    const bool fp8 = f8.on && probe == 0;
    const bool fp4n = fp8 && moe_native_fp4_enabled();
    const float probe_level = probe == 2 ? kMoeInt2Max : kMoeFp4Max;
    const bool dbg = gradnorm_debug_enabled() && layer_idx <= 1;

    if (probe != 0) {
        // Shared fc_in-input: round-trip hidden_gated in place -- safe,
        // nothing downstream in this forward pass reads the pristine value
        // again (moe_router_forward recomputes it fresh next call).
        moe_fake_quant_roundtrip(sb.moe_hidden_gated, sb.moe_hidden_gated,
                                  sb.moe_probe_amax, probe_level, static_cast<std::size_t>(BS) * H, s);
    }

    // Shared fc_in-input quantization: hidden_gated is the SAME tensor fed
    // to the trunk's and every expert's fc_in GEMM, so it's quantized once
    // here and reused directly below -- exactly like ACT_FFN_IN is shared
    // by gate_proj/up_proj today (both read sb.normed2). fp4n packs to FP4
    // then decodes to FP8 into sb.moe_fp4_decode_scratch (own buffer, NOT
    // f8.act8 -- see StepBuffers comment); plain fp8 quantizes straight into
    // f8.act8 as before. act_buf is whichever of the two feeds the GEMMs below.
    ActSlot* fcin_act = nullptr;
    cudaDataType_t fcin_type = CUDA_R_8F_E4M3;
    const float* fcin_descale = nullptr;
    const void* fcin_act_buf = f8.act8;
    if (fp8) {
        fcin_act = &moe_fcin_act_slot(f8, layer_idx);
        if (fp4n) {
            moe_pack_fp4_act(fcin_act->amax, fcin_act->scale, fcin_act->descale,
                              fcin_act->scale_snapshot, fcin_act->descale_snapshot,
                              sb.moe_hidden_gated, sb.moe_fp4_packed_fcin,
                              static_cast<std::size_t>(BS) * H, /*is_recompute=*/false, s);
            moe_decode_packed_fp4_to_e4m3(sb.moe_fp4_packed_fcin, sb.moe_fp4_decode_scratch,
                                          fcin_act->descale_snapshot, static_cast<std::size_t>(BS) * H, s);
            fcin_act_buf = sb.moe_fp4_decode_scratch;
        } else {
            fp8_quant_act(f8, *fcin_act, sb.moe_hidden_gated, static_cast<std::size_t>(BS) * H, s);
            fcin_act_buf = f8.act8;
        }
        // fp4n is always E4M3 (no format calibration, fixed +/-7 grid);
        // plain fp8 keeps its existing E4M3/E5M2 calibration choice.
        fcin_type = fp4n ? CUDA_R_8F_E4M3 : act_fp8_type(*fcin_act);
        fcin_descale = fcin_act->descale_snapshot;
    }

    if (lw.moe_shared_trunk) {
        // trunk_out = fc_out(gelu(fc_in(hidden_gated)))  -- width I (native's
        // own intermediate_size), unconditional, becomes the initial mixed
        // value (weight 1, not gated by any route weight).
        if (fp8) {
            Fp8Slot& w_in = moe_trunk_in_slot(f8, layer_idx);
            lt_gemm_fp8_nt(f8, BS, sb.I, H, fcin_act_buf, H, fcin_type, fcin_descale,
                           w_in.fwd8, w_in.descale, sb.moe_expert_scratch, sb.I, s);
        } else {
            gemm_bf16_nt(handle, BS, sb.I, H, 1.f,
                         sb.moe_hidden_gated, H, ampere_compute_weight(lw.trunk_fc_in_w), H, 0.f,
                         sb.moe_expert_scratch, sb.I);
        }
        gelu_forward(sb.moe_expert_scratch, sb.moe_expert_scratch,
                      static_cast<std::size_t>(BS) * sb.I, s);
        if (dbg) debug_print_activation_stats("moe_trunk_gelu", sb.moe_expert_scratch, static_cast<std::size_t>(BS) * sb.I, s);
        if (probe != 0) {
            moe_fake_quant_roundtrip(sb.moe_expert_scratch, sb.moe_expert_scratch,
                                      sb.moe_probe_amax, probe_level,
                                      static_cast<std::size_t>(BS) * sb.I, s);
        }
        if (fp8) {
            ActSlot& fcout_act = moe_fcout_act_slot(f8, layer_idx, 0);
            const void* fcout_buf;
            if (fp4n) {
                moe_pack_fp4_act(fcout_act.amax, fcout_act.scale, fcout_act.descale,
                                  fcout_act.scale_snapshot, fcout_act.descale_snapshot,
                                  sb.moe_expert_scratch, sb.moe_fp4_packed_fcout,
                                  static_cast<std::size_t>(BS) * sb.I, /*is_recompute=*/false, s);
                moe_decode_packed_fp4_to_e4m3(sb.moe_fp4_packed_fcout, sb.moe_fp4_decode_scratch,
                                              fcout_act.descale_snapshot, static_cast<std::size_t>(BS) * sb.I, s);
                fcout_buf = sb.moe_fp4_decode_scratch;
            } else {
                fp8_quant_act(f8, fcout_act, sb.moe_expert_scratch, static_cast<std::size_t>(BS) * sb.I, s);
                fcout_buf = f8.act8;
            }
            Fp8Slot& w_out = moe_trunk_out_slot(f8, layer_idx);
            lt_gemm_fp8_nt(f8, BS, H, sb.I, fcout_buf, sb.I,
                           fp4n ? CUDA_R_8F_E4M3 : act_fp8_type(fcout_act),
                           fcout_act.descale_snapshot, w_out.fwd8, w_out.descale, sb.ffn_out, H, s);
        } else {
            gemm_bf16_nt(handle, BS, H, sb.I, 1.f,
                         sb.moe_expert_scratch, sb.I, ampere_compute_weight(lw.trunk_fc_out_w), sb.I, 0.f,
                         sb.ffn_out, H);
        }
        if (dbg) debug_print_activation_stats("moe_trunk_out(ffn_out)", sb.ffn_out, static_cast<std::size_t>(BS) * H, s);
    } else {
        bf16_zero(sb.ffn_out, static_cast<std::size_t>(BS) * H, s);
    }

    const int fcout_which_base = lw.moe_shared_trunk ? 1 : 0;
    const bool moe_grouped_will_run = moe_grouped_device_schedule_enabled(lw);
    const bool selected_expert_fwd =
        moe_selected_expert_forward_enabled(lw) && (moe_grouped_will_run || !fp4n);
    if (selected_expert_fwd) {
        moe_build_expert_row_lists(
            sb.moe_logits, sb.moe_dispatch_counts, sb.moe_dispatch_indices,
            BS, lw.num_experts, s);
        const std::size_t saved_count_off = static_cast<std::size_t>(layer_idx) * lw.num_experts;
        const std::size_t saved_index_off = saved_count_off * BS;
        IDA_CUDA_CHECK(cudaMemcpyAsync(
            sb.moe_dispatch_counts_saved + saved_count_off, sb.moe_dispatch_counts,
            static_cast<std::size_t>(lw.num_experts) * sizeof(int),
            cudaMemcpyDeviceToDevice, s));
        IDA_CUDA_CHECK(cudaMemcpyAsync(
            sb.moe_dispatch_indices_saved + saved_index_off, sb.moe_dispatch_indices,
            static_cast<std::size_t>(lw.num_experts) * BS * sizeof(int),
            cudaMemcpyDeviceToDevice, s));

        if (moe_grouped_will_run) {
            moe_expert_offsets_from_counts(
                sb.moe_dispatch_counts, sb.moe_dispatch_offsets, lw.num_experts, s);
            if (fp8) {
                Fp8Slot& w_in = moe_expert_in_slot(f8, layer_idx, lw.moe_shared_trunk);
                Fp8Slot& w_out = moe_expert_out_slot(f8, layer_idx, lw.moe_shared_trunk);
                moe_grouped_expert_forward_fp8(
                    fcin_act_buf, fcin_descale, fcin_type == CUDA_R_8F_E5M2 ? 1 : 0,
                    sb.moe_logits,
                    sb.moe_dispatch_counts, sb.moe_dispatch_offsets, sb.moe_dispatch_indices,
                    w_in.fwd8, w_in.descale, w_out.fwd8, w_out.descale,
                    sb.ffn_out, BS, H, Ie, lw.num_experts, BS * lw.top_k, s);
            } else {
                moe_grouped_expert_forward_bf16(
                    sb.moe_hidden_gated, sb.moe_logits,
                    sb.moe_dispatch_counts, sb.moe_dispatch_offsets, sb.moe_dispatch_indices,
                    ampere_compute_weight(lw.expert_fc_in_w), ampere_compute_weight(lw.expert_fc_out_w),
                    sb.ffn_out, BS, H, Ie, lw.num_experts, BS * lw.top_k, s);
            }
            dump_intermediate_bf16("mixed", sb.ffn_out, static_cast<std::size_t>(BS) * H,
                                   {BS, H}, g_debug_micro_step, s);
            return;
        }

        IDA_CUDA_CHECK(cudaMemcpyAsync(
            sb.moe_dispatch_counts_host.data(), sb.moe_dispatch_counts,
            static_cast<std::size_t>(lw.num_experts) * sizeof(int),
            cudaMemcpyDeviceToHost, s));
        IDA_CUDA_CHECK(cudaStreamSynchronize(s));
        std::copy(
            sb.moe_dispatch_counts_host.begin(),
            sb.moe_dispatch_counts_host.begin() + lw.num_experts,
            sb.moe_dispatch_counts_host_saved.begin() + saved_count_off);

        for (int e = 0; e < lw.num_experts; ++e) {
            const int n_active = sb.moe_dispatch_counts_host[static_cast<std::size_t>(e)];
            if (n_active <= 0) continue;
            const int* row_idx = sb.moe_dispatch_indices + static_cast<std::size_t>(e) * BS;
            if (fp8) {
                moe_gather_rows_u8(
                    static_cast<const std::uint8_t*>(fcin_act_buf),
                    reinterpret_cast<std::uint8_t*>(sb.moe_fp4_decode_scratch),
                    row_idx, n_active, H, s);
                Fp8Slot& w_in = moe_expert_in_slot(f8, layer_idx, lw.moe_shared_trunk);
                const std::size_t in_off = static_cast<std::size_t>(e) * Ie * H;
                lt_gemm_fp8_nt(f8, n_active, Ie, H,
                               sb.moe_fp4_decode_scratch, H, fcin_type, fcin_descale,
                               static_cast<const std::uint8_t*>(w_in.fwd8) + in_off, w_in.descale,
                               sb.moe_expert_scratch, Ie, s);
            } else {
                const __nv_bfloat16* fc_in_w = ampere_compute_weight(lw.expert_fc_in_w) + static_cast<std::size_t>(e) * Ie * H;
                moe_gather_rows_bf16(sb.moe_hidden_gated, sb.moe_expert_out, row_idx, n_active, H, s);
                gemm_bf16_nt(handle, n_active, Ie, H, 1.f,
                             sb.moe_expert_out, H, fc_in_w, H, 0.f,
                             sb.moe_expert_scratch, Ie);
            }
            gelu_forward(sb.moe_expert_scratch, sb.moe_expert_scratch,
                          static_cast<std::size_t>(n_active) * Ie, s);
            if (probe != 0) {
                moe_fake_quant_roundtrip(sb.moe_expert_scratch, sb.moe_expert_scratch,
                                          sb.moe_probe_amax, probe_level,
                                          static_cast<std::size_t>(n_active) * Ie, s);
            }
            if (fp8) {
                ActSlot& fcout_act = moe_fcout_act_slot(f8, layer_idx, fcout_which_base + e);
                fp8_quant_act(f8, fcout_act, sb.moe_expert_scratch,
                              static_cast<std::size_t>(n_active) * Ie, s);
                Fp8Slot& w_out = moe_expert_out_slot(f8, layer_idx, lw.moe_shared_trunk);
                const std::size_t out_off = static_cast<std::size_t>(e) * H * Ie;
                lt_gemm_fp8_nt(f8, n_active, H, Ie, f8.act8, Ie,
                               act_fp8_type(fcout_act), fcout_act.descale_snapshot,
                               static_cast<const std::uint8_t*>(w_out.fwd8) + out_off, w_out.descale,
                               sb.moe_expert_out, H, s);
            } else {
                const __nv_bfloat16* fc_out_w = ampere_compute_weight(lw.expert_fc_out_w) + static_cast<std::size_t>(e) * H * Ie;
                gemm_bf16_nt(handle, n_active, H, Ie, 1.f,
                             sb.moe_expert_scratch, Ie, fc_out_w, Ie, 0.f,
                             sb.moe_expert_out, H);
            }
            moe_scatter_accumulate_rows_bf16(
                sb.ffn_out, sb.moe_expert_out, sb.moe_logits, row_idx,
                n_active, e, lw.num_experts, H, s);
        }
        dump_intermediate_bf16("mixed", sb.ffn_out, static_cast<std::size_t>(BS) * H,
                               {BS, H}, g_debug_micro_step, s);
        return;
    }

    const bool fuse_expert_fcin = moe_fused_expert_fcin_enabled(lw, sb.I);
    if (fuse_expert_fcin) {
        const int all_Ie = lw.num_experts * Ie;
        if (fp8) {
            Fp8Slot& w_in = moe_expert_in_slot(f8, layer_idx, lw.moe_shared_trunk);
            lt_gemm_fp8_nt(f8, BS, all_Ie, H,
                           fcin_act_buf, H, fcin_type, fcin_descale,
                           w_in.fwd8, w_in.descale,
                           sb.moe_expert_scratch, all_Ie, s);
        } else {
            gemm_bf16_nt(handle, BS, all_Ie, H, 1.f,
                         sb.moe_hidden_gated, H, ampere_compute_weight(lw.expert_fc_in_w), H, 0.f,
                         sb.moe_expert_scratch, all_Ie);
        }
    }
    for (int e = 0; e < lw.num_experts; ++e) {
        __nv_bfloat16* expert_fcin = sb.moe_expert_scratch;
        if (fuse_expert_fcin) {
            moe_copy_expert_fcin_slice_bf16(
                sb.moe_expert_scratch, sb.moe_expert_scratch2,
                BS, lw.num_experts, Ie, e, s);
            expert_fcin = sb.moe_expert_scratch2;
        } else if (fp8) {
            Fp8Slot& w_in = moe_expert_in_slot(f8, layer_idx, lw.moe_shared_trunk);
            const std::size_t in_off = static_cast<std::size_t>(e) * Ie * H;
            lt_gemm_fp8_nt(f8, BS, Ie, H,
                           fcin_act_buf, H, fcin_type, fcin_descale,
                           static_cast<const std::uint8_t*>(w_in.fwd8) + in_off, w_in.descale,
                           expert_fcin, Ie, s);
        } else {
            const __nv_bfloat16* fc_in_w = ampere_compute_weight(lw.expert_fc_in_w) + static_cast<std::size_t>(e) * Ie * H;
            gemm_bf16_nt(handle, BS, Ie, H, 1.f,
                         sb.moe_hidden_gated, H, fc_in_w, H, 0.f,
                         expert_fcin, Ie);
        }
        gelu_forward(expert_fcin, expert_fcin,
                      static_cast<std::size_t>(BS) * Ie, s);
        if (probe != 0) {
            moe_fake_quant_roundtrip(expert_fcin, expert_fcin,
                                      sb.moe_probe_amax, probe_level,
                                      static_cast<std::size_t>(BS) * Ie, s);
        }
        if (fp8) {
            ActSlot& fcout_act = moe_fcout_act_slot(f8, layer_idx, fcout_which_base + e);
            const void* fcout_buf;
            if (fp4n) {
                // expert_fcin, NOT sb.moe_expert_scratch: with the fc_in GEMM
                // fused across experts, scratch holds the CONCATENATED
                // un-gelu'd [BS, E*Ie] block and the gelu'd per-expert slice
                // lives in expert_fcin. Reading scratch here consumed pre-GELU
                // values sliced across the wrong axis; the element count
                // happens to match, so nothing faulted. In the non-fused path
                // expert_fcin IS scratch, which is how it hid.
                moe_pack_fp4_act(fcout_act.amax, fcout_act.scale, fcout_act.descale,
                                  fcout_act.scale_snapshot, fcout_act.descale_snapshot,
                                  expert_fcin, sb.moe_fp4_packed_fcout,
                                  static_cast<std::size_t>(BS) * Ie, /*is_recompute=*/false, s);
                moe_decode_packed_fp4_to_e4m3(sb.moe_fp4_packed_fcout, sb.moe_fp4_decode_scratch,
                                              fcout_act.descale_snapshot, static_cast<std::size_t>(BS) * Ie, s);
                fcout_buf = sb.moe_fp4_decode_scratch;
            } else {
                // Same wrong buffer as the fp4 branch above: must be the
                // gelu'd per-expert slice, not the fused concatenated block.
                fp8_quant_act(f8, fcout_act, expert_fcin, static_cast<std::size_t>(BS) * Ie, s);
                fcout_buf = f8.act8;
            }
            Fp8Slot& w_out = moe_expert_out_slot(f8, layer_idx, lw.moe_shared_trunk);
            const std::size_t out_off = static_cast<std::size_t>(e) * H * Ie;
            lt_gemm_fp8_nt(f8, BS, H, Ie, fcout_buf, Ie,
                           fp4n ? CUDA_R_8F_E4M3 : act_fp8_type(fcout_act),
                           fcout_act.descale_snapshot,
                           static_cast<const std::uint8_t*>(w_out.fwd8) + out_off, w_out.descale,
                           sb.moe_expert_out, H, s);
        } else {
            const __nv_bfloat16* fc_out_w = ampere_compute_weight(lw.expert_fc_out_w) + static_cast<std::size_t>(e) * H * Ie;
            // expert_fcin, not scratch: third of the three forward sites that
            // read the concatenated pre-GELU block under FUSE_EXPERT_FCIN=1.
            gemm_bf16_nt(handle, BS, H, Ie, 1.f,
                         expert_fcin, Ie, fc_out_w, Ie, 0.f,
                         sb.moe_expert_out, H);
        }
        if (dbg && e == 0) {
            debug_print_activation_stats("moe_expert0_gelu", expert_fcin, static_cast<std::size_t>(BS) * Ie, s);
            debug_print_activation_stats("moe_expert0_out", sb.moe_expert_out, static_cast<std::size_t>(BS) * H, s);
        }
        moe_scale_accumulate_bf16(sb.ffn_out, sb.moe_expert_out, sb.moe_logits,
                                   BS, e, lw.num_experts, H, s);
        if (dbg && e == 0) {
            debug_print_activation_stats("moe_mixed_after_e0", sb.ffn_out, static_cast<std::size_t>(BS) * H, s);
        }
    }
    // Cross-backend parity dump (Phase 4): "mixed" matches scripts/
    // native_parity_check.py's fixture key -- sb.ffn_out here is the exact
    // real-code-path equivalent of the fixture's expected_mixed_per_position.
    dump_intermediate_bf16("mixed", sb.ffn_out, static_cast<std::size_t>(BS) * H,
                           {BS, H}, g_debug_micro_step, s);
}

// ─── forward pass ────────────────────────────────────────────────────────────
// Saves each layer's input in sb.saved for the recompute-based backward.

static void forward(
    cublasHandle_t handle,
    AttentionBackendKind attention_backend,
    Fp8Ctx& f8,
    PackedFp4AttentionCtx& fp4_attn,
    const NativeRequest& request,
    const LatticeWeights& w,
    const uint32_t* d_tokens,
    StepBuffers& sb,
    NativeArena& arena,
    bool run_embedding = true,
    bool run_output = true
) {
    const int BS = sb.B * sb.S;
    const int H  = sb.H;
    const int I  = sb.I;
    const int V  = sb.V;
    cudaStream_t s = arena.stream;
    const std::size_t bsh = static_cast<std::size_t>(BS) * H;
    if (run_embedding) {
        ++g_debug_micro_step;
        embedding_forward(d_tokens, w.embed, sb.hidden, sb.B, sb.S, H, s);
        if (is_gpt2_contract(request)) {
            position_embedding_forward(sb.hidden, w.position_embeddings, sb.segs,
                                       sb.B, sb.S, H, w.max_position_embeddings, s);
        }
    }

    std::fill(sb.lss_feedback_skip_layer.begin(), sb.lss_feedback_skip_layer.end(), 0);
    sb.lss_feedback_skip_tail = 0;
    if (lss_feedback_skip_enabled() && sb.lrss_s && sb.lrss_s->lss_feedback_ready &&
            sb.lss_feedback_completed_optimizer_steps >= lss_feedback_min_optimizer_steps()) {
        const int tail = std::min(lss_feedback_max_tail_layers(), w.num_layers);
        if (tail > 0 && sb.lrss_s->lss_last_aux <= lss_feedback_aux_max()) {
            sb.lss_feedback_skip_tail = tail;
            for (int l = w.num_layers - tail; l < w.num_layers; ++l) {
                if (l >= 0) sb.lss_feedback_skip_layer[static_cast<std::size_t>(l)] = 1;
            }
        }
    }

    for (int l = 0; l < w.num_layers; ++l) {
        // Save layer input for backward recompute
        saved_fwd_stash_begin(sb, l, s);
        IDA_CUDA_CHECK(cudaMemcpyAsync(saved_slot(sb, l), sb.hidden,
            bsh * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, s));
        saved_fwd_stash_end(sb, l, s);

        layer_forward_body(handle, attention_backend, f8, fp4_attn, request, l, w.layers[l], saved_slot(sb, l), sb, s);

        // Cognitive-architecture sparse MoE router (Step 3, forward-only):
        // no-op when this layer's num_experts==0 (dense path below runs
        // completely unchanged). Computes sb.moe_logits (final route
        // weights); Step 4 below consumes them via moe_expert_bank_forward.
        //
        // Exactly once. This call and its comment block were duplicated
        // verbatim (a merge artifact by the shape of it), so every MoE layer
        // ran its whole router pipeline twice per micro-step on the forward,
        // including the k_moe_pool_scatter pair measured at ~20% of GPU time.
        // Idempotent, so values never changed and the parity harness, which
        // calls the sequence once and compares values, could not see it.
        //
        // Generic per-token top-k SwiGLU MoE (moe_kind==1) is a completely
        // separate function -- no pooling/pressure/lateral-inhibition inputs
        // exist for it, and unlike the legacy split below it already writes
        // the final mixed result straight into sb.ffn_out in this one call
        // (see the moe_kind==1 branch after Step 4's dispatch below).
        if (w.layers[l].moe_kind == 1) {
            moe_generic_forward(w.layers[l], sb, s);
        } else {
            moe_router_forward(w.layers[l], sb, s);
        }

        // 13. Down projection + 14. residual
        // The clip lives inside the two DENSE branches, never above the
        // if-else: routed layers return from layer_forward_body before the
        // gate/up/SwiGLU stage, so sb.swiglu_out is never written for them and
        // an unconditional clip would read a stale buffer.
        // Within the dense branches it IS unconditional -- it was once gated
        // inside f8.on only (same bug class as attn_out/sb.hidden, found
        // 2026-07-09), which left the BF16 profile never clipping swiglu_out
        // at all, an open operand for gl.g_down's activation side.
        if (w.layers[l].num_experts > 0 && w.layers[l].moe_kind == 1) {
            // Generic per-token top-k SwiGLU MoE already computed AND mixed
            // the full result into sb.ffn_out inside moe_generic_forward
            // above (Step 3/Step 4 are one call for this mode) -- nothing
            // further to do here.
        } else if (w.layers[l].num_experts > 0) {
            // Sparse expert bank replaces the dense down-projection entirely
            // for this layer -- sb.swiglu_out (the dense gate/up/SwiGLU path)
            // is skipped by layer_forward_body; only
            // sb.moe_hidden_gated (moe_router_forward's PressureField-gated
            // hidden) feeds the expert bank. Branches internally on f8.on
            // (Step 4b: FP8 for the trunk/expert fc_in/fc_out GEMMs, router
            // projections stay bf16/f32 -- see the Fp8Ctx struct comment).
            moe_expert_bank_forward(handle, f8, l, w.layers[l], sb, s);
        } else if (f8.on) {
            act_row_clip(request, sb.swiglu_out, BS, I, sb.rowdot, s);
            ActSlot& down_act = layer_act_slot(f8, l, ACT_DOWN_IN);
            fp8_quant_act(f8, down_act, sb.swiglu_out, static_cast<std::size_t>(BS) * I, s);
            // descale_snapshot, never the live descale — fp8_quant_act has
            // already advanced slot.descale past the value matching act8.
            lt_gemm_fp8_nt(f8, BS, H, I, f8.act8, I, act_fp8_type(down_act),
                           down_act.descale_snapshot,
                           f8.slots[l * 7 + F8_DOWN].fwd8, f8.slots[l * 7 + F8_DOWN].descale,
                           sb.ffn_out, H, s);
        } else {
            act_row_clip(request, sb.swiglu_out, BS, I, sb.rowdot, s);
            gemm_bf16(handle, BS, H, I, 1.f,
                      sb.swiglu_out, I,
                      ampere_compute_weight(w.layers[l].down_proj), H, 0.f,
                      sb.ffn_out, H);
        }
        if (w.layers[l].ffn_out_bias) {
            bias_add_strided_bf16(sb.ffn_out, ampere_compute_weight(w.layers[l].ffn_out_bias), BS, H, H, 0, s);
        }
        // FFN-stage amplification probe: the l0-bisect series showed the
        // residual stream jumping from rms ~22 (post-attn-residual, clipped)
        // to rms ~1248 at the NEXT layer's input — i.e. the amplification
        // lives in this FFN block, whose output lands in the residual via
        // the un-clipped inplace-add below. Xavier math predicts ffn_out
        // rms ~3; these prints find which stage breaks that prediction.
        if (gradnorm_debug_enabled() && l <= 1 && w.layers[l].num_experts <= 0) {
            debug_print_activation_stats("gate_out",   sb.gate_out,   static_cast<std::size_t>(BS) * I, s);
            debug_print_activation_stats("up_out",     sb.up_out,     static_cast<std::size_t>(BS) * I, s);
            debug_print_activation_stats("swiglu_out", sb.swiglu_out, static_cast<std::size_t>(BS) * I, s);
            debug_print_activation_stats("ffn_out",    sb.ffn_out,    bsh, s);
        }
        // ── PSS Stage 2: tail-layer FFN-output predictor (shadow) ───────────
        // Estimate computed from the dense pre-layer hidden state for dense
        // bodies, or from moe_hidden_gated for routed/expert bodies when
        // IDA_NATIVE_PSS_CONDITIONING is auto/1. That gives PSS the same
        // pressure/routing-modulated input surface consumed by the expert
        // bank without adding checkpoint-shape-changing predictor tensors.
        // Scored against the real sb.ffn_out computed just above, ahead of
        // any feedback residual-scaling, so the comparison is always against
        // the true unmodified tail FFN output.
        // Never writes sb.ffn_out or sb.hidden: this is read-only w.r.t. the
        // body in every mode Stage 2 supports.
        // Gated on the per-burn buffer state (allocated iff w.pss_pred_rank>0
        // from the request), NOT the static-cached env flag -- the worker
        // process serves many burns and its cached env can't represent a
        // per-burn, per-family decision.
        if (l == w.num_layers - 1 && sb.pss_pred_hidden) {
            // forward() is only ever the true forward pass (see its call
            // sites -- the activation-checkpoint recompute path calls
            // layer_forward_body directly with is_recompute=true instead of
            // going through this function), so that flag is always false
            // here.
            const bool is_recompute = false;
            const int R = w.pss_pred_rank;
            const std::size_t bsr = static_cast<std::size_t>(BS) * R;
            const bool condition_pss =
                pss_conditioning_enabled(request, w.layers[l].num_routes, w.layers[l].num_experts) &&
                sb.moe_hidden_gated;
            sb.pss_conditioning_active = condition_pss;
            const __nv_bfloat16* pss_pred_input =
                condition_pss ? sb.moe_hidden_gated : saved_slot(sb, l);
            if (gradnorm_debug_enabled()) {
                debug_print_activation_stats(
                    condition_pss ? "pss_input_gated" : "pss_input_dense",
                    pss_pred_input, bsh, s);
            }
            gemm_bf16(handle, BS, R, H, 1.f, pss_pred_input, H,
                      w.pss_pred_down, R, 0.f, sb.pss_pred_hidden, R);
            k_relu_bf16<<<ceildiv(bsr, 256), 256, 0, s>>>(sb.pss_pred_hidden, bsr);
            if (gradnorm_debug_enabled()) {
                debug_print_activation_stats("pss_pred_hidden", sb.pss_pred_hidden, bsr, s);
            }
            gemm_bf16(handle, BS, H, R, 1.f, sb.pss_pred_hidden, R,
                      w.pss_pred_up, H, 0.f, sb.pss_pred_ffn, H);
            if (gradnorm_debug_enabled()) {
                debug_print_activation_stats("pss_pred_ffn", sb.pss_pred_ffn, bsh, s);
                debug_print_activation_stats("pss_target_ffn", sb.ffn_out, bsh, s);
            }
            // Scoring and delayed-scale advancement belong to the true
            // forward only. The activation-checkpoint recompute still builds
            // pss_pred_ffn so backward can differentiate the predictor, but
            // must not reset/readback state or advance the int2 reference a
            // second time for the same logical micro-step.
            if (!is_recompute) {
                ++sb.pss_scored_micros;
                // Clear only the per-call score fields. The delayed RMS and
                // confidence range live in the same allocation but persist
                // across score calls and optimizer windows respectively.
                IDA_CUDA_CHECK(cudaMemsetAsync(
                    sb.pss_fence_metrics, 0, 5 * sizeof(float), s));
                // Not aliased into pss_fence_metrics (same reasoning as the
                // int2-block arrays above it): separate allocation, cleared
                // every micro-step so the readback below stays a last-
                // micro-step snapshot, matching err_sq/target_sq/covered.
                IDA_CUDA_CHECK(cudaMemsetAsync(
                    sb.pss_mag_bucket_err_sq, 0, kPssMagBuckets * sizeof(float), s));
                IDA_CUDA_CHECK(cudaMemsetAsync(
                    sb.pss_mag_bucket_target_sq, 0, kPssMagBuckets * sizeof(float), s));
                IDA_CUDA_CHECK(cudaMemsetAsync(
                    sb.pss_mag_bucket_covered, 0, kPssMagBuckets * sizeof(float), s));
                IDA_CUDA_CHECK(cudaMemsetAsync(
                    sb.pss_mag_bucket_count, 0, kPssMagBuckets * sizeof(float), s));
                // covered buffer is float-allocated (4 bytes) but counted as uint
                // in the kernel -- see k_pss_pred_score's saturation comment.
                const int int2_block_size = pss_int2_block_size();
                k_pss_pred_score<<<ceildiv(bsh, 256), 256, 0, s>>>(
                    sb.pss_pred_ffn, sb.ffn_out, pss_pred_eps(), bsh,
                    sb.pss_pred_err_sq, sb.pss_pred_target_sq,
                    reinterpret_cast<unsigned int*>(sb.pss_pred_covered),
                    sb.pss_int2_inv_rms,
                    reinterpret_cast<unsigned int*>(sb.pss_int2_matched),
                    reinterpret_cast<unsigned int*>(sb.pss_int2_scored),
                    H, int2_block_size, sb.pss_int2_num_blocks,
                    sb.pss_int2_target_sq_blocks, sb.pss_int2_inv_rms_blocks,
                    sb.pss_mag_bucket_err_sq, sb.pss_mag_bucket_target_sq,
                    reinterpret_cast<unsigned int*>(sb.pss_mag_bucket_covered),
                    reinterpret_cast<unsigned int*>(sb.pss_mag_bucket_count));
                k_pss_record_confidence<<<1, 1, 0, s>>>(
                    sb.pss_pred_covered, bsh, sb.pss_confidence_minmax);
                if (int2_block_size > 0) {
                    k_pss_update_int2_inv_rms_blocks<<<
                        ceildiv(sb.pss_int2_num_blocks, 256), 256, 0, s>>>(
                        sb.pss_int2_target_sq_blocks, bsh, sb.pss_int2_num_blocks,
                        sb.pss_int2_inv_rms_blocks,
                        1.0f / static_cast<float>(pss_int2_scale_window()),
                        sb.pss_int2_target_sq_blocks);
                } else {
                    k_pss_update_int2_inv_rms<<<1, 1, 0, s>>>(
                        sb.pss_pred_target_sq, bsh, sb.pss_int2_inv_rms,
                        1.0f / static_cast<float>(pss_int2_scale_window()));
                }
                sb.pss_last_blend_frac = sb.pss_engaged_frac;
            }
            // PSS Stage 4 actuation: blend AFTER scoring (so the confidence
            // measurement above always compares against the true unblended
            // real output, every step) and BEFORE any LSS feedback scaling
            // below, so the two mechanisms compose as sequential operations
            // on the same buffer rather than racing. sb.pss_engaged_frac is
            // 0 unless IDA_NATIVE_PSS_GOVERNOR=1 has ramped it up at the
            // fence, so this is a no-op call site in every mode validated
            // on GPU so far (shadow mode never sets it above 0).
            if (sb.pss_engaged_frac > 0.0f) {
                k_pss_blend_bf16<<<ceildiv(bsh, 256), 256, 0, s>>>(
                    sb.ffn_out, sb.pss_pred_ffn, bsh, sb.pss_engaged_frac);
            }
        }
        if (!sb.lss_feedback_skip_layer.empty() &&
                sb.lss_feedback_skip_layer[static_cast<std::size_t>(l)]) {
            const float feedback_scale = lss_feedback_residual_scale();
            if (feedback_scale < 1.0f) {
                k_scale_bf16<<<ceildiv(bsh, 256), 256, 0, s>>>(
                    sb.ffn_out, bsh, feedback_scale);
            }
        }
        IDA_LAUNCH(k_inplace_add_bf16, "ffn.residual_add", ceildiv(bsh, 256), 256, 0, s,
                   sb.hidden, sb.ffn_out, bsh);
        if (gradnorm_debug_enabled() && l <= 1) {
            debug_print_activation_stats("layer_out", sb.hidden, bsh, s);
        }
    }

    // LRSS multiscale-memory injection: pool → bank query → gated delta
    // broadcast onto the residual stream, BEFORE the final-norm input is
    // saved (so the recompute-free backward sees the injected hidden).
    if (!run_output) return;

    if (sb.lrss_s) {
        // forward() runs once per micro-step (unlike layer_forward_body,
        // which activation-checkpoint recompute re-enters) -- always the
        // true-forward call, confirmed by this call site's scope (no
        // is_recompute parameter here at all).
        lrss_forward(*sb.lrss_p, *sb.lrss_s, sb.hidden, d_tokens,
                     lrss_common_token_mask(V), sb.B, sb.S, H, s, /*is_recompute=*/false);
    }

    // Save final-norm input
    saved_fwd_stash_begin(sb, w.num_layers, s);
    IDA_CUDA_CHECK(cudaMemcpyAsync(saved_slot(sb, w.num_layers), sb.hidden,
        bsh * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, s));
    saved_fwd_stash_end(sb, w.num_layers, s);

    // Final norm.  LM head + CE run fused and chunked in the backward pass —
    // the full [B*S, V] logits tensor is never materialized.
    if (is_gpt2_contract(request)) layernorm_forward(sb.hidden, w.final_norm, w.final_norm_bias,
                                                      sb.normed, sb.rms_save, BS, H, model_norm_eps(request), s);
    else rmsnorm_forward(sb.hidden, w.final_norm, sb.normed, sb.rms_save, BS, H, 1e-6f, s);
    (void)V;
}

// ─── backward pass: accumulate full-model gradients (no weight updates) ──────

// ─── cognitive-architecture sparse MoE backward (Step 5) ────────────────────
// Hand-written mirror of moe_router_forward + moe_expert_bank_forward.
// Recompute-based like the rest of this engine: moe_router_forward is
// called again here (it's idempotent -- no delayed-scaling state to desync,
// unlike FP8's is_recompute machinery) to regenerate hidden_gated/
// route_weights/T/c1/p/scores fresh; large per-expert activations
// (pre/post-GELU, expert_out) are similarly recomputed one expert at a
// time rather than all saved, matching this engine's existing philosophy
// throughout. d_mixed is the upstream gradient into what forward wrote as
// sb.ffn_out (i.e. d_ffn_residual at the call site) -- this function's own
// output is sb.d_normed, handed off to the SAME rmsnorm_backward call the
// dense path already uses.
//
// FP8-consistent recompute (the activation side): mirrors moe_expert_bank_
// forward's own FP8 branch exactly for the FORWARD-direction GEMMs
// (fc_in/fc_out with W^T, matching lt_gemm_fp8_nt's only supported
// pattern), using is_recompute=true on every fp8_quant_act call so backward
// reuses forward's exact snapshotted scale/descale instead of advancing
// delayed-scaling state a second time -- the same hazard-avoidance the
// dense path's own backward already relies on throughout. This matters for
// real reasons, not ULP-level rounding noise: at low precision, forward and
// backward can disagree by genuine magnitudes if backward recomputes
// pre_gelu/post_gelu/expert_out with a DIFFERENT quantization than forward
// actually used (quantization changes which values saturate/clip, and
// interacts with GELU's curvature and the router's hard top-k/inhibition
// thresholds elsewhere in this pipeline).
//
// The INPUT-gradient GEMMs (d(post_gelu), d(hidden_gated)) deliberately stay
// plain BF16 against the real weight tensor regardless of f8.on --
// lt_gemm_fp8_nt only implements the NT (transposed-weight) pattern
// forward's A@W^T needs; the dense path gets the other direction via a
// SEPARATE bwd8 weight cache, but MoE's expert_fc_in_w/expert_fc_out_w are
// quantized as ONE contiguous multi-expert block (see the Fp8Ctx struct
// comment), so bwd8's transpose-quantize is NOT per-expert-correct there --
// a documented, separate gap. Using the real bf16 weight for this direction
// is correct regardless (only the ACTIVATION values needed to track
// forward's exact precision, which the recompute above already ensures).
// GELU is done out-of-place here (scratch2, not in-place like forward)
// since backward needs BOTH the pre- and post-activation values
// simultaneously, unlike forward which only needs the final result.
// Generic per-token top-k SwiGLU MoE backward (2026-08-23). Mirrors
// moe_backward's own recompute-based-backward philosophy: no per-expert
// forward activations are saved, so each expert's gate/up/swiglu/down chain
// is recomputed here before its gradient is taken. d_mixed is d(sb.ffn_out)
// -- the same incoming gradient legacy's moe_backward receives.
//
// IMPORTANT: moe_small_proj_backward_input OVERWRITES its output (only its
// _backward_weight sibling accumulates via atomicAdd -- verified by reading
// both kernels directly, not assumed from the shared doc comment above
// them). Every use below that needs to ADD a contribution to a running
// total therefore writes into sb.generic_moe_d_hidden_contrib_f32 first,
// then add_inplace_f32's it into the real accumulator.
static void moe_generic_backward(
    int layer_idx, const LatticeLayerWeights& lw, LatticeGradLayer& gl,
    const __nv_bfloat16* d_mixed, StepBuffers& sb, cudaStream_t s
) {
    if (lw.num_experts <= 0) return;
    const int BS = sb.B * sb.S;
    const int H = sb.H;
    const int Ie = lw.expert_intermediate_size;
    const int N = lw.num_experts;
    const std::size_t bsh = static_cast<std::size_t>(BS) * H;
    const std::size_t bsi = static_cast<std::size_t>(BS) * Ie;

    IDA_CUDA_CHECK(cudaMemsetAsync(sb.generic_moe_d_hidden_f32, 0, bsh * sizeof(float), s));

    for (int e = 0; e < N; ++e) {
        const __nv_bfloat16* gate_w = ampere_compute_weight(
            lw.generic_gate_proj_w + static_cast<std::size_t>(e) * Ie * H);
        const __nv_bfloat16* up_w = ampere_compute_weight(
            lw.generic_up_proj_w + static_cast<std::size_t>(e) * Ie * H);
        const __nv_bfloat16* down_w = ampere_compute_weight(
            lw.generic_down_proj_w + static_cast<std::size_t>(e) * H * Ie);

        // Recompute this expert's forward (gate, up, swiglu h, down) --
        // needed both for the row-dot against d_mixed and as the "x" side
        // of every weight-gradient call below.
        moe_small_proj_f32(sb.generic_moe_hidden_f32, gate_w, sb.generic_moe_gate_f32, BS, H, Ie, s);
        moe_small_proj_f32(sb.generic_moe_hidden_f32, up_w, sb.generic_moe_up_f32, BS, H, Ie, s);
        cast_f32_to_bf16(sb.generic_moe_gate_f32, sb.generic_moe_gate_bf16, bsi, s);
        cast_f32_to_bf16(sb.generic_moe_up_f32, sb.generic_moe_up_bf16, bsi, s);
        swiglu_forward(sb.generic_moe_gate_bf16, sb.generic_moe_up_bf16, sb.generic_moe_h_bf16,
                        static_cast<int>(bsi), s);
        cast_bf16_to_f32(sb.generic_moe_h_bf16, sb.generic_moe_h_f32, bsi, s);
        moe_small_proj_f32(sb.generic_moe_h_f32, down_w, sb.generic_moe_down_f32, BS, Ie, H, s);
        cast_f32_to_bf16(sb.generic_moe_down_f32, sb.generic_moe_down_bf16, bsh, s);

        // d(route_weight[:,e]) = dot(d_mixed, down_bf16_e) -- forward's own
        // product-rule term (ffn_out = sum_e route_weight[:,e]*down_e).
        moe_row_dot_bf16(d_mixed, sb.generic_moe_down_bf16, sb.generic_moe_d_logits, BS, H, N, e, s);

        // d(down_bf16_e) = route_weight[:,e] * d_mixed -- overwrite
        // generic_moe_down_bf16 in place (its forward value was already
        // consumed by the row-dot above), same pattern legacy's
        // moe_backward uses for its own expert_out gradient.
        bf16_zero(sb.generic_moe_down_bf16, bsh, s);
        moe_scale_accumulate_bf16(sb.generic_moe_down_bf16, d_mixed, sb.moe_logits, BS, e, N, H, s);

        cast_bf16_to_f32(sb.generic_moe_down_bf16, sb.generic_moe_down_f32, bsh, s);
        moe_small_proj_backward_input(sb.generic_moe_down_f32, down_w, sb.generic_moe_d_h_f32, BS, Ie, H, s);
        moe_small_proj_backward_weight(
            sb.generic_moe_down_f32, sb.generic_moe_h_f32,
            gl.g_generic_down + static_cast<std::size_t>(e) * H * Ie, BS, Ie, H, s);

        cast_f32_to_bf16(sb.generic_moe_d_h_f32, sb.generic_moe_d_h_bf16, bsi, s);
        swiglu_backward(sb.generic_moe_d_h_bf16, sb.generic_moe_gate_bf16, sb.generic_moe_up_bf16,
                         sb.generic_moe_d_gate_bf16, sb.generic_moe_d_up_bf16,
                         static_cast<int>(bsi), s);
        cast_bf16_to_f32(sb.generic_moe_d_gate_bf16, sb.generic_moe_d_gate_f32, bsi, s);
        cast_bf16_to_f32(sb.generic_moe_d_up_bf16, sb.generic_moe_d_up_f32, bsi, s);

        moe_small_proj_backward_weight(
            sb.generic_moe_d_gate_f32, sb.generic_moe_hidden_f32,
            gl.g_generic_gate + static_cast<std::size_t>(e) * Ie * H, BS, H, Ie, s);
        moe_small_proj_backward_weight(
            sb.generic_moe_d_up_f32, sb.generic_moe_hidden_f32,
            gl.g_generic_up + static_cast<std::size_t>(e) * Ie * H, BS, H, Ie, s);

        moe_small_proj_backward_input(sb.generic_moe_d_gate_f32, gate_w,
                                       sb.generic_moe_d_hidden_contrib_f32, BS, H, Ie, s);
        add_inplace_f32(sb.generic_moe_d_hidden_f32, sb.generic_moe_d_hidden_contrib_f32, bsh, s);
        moe_small_proj_backward_input(sb.generic_moe_d_up_f32, up_w,
                                       sb.generic_moe_d_hidden_contrib_f32, BS, H, Ie, s);
        add_inplace_f32(sb.generic_moe_d_hidden_f32, sb.generic_moe_d_hidden_contrib_f32, bsh, s);
    }

    // ── Shared expert backward ──────────────────────────────────────────────
    // Mirrors the routed-expert loop's recompute-based-backward pattern,
    // but with no routing weight -- the "route weight" is this expert's own
    // per-token sigmoid gate, so the row-dot/scale-accumulate steps below
    // use a [BS,1] tensor (num_experts=1, expert_idx=0) instead of a column
    // of sb.moe_logits.
    if (lw.shared_expert_width > 0) {
        const int Ws = lw.shared_expert_width;
        const std::size_t bsw = static_cast<std::size_t>(BS) * Ws;
        const __nv_bfloat16* sgate_w = ampere_compute_weight(lw.generic_shared_gate_proj_w);
        const __nv_bfloat16* sup_w   = ampere_compute_weight(lw.generic_shared_up_proj_w);
        const __nv_bfloat16* sdown_w = ampere_compute_weight(lw.generic_shared_down_proj_w);
        const __nv_bfloat16* sscore_w = ampere_compute_weight(lw.generic_shared_gate_score_w);

        // Recompute forward: gate_scalar, gate, up, h, down.
        moe_small_proj_f32(sb.generic_moe_hidden_f32, sscore_w,
                            sb.generic_moe_shared_gate_scalar_f32, BS, H, 1, s);
        sigmoid_inplace_f32(sb.generic_moe_shared_gate_scalar_f32, static_cast<std::size_t>(BS), s);

        moe_small_proj_f32(sb.generic_moe_hidden_f32, sgate_w, sb.generic_moe_gate_f32, BS, H, Ws, s);
        moe_small_proj_f32(sb.generic_moe_hidden_f32, sup_w, sb.generic_moe_up_f32, BS, H, Ws, s);
        cast_f32_to_bf16(sb.generic_moe_gate_f32, sb.generic_moe_gate_bf16, bsw, s);
        cast_f32_to_bf16(sb.generic_moe_up_f32, sb.generic_moe_up_bf16, bsw, s);
        swiglu_forward(sb.generic_moe_gate_bf16, sb.generic_moe_up_bf16, sb.generic_moe_h_bf16,
                        static_cast<int>(bsw), s);
        cast_bf16_to_f32(sb.generic_moe_h_bf16, sb.generic_moe_h_f32, bsw, s);
        moe_small_proj_f32(sb.generic_moe_h_f32, sdown_w, sb.generic_moe_down_f32, BS, Ws, H, s);
        cast_f32_to_bf16(sb.generic_moe_down_f32, sb.generic_moe_down_bf16, bsh, s);

        // d(gate_scalar) = dot(d_mixed, down_bf16) per row -- forward's own
        // product-rule term (ffn_out += gate_scalar * down).
        moe_row_dot_bf16(d_mixed, sb.generic_moe_down_bf16,
                          sb.generic_moe_shared_dgate_scalar_f32, BS, H, 1, 0, s);

        // d(down_bf16) = gate_scalar * d_mixed -- overwrite down_bf16 in
        // place (its forward value was already consumed by the row-dot
        // above), same pattern the routed-expert loop uses.
        bf16_zero(sb.generic_moe_down_bf16, bsh, s);
        moe_scale_accumulate_bf16(sb.generic_moe_down_bf16, d_mixed,
                                   sb.generic_moe_shared_gate_scalar_f32, BS, 0, 1, H, s);

        cast_bf16_to_f32(sb.generic_moe_down_bf16, sb.generic_moe_down_f32, bsh, s);
        moe_small_proj_backward_input(sb.generic_moe_down_f32, sdown_w, sb.generic_moe_d_h_f32, BS, Ws, H, s);
        moe_small_proj_backward_weight(sb.generic_moe_down_f32, sb.generic_moe_h_f32,
                                        gl.g_generic_shared_down, BS, Ws, H, s);

        cast_f32_to_bf16(sb.generic_moe_d_h_f32, sb.generic_moe_d_h_bf16, bsw, s);
        swiglu_backward(sb.generic_moe_d_h_bf16, sb.generic_moe_gate_bf16, sb.generic_moe_up_bf16,
                         sb.generic_moe_d_gate_bf16, sb.generic_moe_d_up_bf16,
                         static_cast<int>(bsw), s);
        cast_bf16_to_f32(sb.generic_moe_d_gate_bf16, sb.generic_moe_d_gate_f32, bsw, s);
        cast_bf16_to_f32(sb.generic_moe_d_up_bf16, sb.generic_moe_d_up_f32, bsw, s);

        moe_small_proj_backward_weight(sb.generic_moe_d_gate_f32, sb.generic_moe_hidden_f32,
                                        gl.g_generic_shared_gate, BS, H, Ws, s);
        moe_small_proj_backward_weight(sb.generic_moe_d_up_f32, sb.generic_moe_hidden_f32,
                                        gl.g_generic_shared_up, BS, H, Ws, s);

        moe_small_proj_backward_input(sb.generic_moe_d_gate_f32, sgate_w,
                                       sb.generic_moe_d_hidden_contrib_f32, BS, H, Ws, s);
        add_inplace_f32(sb.generic_moe_d_hidden_f32, sb.generic_moe_d_hidden_contrib_f32, bsh, s);
        moe_small_proj_backward_input(sb.generic_moe_d_up_f32, sup_w,
                                       sb.generic_moe_d_hidden_contrib_f32, BS, H, Ws, s);
        add_inplace_f32(sb.generic_moe_d_hidden_f32, sb.generic_moe_d_hidden_contrib_f32, bsh, s);

        // Gate-score branch: d(pre-sigmoid) via sigmoid_backward_f32
        // (in-place: d(post) buffer overwritten with d(pre), elementwise-safe).
        sigmoid_backward_f32(sb.generic_moe_shared_gate_scalar_f32,
                              sb.generic_moe_shared_dgate_scalar_f32,
                              sb.generic_moe_shared_dgate_scalar_f32,
                              static_cast<std::size_t>(BS), s);
        moe_small_proj_backward_weight(sb.generic_moe_shared_dgate_scalar_f32, sb.generic_moe_hidden_f32,
                                        gl.g_generic_shared_gate_score, BS, H, 1, s);
        moe_small_proj_backward_input(sb.generic_moe_shared_dgate_scalar_f32, sscore_w,
                                       sb.generic_moe_d_hidden_contrib_f32, BS, H, 1, s);
        add_inplace_f32(sb.generic_moe_d_hidden_f32, sb.generic_moe_d_hidden_contrib_f32, bsh, s);
    }

    // ── Router backward ─────────────────────────────────────────────────────
    // sb.generic_moe_d_logits now holds d(route_weights); sb.moe_logits
    // still holds route_weights itself (untouched since moe_generic_forward
    // populated it -- this mode has no lateral-inhibition stage to mutate it
    // further, unlike the legacy path, so no separate saved snapshot is
    // needed here the way legacy needs moe_topk_p_save).
    if (lw.generic_moe_normalize_topk) {
        moe_norm_by_sum_backward_f32(
            sb.moe_logits, sb.generic_moe_d_logits, sb.moe_topk_row_sum,
            sb.generic_moe_d_logits, BS, N, s);
    }
    // route_weight_i is a hard graph constant (0) at unselected positions --
    // discard whatever the renormalize formula produced there.
    moe_zero_where_zero_f32(sb.moe_logits, sb.generic_moe_d_logits, static_cast<std::size_t>(BS) * N, s);
    // Softmax backward needs the pure pre-top-k probabilities (saved by
    // moe_generic_forward's moe_topk_route_f32 call).
    attn_softmax_backward_f32(sb.moe_scores_save, sb.generic_moe_d_logits, BS, N, s);
    // d(router logits) now in sb.generic_moe_d_logits. logits = router_score(hidden) directly (no pooling/pressure).
    moe_small_proj_backward_input(sb.generic_moe_d_logits, ampere_compute_weight(lw.router_score_w),
                                   sb.generic_moe_d_hidden_contrib_f32, BS, H, N, s);
    add_inplace_f32(sb.generic_moe_d_hidden_f32, sb.generic_moe_d_hidden_contrib_f32, bsh, s);
    moe_small_proj_backward_weight(sb.generic_moe_d_logits, sb.generic_moe_hidden_f32,
                                    gl.g_router_score, BS, H, N, s);

    // Final output target: sb.d_normed, same as every other FFN-variant
    // backward (dense, legacy MoE) -- fed to the shared ffn-norm
    // rmsnorm_backward immediately after this call returns.
    cast_f32_to_bf16(sb.generic_moe_d_hidden_f32, sb.d_normed, bsh, s);
}

static void moe_backward(
    cublasHandle_t handle, Fp8Ctx& f8, int layer_idx, const LatticeLayerWeights& lw, LatticeGradLayer& gl,
    const __nv_bfloat16* d_mixed, StepBuffers& sb, cudaStream_t s
) {
    IDA_GEMM_ROLE(::ida_native::gemm_trace::R_REGION_MOE_BWD);
    if (lw.num_experts <= 0) return;
    const int BS = sb.B * sb.S;
    const int H = sb.H;
    const int Ie = lw.expert_intermediate_size;
    const int num_experts = lw.num_experts;
    const int num_routes = lw.num_routes;
    const std::size_t bsh = static_cast<std::size_t>(BS) * H;
    const int probe = moe_act_precision_mode();
    const bool fp8 = f8.on && probe == 0;
    const bool fp4n = fp8 && moe_native_fp4_enabled();
    const float probe_level = probe == 2 ? kMoeInt2Max : kMoeFp4Max;
    const bool dbg = gradnorm_debug_enabled() && layer_idx <= 1;

    moe_router_forward(lw, sb, s);
    moe_compute_sample_counts(sb.segs, sb.moe_count, sb.B, sb.S, s);
    bf16_zero(sb.moe_d_hidden_gated, bsh, s);

    if (probe != 0) {
        // Must match forward's round-trip exactly (same amax-scaled
        // rounding) since this recompute's gradients are computed against
        // whatever value forward actually used -- same principle as the
        // FP8 is_recompute fix above, applied to this probe instead.
        moe_fake_quant_roundtrip(sb.moe_hidden_gated, sb.moe_hidden_gated, sb.moe_probe_amax, probe_level, bsh, s);
    }

    // Shared fc_in-input quantization (is_recompute=true): same ActSlot,
    // same snapshotted scale/descale forward's own call already captured.
    // fp4n packs (is_recompute=true reuses the snapshot, no fresh amax)
    // then decodes -- same consistency requirement as the plain-fp8 branch.
    ActSlot* fcin_act = nullptr;
    cudaDataType_t fcin_type = CUDA_R_8F_E4M3;
    const float* fcin_descale = nullptr;
    const void* fcin_act_buf = f8.act8;
    if (fp8) {
        fcin_act = &moe_fcin_act_slot(f8, layer_idx);
        if (fp4n) {
            moe_pack_fp4_act(fcin_act->amax, fcin_act->scale, fcin_act->descale,
                              fcin_act->scale_snapshot, fcin_act->descale_snapshot,
                              sb.moe_hidden_gated, sb.moe_fp4_packed_fcin, bsh,
                              /*is_recompute=*/true, s);
            moe_decode_packed_fp4_to_e4m3(sb.moe_fp4_packed_fcin, sb.moe_fp4_decode_scratch,
                                          fcin_act->descale_snapshot, bsh, s);
            fcin_act_buf = sb.moe_fp4_decode_scratch;
        } else {
            fp8_quant_act(f8, *fcin_act, sb.moe_hidden_gated, bsh, s, /*is_recompute=*/true);
            fcin_act_buf = f8.act8;
        }
        fcin_type = fp4n ? CUDA_R_8F_E4M3 : act_fp8_type(*fcin_act);
        fcin_descale = fcin_act->descale_snapshot;
    }

    // ── Expert-bank backward ────────────────────────────────────────────────
    if (lw.moe_shared_trunk) {
        if (fp8) {
            Fp8Slot& w_in = moe_trunk_in_slot(f8, layer_idx);
            lt_gemm_fp8_nt(f8, BS, sb.I, H, fcin_act_buf, H, fcin_type, fcin_descale,
                           w_in.fwd8, w_in.descale, sb.moe_expert_scratch, sb.I, s);
        } else {
            gemm_bf16_nt(handle, BS, sb.I, H, 1.f, sb.moe_hidden_gated, H, ampere_compute_weight(lw.trunk_fc_in_w), H, 0.f,
                         sb.moe_expert_scratch, sb.I);
        }
        gelu_forward(sb.moe_expert_scratch, sb.moe_expert_scratch2, static_cast<std::size_t>(BS) * sb.I, s);
        if (probe != 0) {
            moe_fake_quant_roundtrip(sb.moe_expert_scratch2, sb.moe_expert_scratch2, sb.moe_probe_amax,
                                      probe_level, static_cast<std::size_t>(BS) * sb.I, s);
        }
        // Trunk's weight is fixed at 1 (unconditional, no route gate) -- d(trunk_out)=d_mixed directly.
        gemm_bf16_tn_f32(handle, H, sb.I, BS, 1.f, d_mixed, H, sb.moe_expert_scratch2, sb.I,
                          1.f, gl.g_moe_trunk_out, sb.I);
        // d(post_gelu) = d_mixed @ trunk_fc_out_w -- plain BF16 against the
        // real weight regardless of f8.on; see the matching comment on the
        // per-expert loop's own d(post_gelu) step below for why (lt_gemm_
        // fp8_nt only implements the NT/transposed-weight direction, and
        // this is the other one).
        gemm_bf16(handle, BS, sb.I, H, 1.f, d_mixed, H, ampere_compute_weight(lw.trunk_fc_out_w), sb.I, 0.f,
                  sb.moe_d_expert_scratch, sb.I);
        gelu_backward(sb.moe_expert_scratch, sb.moe_d_expert_scratch, sb.moe_d_expert_scratch,
                      static_cast<std::size_t>(BS) * sb.I, s);
        gemm_bf16_tn_f32(handle, sb.I, H, BS, 1.f, sb.moe_d_expert_scratch, sb.I, sb.moe_hidden_gated, H,
                          1.f, gl.g_moe_trunk_in, H);
        gemm_bf16(handle, BS, H, sb.I, 1.f, sb.moe_d_expert_scratch, sb.I, ampere_compute_weight(lw.trunk_fc_in_w), H,
                  1.f, sb.moe_d_hidden_gated, H);
    }

    const int fcout_which_base = lw.moe_shared_trunk ? 1 : 0;
    const bool selected_expert_bwd = moe_selected_expert_backward_enabled(lw) && !fp4n;
    const bool fuse_expert_fcin = !selected_expert_bwd && moe_fused_expert_fcin_enabled(lw, sb.I);
    if (fuse_expert_fcin) {
        const int all_Ie = num_experts * Ie;
        if (fp8) {
            Fp8Slot& w_in = moe_expert_in_slot(f8, layer_idx, lw.moe_shared_trunk);
            lt_gemm_fp8_nt(f8, BS, all_Ie, H, fcin_act_buf, H, fcin_type, fcin_descale,
                           w_in.fwd8, w_in.descale, sb.moe_expert_scratch, all_Ie, s);
        } else {
            gemm_bf16_nt(handle, BS, all_Ie, H, 1.f, sb.moe_hidden_gated, H,
                         ampere_compute_weight(lw.expert_fc_in_w), H, 0.f, sb.moe_expert_scratch, all_Ie);
        }
    }
    if (selected_expert_bwd) {
        IDA_CUDA_CHECK(cudaMemsetAsync(
            sb.moe_d_logits, 0,
            static_cast<std::size_t>(BS) * num_experts * sizeof(float), s));
        const std::size_t saved_count_off = static_cast<std::size_t>(layer_idx) * num_experts;
        const std::size_t saved_index_off = saved_count_off * BS;
        const int* saved_row_indices = sb.moe_dispatch_indices_saved + saved_index_off;

        for (int e = 0; e < num_experts; ++e) {
            const int n_active = sb.moe_dispatch_counts_host_saved[saved_count_off + static_cast<std::size_t>(e)];
            if (n_active <= 0) continue;
            const int* row_idx = saved_row_indices + static_cast<std::size_t>(e) * BS;
            const __nv_bfloat16* fc_in_w  = ampere_compute_weight(lw.expert_fc_in_w) + static_cast<std::size_t>(e) * Ie * H;
            const __nv_bfloat16* fc_out_w = ampere_compute_weight(lw.expert_fc_out_w) + static_cast<std::size_t>(e) * H * Ie;
            __nv_bfloat16* g_in_slice  = gl.g_moe_expert_in  + static_cast<std::size_t>(e) * Ie * H;
            __nv_bfloat16* g_out_slice = gl.g_moe_expert_out + static_cast<std::size_t>(e) * H * Ie;

            if (fp8) {
                moe_gather_rows_u8(
                    static_cast<const std::uint8_t*>(fcin_act_buf),
                    reinterpret_cast<std::uint8_t*>(sb.moe_fp4_decode_scratch),
                    row_idx, n_active, H, s);
                Fp8Slot& w_in = moe_expert_in_slot(f8, layer_idx, lw.moe_shared_trunk);
                const std::size_t in_off = static_cast<std::size_t>(e) * Ie * H;
                lt_gemm_fp8_nt(f8, n_active, Ie, H,
                               sb.moe_fp4_decode_scratch, H, fcin_type, fcin_descale,
                               static_cast<const std::uint8_t*>(w_in.fwd8) + in_off, w_in.descale,
                               sb.moe_expert_scratch, Ie, s);
            } else {
                moe_gather_rows_bf16(sb.moe_hidden_gated, sb.moe_expert_out, row_idx, n_active, H, s);
                gemm_bf16_nt(handle, n_active, Ie, H, 1.f, sb.moe_expert_out, H,
                             fc_in_w, H, 0.f, sb.moe_expert_scratch, Ie);
            }
            gelu_forward(sb.moe_expert_scratch, sb.moe_expert_scratch2,
                         static_cast<std::size_t>(n_active) * Ie, s);
            if (probe != 0) {
                moe_fake_quant_roundtrip(sb.moe_expert_scratch2, sb.moe_expert_scratch2,
                                          sb.moe_probe_amax, probe_level,
                                          static_cast<std::size_t>(n_active) * Ie, s);
            }
            if (fp8) {
                ActSlot& fcout_act = moe_fcout_act_slot(f8, layer_idx, fcout_which_base + e);
                fp8_quant_act(f8, fcout_act, sb.moe_expert_scratch2,
                              static_cast<std::size_t>(n_active) * Ie, s,
                              /*is_recompute=*/true);
                Fp8Slot& w_out = moe_expert_out_slot(f8, layer_idx, lw.moe_shared_trunk);
                const std::size_t out_off = static_cast<std::size_t>(e) * H * Ie;
                lt_gemm_fp8_nt(f8, n_active, H, Ie, f8.act8, Ie,
                               act_fp8_type(fcout_act), fcout_act.descale_snapshot,
                               static_cast<const std::uint8_t*>(w_out.fwd8) + out_off, w_out.descale,
                               sb.moe_expert_out, H, s);
            } else {
                gemm_bf16_nt(handle, n_active, H, Ie, 1.f, sb.moe_expert_scratch2, Ie,
                             fc_out_w, Ie, 0.f, sb.moe_expert_out, H);
            }

            moe_row_dot_selected_bf16(d_mixed, sb.moe_expert_out, sb.moe_d_logits,
                                      row_idx, n_active, H, num_experts, e, s);
            moe_gather_scale_rows_bf16(d_mixed, sb.moe_expert_out, sb.moe_logits,
                                       row_idx, n_active, e, num_experts, H, s);

            gemm_bf16_tn_f32(handle, H, Ie, n_active, 1.f, sb.moe_expert_out, H,
                              sb.moe_expert_scratch2, Ie, 1.f, g_out_slice, Ie);
            gemm_bf16(handle, n_active, Ie, H, 1.f, sb.moe_expert_out, H,
                      fc_out_w, Ie, 0.f, sb.moe_d_expert_scratch, Ie);
            gelu_backward(sb.moe_expert_scratch, sb.moe_d_expert_scratch, sb.moe_d_expert_scratch,
                          static_cast<std::size_t>(n_active) * Ie, s);

            moe_gather_rows_bf16(sb.moe_hidden_gated, sb.moe_expert_out, row_idx, n_active, H, s);
            gemm_bf16_tn_f32(handle, Ie, H, n_active, 1.f, sb.moe_d_expert_scratch, Ie,
                              sb.moe_expert_out, H, 1.f, g_in_slice, H);
            gemm_bf16(handle, n_active, H, Ie, 1.f, sb.moe_d_expert_scratch, Ie,
                      fc_in_w, H, 0.f, sb.moe_expert_out, H);
            moe_scatter_add_rows_bf16(sb.moe_d_hidden_gated, sb.moe_expert_out,
                                      row_idx, n_active, H, s);
        }
    } else
    for (int e = 0; e < num_experts; ++e) {
        const __nv_bfloat16* fc_in_w  = ampere_compute_weight(lw.expert_fc_in_w) + static_cast<std::size_t>(e) * Ie * H;
        const __nv_bfloat16* fc_out_w = ampere_compute_weight(lw.expert_fc_out_w) + static_cast<std::size_t>(e) * H * Ie;
        __nv_bfloat16* g_in_slice  = gl.g_moe_expert_in  + static_cast<std::size_t>(e) * Ie * H;
        __nv_bfloat16* g_out_slice = gl.g_moe_expert_out + static_cast<std::size_t>(e) * H * Ie;

        // Recompute pre/post-GELU and the expert's own output -- the output
        // is needed (not just its gradient) for d(route_weight[:,e])'s dot
        // product below. FP8-consistent with forward via the shared
        // fcin quantization above and this expert's own fcout ActSlot.
        // expert_pre/expert_post indirection: when the fc_in GEMM was FUSED
        // across all experts in the forward, the per-expert pre-GELU slice is
        // already resident and only needs copying out -- recomputing it here
        // would throw away the whole point of the fusion. Restored 2026-07-31
        // (0a4ebd28 dropped it and hardwired the scratch buffers, which left
        // moe_fused_expert_fcin_enabled() computing a fusion the backward then
        // ignored).
        __nv_bfloat16* expert_pre = sb.moe_expert_scratch;
        __nv_bfloat16* expert_post = sb.moe_expert_scratch2;
        if (fuse_expert_fcin) {
            moe_copy_expert_fcin_slice_bf16(
                sb.moe_expert_scratch, sb.moe_expert_scratch2,
                BS, num_experts, Ie, e, s);
            expert_pre = sb.moe_expert_scratch2;
            expert_post = sb.moe_d_expert_scratch;
        } else if (fp8) {
            Fp8Slot& w_in = moe_expert_in_slot(f8, layer_idx, lw.moe_shared_trunk);
            const std::size_t in_off = static_cast<std::size_t>(e) * Ie * H;
            lt_gemm_fp8_nt(f8, BS, Ie, H, fcin_act_buf, H, fcin_type, fcin_descale,
                           static_cast<const std::uint8_t*>(w_in.fwd8) + in_off, w_in.descale,
                           expert_pre, Ie, s);
        } else {
            gemm_bf16_nt(handle, BS, Ie, H, 1.f, sb.moe_hidden_gated, H, fc_in_w, H, 0.f,
                         expert_pre, Ie);
        }
        gelu_forward(expert_pre, expert_post, static_cast<std::size_t>(BS) * Ie, s);
        if (probe != 0) {
            moe_fake_quant_roundtrip(expert_post, expert_post, sb.moe_probe_amax,
                                      probe_level, static_cast<std::size_t>(BS) * Ie, s);
        }
        if (fp8) {
            ActSlot& fcout_act = moe_fcout_act_slot(f8, layer_idx, fcout_which_base + e);
            const void* fcout_buf;
            if (fp4n) {
                moe_pack_fp4_act(fcout_act.amax, fcout_act.scale, fcout_act.descale,
                                  fcout_act.scale_snapshot, fcout_act.descale_snapshot,
                                  expert_post, sb.moe_fp4_packed_fcout,
                                  static_cast<std::size_t>(BS) * Ie, /*is_recompute=*/true, s);
                moe_decode_packed_fp4_to_e4m3(sb.moe_fp4_packed_fcout, sb.moe_fp4_decode_scratch,
                                              fcout_act.descale_snapshot, static_cast<std::size_t>(BS) * Ie, s);
                fcout_buf = sb.moe_fp4_decode_scratch;
            } else {
                fp8_quant_act(f8, fcout_act, expert_post, static_cast<std::size_t>(BS) * Ie, s,
                              /*is_recompute=*/true);
                fcout_buf = f8.act8;
            }
            Fp8Slot& w_out = moe_expert_out_slot(f8, layer_idx, lw.moe_shared_trunk);
            const std::size_t out_off = static_cast<std::size_t>(e) * H * Ie;
            lt_gemm_fp8_nt(f8, BS, H, Ie, fcout_buf, Ie,
                           fp4n ? CUDA_R_8F_E4M3 : act_fp8_type(fcout_act),
                           fcout_act.descale_snapshot,
                           static_cast<const std::uint8_t*>(w_out.fwd8) + out_off, w_out.descale,
                           sb.moe_expert_out, H, s);
        } else {
            // expert_post, not scratch2: under FUSE_EXPERT_FCIN=1 scratch2
            // holds the PRE-gelu slice (it is expert_pre) and the gelu'd
            // values are in expert_post. Aliased in the non-fused path,
            // which is how it hid.
            gemm_bf16_nt(handle, BS, H, Ie, 1.f, expert_post, Ie, fc_out_w, Ie, 0.f,
                         sb.moe_expert_out, H);
        }

        // d(route_weight[:,e]) = dot(d_mixed, expert_out_e) per row -- mixed
        // = ... + expert_out_e * route_weight[:,e], standard product rule.
        moe_row_dot_bf16(d_mixed, sb.moe_expert_out, sb.moe_d_logits, BS, H, num_experts, e, s);

        // d(expert_out_e) = d_mixed * route_weight[:,e] -- overwrite
        // moe_expert_out in place (its forward value was already consumed
        // by the row-dot above).
        bf16_zero(sb.moe_expert_out, bsh, s);
        moe_scale_accumulate_bf16(sb.moe_expert_out, d_mixed, sb.moe_logits, BS, e, num_experts, H, s);

        // expert_post again: fc_out's dW needs the POST-gelu activation, and
        // scratch2 is pre-gelu when the forward fc_in was fused.
        gemm_bf16_tn_f32(handle, H, Ie, BS, 1.f, sb.moe_expert_out, H, expert_post, Ie,
                          1.f, g_out_slice, Ie);
        // d(post_gelu) = d(expert_out) @ fc_out_w -- plain BF16 against the
        // REAL weight tensor regardless of f8.on. This is deliberate, not a
        // shortcut: lt_gemm_fp8_nt only implements the NT (transposed-
        // weight) pattern that forward's A@W^T convention needs; the
        // dense path's own backward gets this same direction via a
        // SEPARATE bwd8 weight cache (quantized in the opposite layout).
        // MoE's expert_fc_in_w/expert_fc_out_w are quantized as ONE
        // contiguous multi-expert block (see the Fp8Ctx struct comment),
        // so their bwd8 cache's transpose-quantize is NOT per-expert-
        // correct (a documented, separate gap) -- using it here would
        // silently compute a wrong gradient. The real bf16 weight is
        // always exactly correct for this direction regardless of
        // whatever precision forward used for its own GEMM; only the
        // ACTIVATION values (pre/post-GELU, expert_out, all recomputed
        // with matching FP8 quantization above) needed to track forward's
        // real precision to avoid a fwd/bwd mismatch.
        gemm_bf16(handle, BS, Ie, H, 1.f, sb.moe_expert_out, H, fc_out_w, Ie, 0.f,
                  sb.moe_d_expert_scratch, Ie);
        // expert_pre, not scratch: gelu's derivative needs this expert's
        // PRE-gelu slice, and under FUSE_EXPERT_FCIN=1 scratch holds the
        // concatenated [BS, E*Ie] block, which read with leading dimension
        // Ie slices across the wrong axis with a matching element count.
        gelu_backward(expert_pre, sb.moe_d_expert_scratch, sb.moe_d_expert_scratch,
                      static_cast<std::size_t>(BS) * Ie, s);
        gemm_bf16_tn_f32(handle, Ie, H, BS, 1.f, sb.moe_d_expert_scratch, Ie, sb.moe_hidden_gated, H,
                          1.f, g_in_slice, H);
        gemm_bf16(handle, BS, H, Ie, 1.f, sb.moe_d_expert_scratch, Ie, fc_in_w, H,
                  1.f, sb.moe_d_hidden_gated, H);
    }
    if (dbg) {
        debug_print_activation_stats("moe_bwd_d_hidden_gated_expbank", sb.moe_d_hidden_gated, bsh, s);
        debug_print_f32_activation_stats("moe_bwd_d_logits_postexp", sb.moe_d_logits,
                                          static_cast<std::size_t>(BS) * num_experts, s);
    }

    // ── Router backward ─────────────────────────────────────────────────────
    // sb.moe_d_logits now holds d(route_weights) (final, post-inhibition).
    // sb.moe_logits still holds route_weights itself (untouched by the
    // expert-bank loop above) -- both needed for the final renormalize's
    // own norm-by-sum backward.
    moe_norm_by_sum_backward_f32(sb.moe_logits, sb.moe_d_logits, sb.moe_inhib_row_sum, sb.moe_d_logits,
                                  BS, num_experts, s);
    // d(bounded) now in sb.moe_d_logits. Clamp-gate backward -> d(p).
    moe_inhib_clamp_backward_f32(sb.moe_inhib_c1, 0.72f, 0.08f, sb.moe_d_logits,
                                  static_cast<std::size_t>(BS) * num_experts, s);
    // Top-k renormalize backward (sparse->p) needs p itself (saved) -> d(sparse).
    moe_norm_by_sum_backward_f32(sb.moe_topk_p_save, sb.moe_d_logits, sb.moe_topk_row_sum, sb.moe_d_logits,
                                  BS, num_experts, s);
    // sparse_i is a hard graph constant (0) at unselected positions -- discard
    // whatever the renormalize formula produced there.
    moe_zero_where_zero_f32(sb.moe_topk_p_save, sb.moe_d_logits, static_cast<std::size_t>(BS) * num_experts, s);
    // Load-balancing auxiliary gradient. MUST land here: moe_d_logits holds
    // d(scores) pre-softmax, which is where dL_aux/dP_e applies.
    {
        const float bcoef = expert_balance_coef(lw.expert_balancing_loss_coef);
        if (bcoef > 0.0f && num_experts > 1) {
            float* d_sumP = nullptr; float* d_sumF = nullptr;
            const std::size_t eb = static_cast<std::size_t>(num_experts) * sizeof(float);
            if (cudaMallocAsync(&d_sumP, eb, s) == cudaSuccess &&
                cudaMallocAsync(&d_sumF, eb, s) == cudaSuccess) {
                k_moe_balance_stats<<<(num_experts + 63) / 64, 64, 0, s>>>(
                    sb.moe_scores_save, sb.moe_logits, static_cast<int>(BS),
                    num_experts, d_sumP, d_sumF);
                const std::size_t n = BS * static_cast<std::size_t>(num_experts);
                k_moe_balance_grad<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(
                    sb.moe_d_logits, d_sumF, static_cast<int>(BS), num_experts, bcoef);
            }
            if (d_sumP) cudaFreeAsync(d_sumP, s);
            if (d_sumF) cudaFreeAsync(d_sumF, s);
        }
    }
    // Softmax backward needs the pure pre-top-k probabilities (saved).
    attn_softmax_backward_f32(sb.moe_scores_save, sb.moe_d_logits, BS, num_experts, s);
    // d(logits) now in sb.moe_d_logits. logits = router_score(pooled2) + routed_pressure(pressure).
    moe_small_proj_backward_input(sb.moe_d_logits, ampere_compute_weight(lw.router_score_w), sb.moe_d_pooled2, BS, H, num_experts, s);
    moe_small_proj_backward_weight(sb.moe_d_logits, sb.moe_pooled2, gl.g_router_score, BS, H, num_experts, s);
    if (lw.pressure_to_routes_w != nullptr) {
        moe_small_proj_backward_input(sb.moe_d_logits, ampere_compute_weight(lw.pressure_to_routes_w), sb.moe_d_pressure,
                                       BS, num_routes, num_experts, s);
        moe_small_proj_backward_weight(sb.moe_d_logits, sb.moe_pressure, gl.g_pressure_to_routes,
                                        BS, num_routes, num_experts, s);
    } else {
        // Identity branch (num_routes==num_experts): pass-through, no params.
        IDA_CUDA_CHECK(cudaMemcpyAsync(sb.moe_d_pressure, sb.moe_d_logits,
            BS * static_cast<std::size_t>(num_routes) * sizeof(float), cudaMemcpyDeviceToDevice, s));
    }
    if (dbg) {
        debug_print_f32_activation_stats("moe_bwd_d_pooled2", sb.moe_d_pooled2, bsh, s);
        debug_print_f32_activation_stats("moe_bwd_d_pressure_route", sb.moe_d_pressure,
                                          BS * static_cast<std::size_t>(num_routes), s);
    }

    // Reduce d(pooled2) back into sb.moe_d_hidden_gated (ADD -- combines
    // with the expert-bank's own contribution already there, since
    // pooled2 = pool(hidden_gated)).
    moe_pool_by_sample_backward(sb.moe_d_pooled2, sb.segs, sb.moe_count, sb.moe_start_grad_scratch,
                                 sb.moe_d_hidden_gated, sb.B, sb.S, H, s);
    if (dbg) {
        debug_print_activation_stats("moe_bwd_d_hidden_gated_total", sb.moe_d_hidden_gated, bsh, s);
    }

    // hidden_gated = normed2 * modulation backward. sb.d_normed is this
    // function's final output target (fed to the existing ffn-norm
    // rmsnorm_backward below, same as the dense path) -- zero it fresh
    // since this replaces the dense path's beta=0 GEMM writes there.
    bf16_zero(sb.d_normed, bsh, s);
    moe_modulate_hidden_backward(sb.moe_d_hidden_gated, sb.normed2, sb.moe_modulation,
                                  sb.d_normed, sb.moe_d_modulation, bsh, s);
    if (dbg) {
        debug_print_activation_stats("moe_bwd_d_normed_stage1", sb.d_normed, bsh, s);
        debug_print_f32_activation_stats("moe_bwd_d_modulation", sb.moe_d_modulation, bsh, s);
    }

    // modulation = sigmoid(pressure_mod(pressure)) backward.
    sigmoid_backward_f32(sb.moe_modulation, sb.moe_d_modulation, sb.moe_d_modulation, bsh, s);
    moe_small_proj_backward_input(sb.moe_d_modulation, ampere_compute_weight(lw.pressure_mod_w), sb.moe_d_pressure_contrib2,
                                   BS, num_routes, H, s);
    moe_small_proj_backward_weight(sb.moe_d_modulation, sb.moe_pressure, gl.g_pressure_mod,
                                    BS, num_routes, H, s);
    add_inplace_f32(sb.moe_d_pressure, sb.moe_d_pressure_contrib2,
                     BS * static_cast<std::size_t>(num_routes), s);
    if (dbg) {
        debug_print_f32_activation_stats("moe_bwd_d_pressure_combined", sb.moe_d_pressure,
                                          BS * static_cast<std::size_t>(num_routes), s);
    }

    // pressure = tanh(pressure_proj(pooled)) backward. sb.moe_d_pressure now
    // holds both paths' combined contribution.
    tanh_backward_f32(sb.moe_pressure, sb.moe_d_pressure, sb.moe_d_pressure,
                       BS * static_cast<std::size_t>(num_routes), s);
    moe_small_proj_backward_input(sb.moe_d_pressure, ampere_compute_weight(lw.pressure_proj_w), sb.moe_d_pooled, BS, H, num_routes, s);
    moe_small_proj_backward_weight(sb.moe_d_pressure, sb.moe_pooled, gl.g_pressure_proj, BS, H, num_routes, s);
    if (dbg) {
        debug_print_f32_activation_stats("moe_bwd_d_pooled", sb.moe_d_pooled, bsh, s);
    }

    // Reduce d(pooled) back into sb.d_normed (ADD -- combines with the
    // direct modulation-multiply contribution above, since pooled =
    // pool(normed2), completing the total gradient into this layer's
    // FFN-input point).
    moe_pool_by_sample_backward(sb.moe_d_pooled, sb.segs, sb.moe_count, sb.moe_start_grad_scratch,
                                 sb.d_normed, sb.B, sb.S, H, s);
    if (dbg) {
        debug_print_activation_stats("moe_bwd_d_normed_final", sb.d_normed, bsh, s);
    }
}

static void backward_accumulate(
    cublasHandle_t handle,
    AttentionBackendKind attention_backend,
    Fp8Ctx& f8,
    PackedFp4AttentionCtx& fp4_attn,
    const NativeRequest& request,
    const LatticeWeights& w,
    LatticeGrads& g,
    const uint32_t* d_tokens,
    const int32_t*  d_labels,
    StepBuffers& sb,
    NativeArena& arena,
    float pss_pred_aux_weight_resolved,
    bool run_output = true,
    bool run_embedding = true
) {
    IDA_GEMM_ROLE(::ida_native::gemm_trace::R_REGION_BWD_ACCUM);
    const int BS = sb.B * sb.S;
    const int H  = sb.H;
    const int I  = sb.I;
    const int V  = sb.V;
    const int nH = sb.nH;
    const int Hd = sb.Hd;
    const int KV = sb.kvH;
    const bool gqa = KV != H;
    cudaStream_t s = arena.stream;
    const std::size_t bsh = static_cast<std::size_t>(BS) * H;

    // ── Fused chunked LM head + cross-entropy ────────────────────────────────
    // Processes [ce_chunk, V] logits at a time: project → CE fwd+bwd in place →
    // dW_lm_head accumulate → d(normed) chunk.  Full [B*S, V] never exists.
    if (run_output) {
    IDA_CUDA_CHECK(cudaMemsetAsync(sb.n_valid, 0, sizeof(int), s));
    IDA_CUDA_CHECK(cudaMemsetAsync(sb.loss, 0, sizeof(float), s));
    cross_entropy_count_valid(d_labels, BS, sb.n_valid, s);
    const Fp8Slot* lmh8 = f8.on
        ? &f8.slots[static_cast<std::size_t>(f8.lm_head_weight_slot)] : nullptr;
    ActSlot* lmhead_act = f8.on ? &global_act_slot(f8, ACT_LMHEAD_IN) : nullptr;
    ActSlot* logits_grad_act = f8.on ? &global_act_slot(f8, ACT_LOGITS_GRAD) : nullptr;
    for (int r0 = 0; r0 < BS; r0 += sb.ce_chunk) {
        const int rows = std::min(sb.ce_chunk, BS - r0);
        const __nv_bfloat16* normed_c = sb.normed + static_cast<std::size_t>(r0) * H;
        // logits_chunk = normed_c @ lm_head^T
        if (f8.on) {
            fp8_quant_act(f8, *lmhead_act, normed_c, static_cast<std::size_t>(rows) * H, s);
            lt_gemm_fp8_nt(f8, rows, V, H, f8.act8, H, act_fp8_type(*lmhead_act), lmhead_act->descale_snapshot,
                           lmh8->fwd8, lmh8->descale, sb.logits_chunk, V, s);
        } else {
            gemm_bf16_nt(handle, rows, V, H, 1.f,
                         normed_c, H, ampere_compute_weight(w.lm_head), H,
                         0.f, sb.logits_chunk, V);
        }
        // CE fwd+bwd, gradient replaces logits in place (scaled by 1/n_valid)
        cross_entropy_fwd_bwd_chunk(sb.logits_chunk, d_labels + r0, sb.loss,
                                    sb.logits_chunk, rows, V, sb.n_valid, s);
        // dW_lm_head[V,H] += grad_chunk^T @ normed_c  (BF16 → FP32, always)
        gemm_bf16_tn_f32(handle, V, H, rows, 1.f,
                         sb.logits_chunk, V, normed_c, H, 1.f, g.g_lm_head, H);
        // d(normed) chunk = grad_chunk @ lm_head[V,H]
        if (f8.on) {
            fp8_quant_act(f8, *logits_grad_act, sb.logits_chunk, static_cast<std::size_t>(rows) * V, s);
            lt_gemm_fp8_nt(f8, rows, H, V, f8.act8, V, CUDA_R_8F_E5M2, logits_grad_act->descale_snapshot,
                           lmh8->bwd8, lmh8->descale,
                           sb.d_hidden + static_cast<std::size_t>(r0) * H, H, s);
        } else {
            gemm_bf16(handle, rows, H, V, 1.f,
                      sb.logits_chunk, V, ampere_compute_weight(w.lm_head), H, 0.f,
                      sb.d_hidden + static_cast<std::size_t>(r0) * H, H);
        }
    }

    // ── Final RMSNorm backward ────────────────────────────────────────────────
    const __nv_bfloat16* h_final = saved_slot(sb, w.num_layers);
    saved_bwd_fetch_begin(sb, w.num_layers, s);
    k_zeros_f32<<<ceildiv(H, 256), 256, 0, s>>>(sb.g_scale, H);
    if (is_gpt2_contract(request)) {
        k_zeros_f32<<<ceildiv(H, 256), 256, 0, s>>>(sb.norm_bias_scale, H);
        layernorm_backward(sb.d_hidden, h_final, w.final_norm, sb.rms_save,
                           sb.d_normed, sb.g_scale, sb.norm_bias_scale,
                           BS, H, model_norm_eps(request), s);
        k_acc_f32<<<ceildiv(H, 256), 256, 0, s>>>(g.g_fnorm_bias, sb.norm_bias_scale, H);
    } else {
        rmsnorm_backward(sb.d_hidden, h_final, w.final_norm, sb.rms_save,
                         sb.d_normed, sb.g_scale, BS, H, 1e-6f, s);
    }
    saved_bwd_fetch_end(sb, w.num_layers, s);
    k_acc_f32<<<ceildiv(H, 256), 256, 0, s>>>(g.g_fnorm, sb.g_scale, H);
    // Final norm is not a residual fork: d_hidden ← d_normed (replace, not add)
    IDA_CUDA_CHECK(cudaMemcpyAsync(sb.d_hidden, sb.d_normed,
        bsh * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, s));

    // LRSS backward: param grads from the saved forward tensors, plus the
    // pool-path gradient scattered back into d_hidden.  Must run before the
    // layer loop consumes d_hidden (delta was injected after the layers).
    if (sb.lrss_s) {
        lrss_backward(*sb.lrss_p, *sb.lrss_s, *sb.lrss_g, sb.d_hidden,
                      d_tokens, lrss_common_token_mask(V), sb.B, sb.S, H, s);
    }

    // TEMP diagnostic (2026-07-09): is d_hidden already large BEFORE any
    // transformer layer's backward runs — i.e. does the explosion
    // originate in the LM-head/CE-loss gradient itself, upstream of the
    // whole layer stack, rather than compounding through attention?
    if (gradnorm_debug_enabled()) {
        static int pre_layer_probes_left = 10;
        if (pre_layer_probes_left > 0) {
            --pre_layer_probes_left;
            float* pst = nullptr;
            cudaMallocAsync(&pst, 2 * sizeof(float), s);
            if (pst) {
                cudaMemsetAsync(pst, 0, 2 * sizeof(float), s);
                k_bf16_absmax_nonfinite<<<256, 256, 0, s>>>(sb.d_hidden, bsh, pst);
                float ph[2] = {0, 0};
                cudaStreamSynchronize(s);
                cudaMemcpy(ph, pst, sizeof(ph), cudaMemcpyDeviceToHost);
                std::fprintf(stderr,
                    "[pre-layer-dhidden] max|d_hidden|=%.6g nonfinite=%.0f "
                    "(before any layer's attention backward)\n",
                    ph[0], ph[1]);
            }
            cudaFreeAsync(pst, s);
        }
    }

    // ── Transformer layers (reverse, recompute activations per layer) ────────
    }

    for (int l = w.num_layers - 1; l >= 0; --l) {
        const auto& lw = w.layers[l];
        auto&       gl = g.layers[l];
        const __nv_bfloat16* h_in = saved_slot(sb, l);

        saved_bwd_fetch_begin(sb, l, s);
        // Moved ahead of the PSS block below (was originally computed after
        // it): PSS Stage 4 engagement routing needs d_ffn_residual, and this
        // computation only depends on sb.d_hidden (set once before the layer
        // loop starts) and sb.lss_feedback_skip_layer[l], both already
        // available at the top of every iteration -- reordering is safe.
        const bool feedback_scaled_layer =
            !sb.lss_feedback_skip_layer.empty() &&
            sb.lss_feedback_skip_layer[static_cast<std::size_t>(l)];
        const float feedback_scale = feedback_scaled_layer
            ? lss_feedback_residual_scale() : 1.0f;
        const __nv_bfloat16* d_ffn_residual = sb.d_hidden;
        if (feedback_scaled_layer && feedback_scale < 1.0f) {
            IDA_CUDA_CHECK(cudaMemcpyAsync(sb.d_hidden_feedback, sb.d_hidden,
                bsh * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, s));
            k_scale_bf16<<<ceildiv(bsh, 256), 256, 0, s>>>(
                sb.d_hidden_feedback, bsh, feedback_scale);
            d_ffn_residual = sb.d_hidden_feedback;
        }

        // PSS Stage 2/4 backward: aux gradient always (shadow mode never
        // stops scoring), plus -- only when Stage 4 has engaged this layer
        // (sb.pss_engaged_frac > 0, tail layer only) -- the engaged fraction
        // of the real trunk gradient, exactly mirroring the forward blend's
        // split. At engagement 0 this is bit-identical to pure shadow mode:
        // no engaged branch runs, d_ffn_residual is untouched, the real
        // down_proj backward below sees the full unscaled gradient.
        // Buffer-gated, not env-gated -- same worker-process rationale as
        // the forward hook.
        if (l == w.num_layers - 1 && sb.pss_pred_hidden) {
            const int R = w.pss_pred_rank;
            // d_pred_ffn = aux_weight * 2*(pred - detach(real)) / (BS*H)
            const bool normalize_pss_aux =
                pss_aux_normalize_enabled(request, H, lw.num_routes, lw.num_experts);
            sb.pss_aux_weight_used =
                pss_pred_aux_weight_resolved * sb.support_transition_scale;
            sb.pss_aux_normalize_used = normalize_pss_aux;
            k_pss_pred_daux<<<ceildiv(bsh, 256), 256, 0, s>>>(
                sb.pss_pred_ffn, sb.ffn_out, sb.d_pss_pred_ffn,
                pss_pred_aux_weight_resolved * sb.support_transition_scale, bsh,
                normalize_pss_aux ? sb.pss_pred_target_sq : nullptr);
            if (gradnorm_debug_enabled()) {
                debug_print_activation_stats("pss_aux_grad", sb.d_pss_pred_ffn, bsh, s);
            }
            const float e = sb.pss_engaged_frac;
            if (e > 0.0f) {
                // Route e·d_ffn_residual into the predictor's gradient, on
                // top of the aux term just computed above.
                k_pss_add_scaled_bf16<<<ceildiv(bsh, 256), 256, 0, s>>>(
                    sb.d_pss_pred_ffn, d_ffn_residual, bsh, e);
                // The real path only ever sees the complementary (1-e)
                // fraction -- same copy-then-scale pattern LSS feedback
                // already uses, composing multiplicatively with it if that
                // was also active this layer (rare: both require explicit
                // opt-in and their own independent confidence gates).
                if (d_ffn_residual != sb.d_hidden_feedback) {
                    IDA_CUDA_CHECK(cudaMemcpyAsync(sb.d_hidden_feedback, d_ffn_residual,
                        bsh * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, s));
                }
                k_scale_bf16<<<ceildiv(bsh, 256), 256, 0, s>>>(
                    sb.d_hidden_feedback, bsh, 1.0f - e);
                d_ffn_residual = sb.d_hidden_feedback;
            }
            // dW_up[R,H] += pred_hidden^T @ d_pred_ffn
            gemm_bf16_tn_f32(handle, R, H, BS, 1.f,
                              sb.pss_pred_hidden, R, sb.d_pss_pred_ffn, H,
                              1.f, g.g_pss_pred_up, H);
            // d_pred_hidden[BS,R] = d_pred_ffn @ pred_up^T
            const std::size_t bsr = static_cast<std::size_t>(BS) * R;
            gemm_bf16_nt(handle, BS, R, H, 1.f,
                         sb.d_pss_pred_ffn, H, w.pss_pred_up, H,
                         0.f, sb.d_pss_pred_hidden, R);
            k_relu_bwd_bf16<<<ceildiv(bsr, 256), 256, 0, s>>>(
                sb.pss_pred_hidden, sb.d_pss_pred_hidden, bsr);
            // dW_down[H,R] += pre_tail_hidden^T @ d_pred_hidden
            // Mirror the forward PSS hook's input choice exactly: in routed
            // mode, routed/expert bodies use the same moe_hidden_gated
            // surface as the forward PSS hook; dense bodies stay on h_in.
            // If these two disagree the predictor is differentiated w.r.t. an
            // input it was never evaluated on.
            const bool condition_pss =
                pss_conditioning_enabled(request, lw.num_routes, lw.num_experts) &&
                sb.moe_hidden_gated;
            const __nv_bfloat16* pss_pred_input =
                condition_pss ? sb.moe_hidden_gated : h_in;
            gemm_bf16_tn_f32(handle, H, R, BS, 1.f,
                              pss_pred_input, H, sb.d_pss_pred_hidden, R,
                              1.f, g.g_pss_pred_down, R);
            slot_grad_abs_clip(g.g_pss_pred_up,   static_cast<std::size_t>(R) * H, sb.norm_acc, s);
            slot_grad_abs_clip(g.g_pss_pred_down, static_cast<std::size_t>(H) * R, sb.norm_acc, s);
        }
        const Fp8Slot* fsl = f8.on ? &f8.slots[l * 7] : nullptr;
        ActSlot* dhidden_down_act = f8.on ? &layer_act_slot(f8, l, ACT_DHIDDEN_DOWN) : nullptr;
        ActSlot* gate_grad_act = f8.on ? &layer_act_slot(f8, l, ACT_GATE_GRAD) : nullptr;
        ActSlot* up_grad_act = f8.on ? &layer_act_slot(f8, l, ACT_UP_GRAD) : nullptr;
        ActSlot* dhidden_o_act = f8.on ? &layer_act_slot(f8, l, ACT_DHIDDEN_O) : nullptr;
        ActSlot* qkv_grad_act = f8.on ? &layer_act_slot(f8, l, ACT_QKV_GRAD) : nullptr;

        // Recompute this layer's activations from the saved input. is_recompute=true
        // so any FP8 delayed-scaling quant reuses forward's exact scale/descale
        // (see fp8_quant_act) instead of advancing it a second time.
        layer_forward_body(handle, attention_backend, f8, fp4_attn, request, l, lw, h_in, sb, s, /*is_recompute=*/true);

        // Match forward: clip the recomputed swiglu_out before any consumer
        // (snap quant or BF16 dW fallback) so backward sees forward's tensor.
        // Unconditional w.r.t. f8.on (that gate was the 2026-07-09 bug class),
        // but NOT w.r.t. routing: routed layers return from layer_forward_body
        // before the gate/up/SwiGLU stage, so swiglu_out is never written for
        // them and clipping it would read a stale buffer.
        if (lw.num_experts <= 0) {
            act_row_clip(request, sb.swiglu_out, BS, I, sb.rowdot, s);
        }

        // Snap down activation (swiglu_out from recompute, not inside layer_forward_body)
        if (lw.num_experts <= 0 && f8.on && f8.snap_down) {
            ActSlot& down_snap_act = layer_act_slot(f8, l, ACT_DOWN_IN);
            if (down_snap_act.fp8_max == kE4M3Max && fp8_snap_fuse_enabled()) {
                // Fused: act8 write is unused here (overwritten by dY quantize below),
                // but the dual kernel is correct and avoids the act8 read-back in transpose.
                fp8_quant_and_transpose_e4m3(sb.swiglu_out, f8.act8, f8.snap_down,
                                             down_snap_act.scale_snapshot, BS, I, s);
            } else {
                fp8_quant_act(f8, down_snap_act, sb.swiglu_out, static_cast<std::size_t>(BS) * I, s, /*is_recompute=*/true);
                if (down_snap_act.fp8_max == kE4M3Max)
                    transpose_u8(f8.act8, f8.snap_down, BS, I, s);  // → [I,BS] for TN dW
            }
        }

        // ─ FFN backward ──────────────────────────────────────────────────────
        // d_swiglu = d_hidden @ down^T;  dW_down[I,H] += swiglu_snap^T @ d_hidden
        if (lw.num_experts > 0 && lw.moe_kind == 1) {
            moe_generic_backward(l, lw, gl, d_ffn_residual, sb, s);
        } else if (lw.num_experts > 0) {
            // Sparse expert bank replaces the dense down/gate/up/SwiGLU
            // backward entirely for this layer, mirroring forward's own
            // branch. Produces sb.d_normed directly (fully combined, both
            // the modulation-multiply and pooled-path contributions) --
            // no dense weight gradients to clip, nothing else to combine.
            moe_backward(handle, f8, l, lw, gl, d_ffn_residual, sb, s);
        } else if (is_gpt2_contract(request)) {
            gemm_bf16_tn_f32(handle, I, H, BS, 1.f,
                             sb.swiglu_out, I, d_ffn_residual, H, 1.f, gl.g_down, H);
            k_zeros_f32<<<ceildiv(H, 256), 256, 0, s>>>(sb.g_scale, H);
            col_sum_strided_bf16(d_ffn_residual, sb.g_scale, BS, H, H, 0, s);
            k_acc_f32<<<ceildiv(H, 256), 256, 0, s>>>(gl.g_ffn_out_bias, sb.g_scale, H);
            gemm_bf16_nt(handle, BS, I, H, 1.f,
                         d_ffn_residual, H, ampere_compute_weight(lw.down_proj), H,
                         0.f, sb.d_ffn_i, I);
            gelu_new_backward(sb.d_ffn_i, sb.gate_out, sb.gate_out,
                              static_cast<std::size_t>(BS) * I, s);
            act_row_clip(request, sb.gate_out, BS, I, sb.rowdot, s);
            gemm_bf16_tn_f32(handle, H, I, BS, 1.f,
                             sb.normed2, H, sb.gate_out, I, 1.f, gl.g_gate, I);
            k_zeros_f32<<<ceildiv(I, 256), 256, 0, s>>>(sb.g_scale, I);
            col_sum_strided_bf16(sb.gate_out, sb.g_scale, BS, I, I, 0, s);
            k_acc_f32<<<ceildiv(I, 256), 256, 0, s>>>(gl.g_ffn_in_bias, sb.g_scale, I);
            gemm_bf16_nt(handle, BS, H, I, 1.f,
                         sb.gate_out, I, ampere_compute_weight(lw.gate_proj), I,
                         0.f, sb.d_normed, H);
            slot_grad_abs_clip(gl.g_down, static_cast<std::size_t>(I) * H, sb.norm_acc, s);
            slot_grad_abs_clip(gl.g_gate, static_cast<std::size_t>(H) * I, sb.norm_acc, s);
        } else if (f8.on) {
            fp8_quant_act(f8, *dhidden_down_act, d_ffn_residual, bsh, s);
            ActSlot& down_fwd_act = layer_act_slot(f8, l, ACT_DOWN_IN);
            if (f8.snap_down && down_fwd_act.fp8_max == kE4M3Max) {
                transpose_u8(f8.act8, f8.grad8_t, BS, H, s);  // dY^T [H,BS]
                lt_gemm_fp8_tn_accum_f32(f8, I, H, BS,
                    f8.snap_down, down_fwd_act.descale_snapshot,
                    f8.grad8_t, dhidden_down_act->descale_snapshot,
                    gl.g_down, H, s);
            } else {
                gemm_bf16_tn_f32(handle, I, H, BS, 1.f,
                                 sb.swiglu_out, I, d_ffn_residual, H, 1.f, gl.g_down, H);
            }
            lt_gemm_fp8_nt(f8, BS, I, H, f8.act8, H, CUDA_R_8F_E5M2, dhidden_down_act->descale_snapshot,
                           fsl[F8_DOWN].bwd8, fsl[F8_DOWN].descale, sb.d_ffn_i, I, s);
            slot_grad_abs_clip(gl.g_down, static_cast<std::size_t>(I) * H, sb.norm_acc, s);
            swiglu_backward(sb.d_ffn_i, sb.gate_out, sb.up_out,
                            sb.gate_out, sb.up_out, BS * I, s);
            act_row_clip(request, sb.gate_out, BS, I, sb.rowdot, s);
            act_row_clip(request, sb.up_out,   BS, I, sb.rowdot, s);
            fp8_quant_act(f8, *gate_grad_act, sb.gate_out, static_cast<std::size_t>(BS) * I, s);
            ActSlot& ffn_fwd_act = layer_act_slot(f8, l, ACT_FFN_IN);
            if (f8.snap_ffn && ffn_fwd_act.fp8_max == kE4M3Max) {
                transpose_u8(f8.act8, f8.grad8_t, BS, I, s);  // dY^T [I,BS]
                lt_gemm_fp8_tn_accum_f32(f8, H, I, BS,
                    f8.snap_ffn, ffn_fwd_act.descale_snapshot,
                    f8.grad8_t, gate_grad_act->descale_snapshot,
                    gl.g_gate, I, s);
            } else {
                gemm_bf16_tn_f32(handle, H, I, BS, 1.f,
                                 sb.normed2, H, sb.gate_out, I, 1.f, gl.g_gate, I);
            }
            lt_gemm_fp8_nt(f8, BS, H, I, f8.act8, I, CUDA_R_8F_E5M2, gate_grad_act->descale_snapshot,
                           fsl[F8_GATE].bwd8, fsl[F8_GATE].descale, sb.d_normed, H, s);
            fp8_quant_act(f8, *up_grad_act, sb.up_out, static_cast<std::size_t>(BS) * I, s);
            if (f8.snap_ffn && ffn_fwd_act.fp8_max == kE4M3Max) {
                transpose_u8(f8.act8, f8.grad8_t, BS, I, s);  // dY^T [I,BS]
                lt_gemm_fp8_tn_accum_f32(f8, H, I, BS,
                    f8.snap_ffn, ffn_fwd_act.descale_snapshot,
                    f8.grad8_t, up_grad_act->descale_snapshot,
                    gl.g_up, I, s);
            } else {
                gemm_bf16_tn_f32(handle, H, I, BS, 1.f,
                                 sb.normed2, H, sb.up_out, I, 1.f, gl.g_up, I);
            }
            lt_gemm_fp8_nt(f8, BS, H, I, f8.act8, I, CUDA_R_8F_E5M2, up_grad_act->descale_snapshot,
                           fsl[F8_UP].bwd8, fsl[F8_UP].descale, sb.ffn_out, H, s);
            slot_grad_abs_clip(gl.g_gate, static_cast<std::size_t>(H) * I, sb.norm_acc, s);
            slot_grad_abs_clip(gl.g_up,   static_cast<std::size_t>(H) * I, sb.norm_acc, s);
            k_inplace_add_bf16<<<ceildiv(bsh, 256), 256, 0, s>>>(sb.d_normed, sb.ffn_out, bsh);
        } else {
            gemm_bf16_tn_f32(handle, I, H, BS, 1.f,
                             sb.swiglu_out, I, d_ffn_residual, H, 1.f, gl.g_down, H);
            gemm_bf16_nt(handle, BS, I, H, 1.f,
                         d_ffn_residual, H, ampere_compute_weight(lw.down_proj), H,
                         0.f, sb.d_ffn_i, I);
            // Per-slot absolute clip on the accumulated weight gradient itself —
            // see slot_grad_abs_clip's comment for why row-level clipping alone
            // cannot bound the aggregate sum over all BS rows.
            slot_grad_abs_clip(gl.g_down, static_cast<std::size_t>(I) * H, sb.norm_acc, s);
            // SwiGLU backward (elementwise; overwrites gate_out/up_out with d_gate/d_up)
            swiglu_backward(sb.d_ffn_i, sb.gate_out, sb.up_out,
                            sb.gate_out, sb.up_out, BS * I, s);
            // gate_out/up_out now hold d_gate/d_up — the FFN backward's own
            // computed gradient, analogous to dQ/dK/dV.  Same class of
            // unprotected operand as dqkv_row_clip fixes for attention; these
            // feed gl.g_gate/gl.g_up's gradient-signal side directly below.
            act_row_clip(request, sb.gate_out, BS, I, sb.rowdot, s);
            act_row_clip(request, sb.up_out,   BS, I, sb.rowdot, s);
            // dW_gate/up[H,I] += normed2_snap^T @ d_gate/d_up;  d(normed2) from both
            gemm_bf16_tn_f32(handle, H, I, BS, 1.f,
                             sb.normed2, H, sb.gate_out, I, 1.f, gl.g_gate, I);
            gemm_bf16_tn_f32(handle, H, I, BS, 1.f,
                             sb.normed2, H, sb.up_out,   I, 1.f, gl.g_up,   I);
            gemm_bf16_nt(handle, BS, H, I, 1.f,
                         sb.gate_out, I, ampere_compute_weight(lw.gate_proj), I,
                         0.f, sb.d_normed, H);
            gemm_bf16_nt(handle, BS, H, I, 1.f,
                         sb.up_out,   I, ampere_compute_weight(lw.up_proj), I,
                         0.f, sb.ffn_out, H);
            slot_grad_abs_clip(gl.g_gate, static_cast<std::size_t>(H) * I, sb.norm_acc, s);
            slot_grad_abs_clip(gl.g_up,   static_cast<std::size_t>(H) * I, sb.norm_acc, s);
            k_inplace_add_bf16<<<ceildiv(bsh, 256), 256, 0, s>>>(sb.d_normed, sb.ffn_out, bsh);
        }
        // d_normed is ffn_norm's OTHER operand (sb.hidden, already clipped
        // above, is the first) — the gap identified 2026-07-09: clipping
        // sb.hidden alone moved the dominant slot L1.ffn_norm->L0.ffn_norm
        // without reducing magnitude, because this gradient-signal operand
        // (accumulated from gate/up_proj's own backward) was still open.
        act_row_clip(request, sb.d_normed, BS, H, sb.rowdot, s);
        // FFN norm backward: x = post-attention residual (sb.hidden), rms = rms_save
        k_zeros_f32<<<ceildiv(H, 256), 256, 0, s>>>(sb.g_scale, H);
        if (is_gpt2_contract(request)) {
            k_zeros_f32<<<ceildiv(H, 256), 256, 0, s>>>(sb.norm_bias_scale, H);
            layernorm_backward(sb.d_normed, sb.hidden, lw.ffn_norm, sb.rms_save,
                               sb.ffn_out, sb.g_scale, sb.norm_bias_scale,
                               BS, H, model_norm_eps(request), s);
            k_acc_bf16<<<ceildiv(H, 256), 256, 0, s>>>(gl.g_fnorm, sb.g_scale, H);
            k_acc_f32<<<ceildiv(H, 256), 256, 0, s>>>(gl.g_ffn_norm_bias, sb.norm_bias_scale, H);
        } else {
            rmsnorm_backward(sb.d_normed, sb.hidden, lw.ffn_norm, sb.rms_save,
                             sb.ffn_out, sb.g_scale, BS, H, 1e-6f, s);
            k_acc_bf16<<<ceildiv(H, 256), 256, 0, s>>>(gl.g_fnorm, sb.g_scale, H);
        }
        slot_grad_abs_clip(gl.g_fnorm, static_cast<std::size_t>(H), sb.norm_acc, s);
        // Residual fork: d_hidden += d(pre-ffn-norm hidden)
        k_inplace_add_bf16<<<ceildiv(bsh, 256), 256, 0, s>>>(sb.d_hidden, sb.ffn_out, bsh);

        // TEMP diagnostic (2026-07-09): d_hidden right here is the input to
        // this layer's own attention O-proj backward — bisects whether THIS
        // layer's FFN backward already amplified a tiny (2.7e-11) pre-layer
        // d_hidden, or whether it stays tiny into attention (meaning the
        // ~1e17x amplification is entirely inside the attention backward's
        // own math, not inherited from FFN).
        if (gradnorm_debug_enabled() && l == w.num_layers - 1) {
            static int ffn_out_probes_left = 4;
            if (ffn_out_probes_left > 0) {
                --ffn_out_probes_left;
                float* pst = nullptr;
                cudaMallocAsync(&pst, 2 * sizeof(float), s);
                if (pst) {
                    cudaMemsetAsync(pst, 0, 2 * sizeof(float), s);
                    k_bf16_absmax_nonfinite<<<256, 256, 0, s>>>(sb.d_hidden, bsh, pst);
                    float ph[2] = {0, 0};
                    cudaStreamSynchronize(s);
                    cudaMemcpy(ph, pst, sizeof(ph), cudaMemcpyDeviceToHost);
                    std::fprintf(stderr,
                        "[post-ffn-dhidden] layer=%d max|d_hidden|=%.6g nonfinite=%.0f "
                        "(after this layer's FFN backward, before its attention backward)\n",
                        l, ph[0], ph[1]);
                }
                cudaFreeAsync(pst, s);
            }
        }

        // Same absolute-ceiling containment as dqkv_row_clip, applied here
        // too: d_hidden right after THIS layer's own FFN residual fork is
        // consumed immediately below by o_proj's gradient GEMM
        // (gl.g_o) — a separate, EARLIER exposure than the inter-layer
        // boundary interlayer_clip already covers.  Found 2026-07-09:
        // fixing dQ/dK/dV (dqkv_row_clip) and the inter-layer boundary
        // (interlayer_clip) left L0.o_proj as the next dominant slot,
        // UNCHANGED in magnitude — proving this layer's own FFN backward
        // was reintroducing largeness into d_hidden after the boundary
        // clip but before o_proj's GEMM, a gap neither prior fix covered.
        interlayer_clip(sb.d_hidden, BS, H, sb.rowdot, s);

        // ─ Attention backward (flash-style recompute, no [S,S] buffers) ──────
        // d(attn_out) = d_hidden @ o_proj^T;  dW_o[H,H] += attn_snap^T @ d_hidden
        if (f8.on) {
            fp8_quant_act(f8, *dhidden_o_act, sb.d_hidden, bsh, s);
            ActSlot& o_fwd_act = layer_act_slot(f8, l, ACT_O_IN);
            if (f8.snap_o && o_fwd_act.fp8_max == kE4M3Max) {
                transpose_u8(f8.act8, f8.grad8_t, BS, H, s);  // dY^T [H,BS]
                lt_gemm_fp8_tn_accum_f32(f8, H, H, BS,
                    f8.snap_o, o_fwd_act.descale_snapshot,
                    f8.grad8_t, dhidden_o_act->descale_snapshot,
                    gl.g_o, H, s);
            } else {
                gemm_bf16_tn_f32(handle, H, H, BS, 1.f,
                                 sb.attn_out, H, sb.d_hidden, H, 1.f, gl.g_o, H);
            }
            lt_gemm_fp8_nt(f8, BS, H, H, f8.act8, H, CUDA_R_8F_E5M2, dhidden_o_act->descale_snapshot,
                           fsl[F8_O].bwd8, fsl[F8_O].descale, sb.o_out, H, s);
        } else {
            gemm_bf16_tn_f32(handle, H, H, BS, 1.f,
                             sb.attn_out, H, sb.d_hidden, H, 1.f, gl.g_o, H);
            gemm_bf16_nt(handle, BS, H, H, 1.f,
                         sb.d_hidden, H, ampere_compute_weight(lw.o_proj), H,
                         0.f, sb.o_out, H);
        }
        if (lw.o_bias) {
            k_zeros_f32<<<ceildiv(H, 256), 256, 0, s>>>(sb.g_scale, H);
            col_sum_strided_bf16(sb.d_hidden, sb.g_scale, BS, H, H, 0, s);
            k_acc_f32<<<ceildiv(H, 256), 256, 0, s>>>(gl.g_o_bias, sb.g_scale, H);
        }
        slot_grad_abs_clip(gl.g_o, static_cast<std::size_t>(H) * H, sb.norm_acc, s);
        // Reshape dY to heads
        {
            dim3 grid(BS, (H + 127) / 128);
            k_reshape_to_heads_bf16<<<grid, 128, 0, s>>>(
                sb.o_out, sb.do_f32, sb.B, sb.S, nH, Hd, H);
        }

        PackedFp4AttentionOperands packed_fp4_local{};
        const PackedFp4AttentionOperands* packed_fp4 = nullptr;
        if (fp4_attn.on) {
            packed_fp4_local = packed_fp4_operands(fp4_attn, l);
            packed_fp4 = &packed_fp4_local;
        }
        // dQ / dK / dV via LSE recompute (q/k/v/o/lse fresh from layer recompute)
        const uint16_t* attn_segs = sb.segs;
        attention_backward(attention_backend,
                           sb.q_f32, sb.k_f32, sb.v_f32, sb.o_f32, sb.do_f32,
                           sb.lse, sb.dq_f32, sb.dk_f32, sb.dv_f32, sb.rowdot,
                           packed_fp4,
                           attn_segs, nH,
                           sb.B * nH, sb.S, Hd,
                           1.0f / sqrtf(static_cast<float>(Hd)), s,
                           KV / Hd);
        // Un-rotate dQ/dK before they reach the dW_q/dW_k GEMMs below, which
        // expect the gradient w.r.t. q_proj/k_proj's raw (pre-RoPE) output,
        // not the rotated-space gradient attention_backward just produced.
        // angle_sign=-1 is the rotation's own transpose (see rope_apply's
        // comment) -- no-op for every existing IDA Lattice body.
        rope_apply(sb.dq_f32, sb.segs, sb.B, nH, sb.S, Hd, request.model.rope_theta, -1.0f, s);
        rope_apply(sb.dk_f32, sb.segs, sb.B, nH, sb.S, Hd, request.model.rope_theta, -1.0f, s);
        // Reshape dQ/dK/dV back to the raw projection slots. GQA's expanded
        // dK/dV must be summed across each Q-head group first.
        {
            dim3 grid(BS, (H + 127) / 128);
            k_reshape_from_heads_bf16<<<grid, 128, 0, s>>>(
                sb.dq_f32, sb.qkv,         sb.B, sb.S, nH, Hd, 3 * H);
            if (!gqa) {
                k_reshape_from_heads_bf16<<<grid, 128, 0, s>>>(
                    sb.dk_f32, sb.qkv + H, sb.B, sb.S, nH, Hd, 3 * H);
                k_reshape_from_heads_bf16<<<grid, 128, 0, s>>>(
                    sb.dv_f32, sb.qkv + 2 * H, sb.B, sb.S, nH, Hd, 3 * H);
            } else {
                const int kv_heads = KV / Hd;
                dim3 kv_grid(BS, (KV + 127) / 128);
                k_reduce_heads_to_kv_bf16<<<kv_grid, 128, 0, s>>>(
                    sb.dk_f32, sb.qkv + H, sb.B, sb.S, nH, kv_heads, Hd, 3 * H);
                k_reduce_heads_to_kv_bf16<<<kv_grid, 128, 0, s>>>(
                    sb.dv_f32, sb.qkv + 2 * H, sb.B, sb.S, nH, kv_heads, Hd, 3 * H);
            }
        }
        // Contain the attention backward's own explosion before it reaches
        // the weight-gradient GEMMs — see dqkv_row_clip's comment.
        dqkv_row_clip(sb.qkv, BS, H, sb.rowdot, s, gqa ? KV : H);
        // dW_q[H,H] and dW_k/v[H,KV] += normed_snap^T @ dQ/dK/dV;
        // d(normed) = dQ@q^T + dK@k^T + dV@v^T.
        if (f8.on && !gqa) {
            fp8_quant_act(f8, *qkv_grad_act, sb.qkv, static_cast<std::size_t>(BS) * 3 * H, s);
            const char* q8 = static_cast<const char*>(f8.act8);
            ActSlot& qkv_fwd_act = layer_act_slot(f8, l, ACT_QKV_IN);
            if (f8.snap_qkv && qkv_fwd_act.fp8_max == kE4M3Max) {
                // Transposed [3H,BS]: q/k/v slices become contiguous [H,BS] blocks.
                transpose_u8(f8.act8, f8.grad8_t, BS, 3 * H, s);
                const char* q8t = static_cast<const char*>(f8.grad8_t);
                const std::size_t qkv_t_stride = static_cast<std::size_t>(H) * BS;
                lt_gemm_fp8_tn_accum_f32(f8, H, H, BS,
                    f8.snap_qkv, qkv_fwd_act.descale_snapshot,
                    q8t,                    qkv_grad_act->descale_snapshot, gl.g_q, H, s);
                lt_gemm_fp8_tn_accum_f32(f8, H, H, BS,
                    f8.snap_qkv, qkv_fwd_act.descale_snapshot,
                    q8t + qkv_t_stride,     qkv_grad_act->descale_snapshot, gl.g_k, H, s);
                lt_gemm_fp8_tn_accum_f32(f8, H, H, BS,
                    f8.snap_qkv, qkv_fwd_act.descale_snapshot,
                    q8t + 2 * qkv_t_stride, qkv_grad_act->descale_snapshot, gl.g_v, H, s);
            } else {
                gemm_bf16_tn_f32(handle, H, H, BS, 1.f,
                                 sb.normed, H, sb.qkv,         3 * H, 1.f, gl.g_q, H);
                gemm_bf16_tn_f32(handle, H, H, BS, 1.f,
                                 sb.normed, H, sb.qkv + H,     3 * H, 1.f, gl.g_k, H);
                gemm_bf16_tn_f32(handle, H, H, BS, 1.f,
                                 sb.normed, H, sb.qkv + 2 * H, 3 * H, 1.f, gl.g_v, H);
            }
            lt_gemm_fp8_nt(f8, BS, H, H, q8,         3 * H, CUDA_R_8F_E5M2, qkv_grad_act->descale_snapshot,
                           fsl[F8_Q].bwd8, fsl[F8_Q].descale, sb.d_normed, H, s);
            lt_gemm_fp8_nt(f8, BS, H, H, q8 + H,     3 * H, CUDA_R_8F_E5M2, qkv_grad_act->descale_snapshot,
                           fsl[F8_K].bwd8, fsl[F8_K].descale, sb.ffn_out, H, s);
            k_inplace_add_bf16<<<ceildiv(bsh, 256), 256, 0, s>>>(sb.d_normed, sb.ffn_out, bsh);
            lt_gemm_fp8_nt(f8, BS, H, H, q8 + 2 * H, 3 * H, CUDA_R_8F_E5M2, qkv_grad_act->descale_snapshot,
                           fsl[F8_V].bwd8, fsl[F8_V].descale, sb.ffn_out, H, s);
            k_inplace_add_bf16<<<ceildiv(bsh, 256), 256, 0, s>>>(sb.d_normed, sb.ffn_out, bsh);
        } else {
            gemm_bf16_tn_f32(handle, H, H, BS, 1.f,
                             sb.normed, H, sb.qkv,         3 * H, 1.f, gl.g_q, H);
            gemm_bf16_tn_f32(handle, H, KV, BS, 1.f,
                             sb.normed, H, sb.qkv + H,     3 * H, 1.f, gl.g_k, KV);
            gemm_bf16_tn_f32(handle, H, KV, BS, 1.f,
                             sb.normed, H, sb.qkv + 2 * H, 3 * H, 1.f, gl.g_v, KV);
            gemm_bf16_nt(handle, BS, H, H, 1.f,
                         sb.qkv,         3 * H, ampere_compute_weight(lw.q_proj), H,
                         0.f, sb.d_normed, H);
            gemm_bf16_nt(handle, BS, H, KV, 1.f,
                         sb.qkv + H,     3 * H, ampere_compute_weight(lw.k_proj), KV,
                         1.f, sb.d_normed, H);
            gemm_bf16_nt(handle, BS, H, KV, 1.f,
                         sb.qkv + 2 * H, 3 * H, ampere_compute_weight(lw.v_proj), KV,
                         1.f, sb.d_normed, H);
        }
        // Qwen2-family QKV bias gradient (2026-08-23): bias-add doesn't
        // change the gradient flowing to Q/K/V pre-activation, so this is a
        // plain column-sum of the SAME d(qkv) sb.qkv already holds above
        // (valid in both the fp8 and bf16 branches) -- no recompute needed.
        // Same zero-scratch -> accumulate-into-persistent-bf16-slot pattern
        // rmsnorm_backward's g_scale/anorm/fnorm handling uses below.
        if (lw.q_bias != nullptr) {
            k_zeros_f32<<<ceildiv(static_cast<std::size_t>(H), 256), 256, 0, s>>>(sb.g_scale, H);
            col_sum_strided_bf16(sb.qkv, sb.g_scale, BS, H, 3 * H, 0, s);
            k_acc_f32<<<ceildiv(static_cast<std::size_t>(H), 256), 256, 0, s>>>(gl.g_q_bias, sb.g_scale, H);

            k_zeros_f32<<<ceildiv(static_cast<std::size_t>(KV), 256), 256, 0, s>>>(sb.g_scale, KV);
            col_sum_strided_bf16(sb.qkv, sb.g_scale, BS, KV, 3 * H, H, s);
            k_acc_f32<<<ceildiv(static_cast<std::size_t>(KV), 256), 256, 0, s>>>(gl.g_k_bias, sb.g_scale, KV);

            k_zeros_f32<<<ceildiv(static_cast<std::size_t>(KV), 256), 256, 0, s>>>(sb.g_scale, KV);
            col_sum_strided_bf16(sb.qkv, sb.g_scale, BS, KV, 3 * H, 2 * H, s);
            k_acc_f32<<<ceildiv(static_cast<std::size_t>(KV), 256), 256, 0, s>>>(gl.g_v_bias, sb.g_scale, KV);
        }
        slot_grad_abs_clip(gl.g_q, static_cast<std::size_t>(H) * H, sb.norm_acc, s);
        slot_grad_abs_clip(gl.g_k, static_cast<std::size_t>(H) * KV, sb.norm_acc, s);
        slot_grad_abs_clip(gl.g_v, static_cast<std::size_t>(H) * KV, sb.norm_acc, s);
        // L0-anomaly bisection probe: the layer-bwd-dhidden series showed a
        // single-layer x10,000 amplification at L0 only (mid-stack additive
        // ~+2/layer, healthy). Two candidate factors multiply INSIDE this
        // block: (a) d_normed's own magnitude out of the attention backward
        // (massive L0 scores), and (b) rmsnorm_backward's 1/rms where L0's
        // input is the raw embedding table (init +-0.02 -> rms ~0.0115 ->
        // x87 amplifier no other layer has). Printing both operands right
        // at the seam splits the product into its factors.
        if (gradnorm_debug_enabled()) {
            float* pst = nullptr;
            cudaMallocAsync(&pst, 2 * sizeof(float), s);
            cudaMemsetAsync(pst, 0, 2 * sizeof(float), s);
            k_bf16_absmax_nonfinite<<<256, 256, 0, s>>>(sb.d_normed, bsh, pst);
            float ph[2] = {0, 0};
            float rms_sample[16] = {0};
            cudaStreamSynchronize(s);
            cudaMemcpy(ph, pst, sizeof(ph), cudaMemcpyDeviceToHost);
            cudaMemcpy(rms_sample, sb.rms_a, sizeof(rms_sample), cudaMemcpyDeviceToHost);
            cudaFreeAsync(pst, s);
            float rmin = rms_sample[0], rsum = 0.0f;
            for (float r : rms_sample) { rmin = fminf(rmin, r); rsum += r; }
            std::fprintf(stderr,
                "[l0-bisect] micro_step=%d layer=%d max|d_normed|=%.6g "
                "rms_a[min of 16]=%.6g rms_a[mean of 16]=%.6g inv_rms=%.4g\n",
                g_debug_micro_step, l, ph[0], rmin, rsum / 16.0f,
                (rmin > 0.0f) ? 1.0f / rmin : 0.0f);
        }
        // Attention norm backward: x = layer input, rms = rms_a
        k_zeros_f32<<<ceildiv(H, 256), 256, 0, s>>>(sb.g_scale, H);
        if (is_gpt2_contract(request)) {
            k_zeros_f32<<<ceildiv(H, 256), 256, 0, s>>>(sb.norm_bias_scale, H);
            layernorm_backward(sb.d_normed, h_in, lw.attn_norm, sb.rms_a,
                               sb.attn_out, sb.g_scale, sb.norm_bias_scale,
                               BS, H, model_norm_eps(request), s);
            k_acc_f32<<<ceildiv(H, 256), 256, 0, s>>>(gl.g_attn_norm_bias, sb.norm_bias_scale, H);
        } else {
            rmsnorm_backward(sb.d_normed, h_in, lw.attn_norm, sb.rms_a,
                             sb.attn_out, sb.g_scale, BS, H, 1e-6f, s);
        }
        // Last read of h_in this layer: release the ring slot and start the
        // H2D prefetch for its next backward tenant (layer l - ring).
        saved_bwd_fetch_end(sb, l, s);
        k_acc_bf16<<<ceildiv(H, 256), 256, 0, s>>>(gl.g_anorm, sb.g_scale, H);
        slot_grad_abs_clip(gl.g_anorm, static_cast<std::size_t>(H), sb.norm_acc, s);
        // Residual fork
        k_inplace_add_bf16<<<ceildiv(bsh, 256), 256, 0, s>>>(sb.d_hidden, sb.attn_out, bsh);
        // Per-layer growth probe: absmax of d_hidden at every layer boundary
        // (BEFORE interlayer_clip masks the raw magnitude), so the sequence
        // of prints across the backward sweep IS the per-layer growth-factor
        // series — the direct measurement of whether residual-scaled init
        // (IDA_NATIVE_RESIDUAL_SCALED_INIT) actually flattens the exponent,
        // vs. the clips merely containing it.
        if (gradnorm_debug_enabled()) {
            float* pst = nullptr;
            cudaMallocAsync(&pst, 2 * sizeof(float), s);
            cudaMemsetAsync(pst, 0, 2 * sizeof(float), s);
            k_bf16_absmax_nonfinite<<<256, 256, 0, s>>>(sb.d_hidden, bsh, pst);
            float ph[2] = {0, 0};
            cudaStreamSynchronize(s);
            cudaMemcpy(ph, pst, sizeof(ph), cudaMemcpyDeviceToHost);
            cudaFreeAsync(pst, s);
            std::fprintf(stderr,
                "[layer-bwd-dhidden] micro_step=%d layer=%d max|d_hidden|=%.6g "
                "(pre-interlayer-clip, layer boundary)\n",
                g_debug_micro_step, l, ph[0]);
        }
        // Inter-layer gradient clip: sb.d_hidden IS this layer's finished
        // input gradient — about to become the next (earlier) layer's dO.
        // This layer's OWN weight grads (g_q/g_k/g_v/g_anorm/...) are
        // already accumulated above, before this point — clipping here
        // only bounds what CARRIES FORWARD, not what this layer recorded.
        // Root cause (2026-07-09, dS-cancellation probe): dp/D in the
        // attention backward's dS = P*(dp-D) stay in close relative
        // agreement (~1%) at every layer, but grow ~10-36x layer over
        // layer through the backward sweep (L3->L0 in a 4-layer AI repro
        // reached 1e6 -> 6e12) — genuine gradient-magnitude compounding
        // through depth, not a numerical-precision bug in any one kernel.
        // L0 is always the dominant blown-up slot because it's the LAST
        // layer the backward sweep touches, inheriting maximum
        // accumulated amplification.  Reuses act_row_clip verbatim (same
        // data-adaptive per-row L2-norm ceiling already proven on
        // attn_out/swiglu_out) at the one point that can actually break
        // the compounding: the inter-layer boundary itself.
        interlayer_clip(sb.d_hidden, BS, H, sb.rowdot, s);
    }

    // ── Embedding backward (atomicAdd accumulates into persistent grads) ─────
    if (run_embedding) {
        if (is_gpt2_contract(request)) {
            position_embedding_backward(sb.d_hidden, sb.segs, g.g_position_embeddings,
                                        sb.B, sb.S, H, w.max_position_embeddings, s);
        }
        embedding_backward(sb.d_hidden, d_tokens, g.g_embed, sb.B, sb.S, H, s);
        if (const float clip = embed_row_clip_threshold(); clip > 0.0f) {
            // Clip right after THIS micro-step's own scatter-add, before its
            // contribution can compound across the grad-accum window (see
            // embedding.cu's k_embed_row_clip for why this exists).
            IDA_CUDA_CHECK(cudaMemsetAsync(sb.embed_clip_stats, 0, 2 * sizeof(float), s));
            const float common_clip = lrss_common_clip_threshold();
            const uint8_t* common_mask = (common_clip > 0.0f) ? lrss_common_token_mask(V) : nullptr;
            if (common_mask) {
                embed_row_clip_masked_with_stats(
                    g.g_embed, V, H, clip, common_clip, common_mask, sb.embed_clip_stats, s);
            } else {
                embed_row_clip_with_stats(g.g_embed, V, H, clip, sb.embed_clip_stats, s);
            }
        }
    }
}

// ─── MLPerf Open Division logging (MLLOG line format) ────────────────────────
// One mlperf_log.txt per burn in the student's output_dir, alongside (not
// replacing) the existing status/JSONL/summary telemetry.
// Format: ':::MLLOG {"namespace", "time_ms", "event_type", "key", "value",
// "metadata"}' — POINT_IN_TIME / INTERVAL_START / INTERVAL_END.

struct MlLog {
    std::ofstream f;

    static long long now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    static std::string q(const std::string& s) { return "\"" + s + "\""; }

    void open(const std::filesystem::path& p) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
        f.open(p, std::ios::trunc);
    }
    void log(const char* etype, const std::string& key, const std::string& value,
             const std::string& meta = "{}") {
        if (!f.is_open()) return;
        f << ":::MLLOG {\"namespace\": \"ida_native\", \"time_ms\": " << now_ms()
          << ", \"event_type\": \"" << etype << "\", \"key\": \"" << key
          << "\", \"value\": " << value << ", \"metadata\": " << meta << "}\n";
        f.flush();
    }
    void point(const std::string& k, const std::string& v, const std::string& m = "{}") {
        log("POINT_IN_TIME", k, v, m);
    }
    void begin(const std::string& k, const std::string& m = "{}") {
        log("INTERVAL_START", k, "null", m);
    }
    void end(const std::string& k, const std::string& m = "{}") {
        log("INTERVAL_END", k, "null", m);
    }
};

// ─── LR schedule: linear warmup → cosine decay to 10% ────────────────────────

static double lr_at(int opt_step, int total_opt_steps, double base_lr) {
    const int warmup = std::max(1, total_opt_steps / 50);
    if (opt_step <= warmup)
        return base_lr * static_cast<double>(opt_step) / warmup;
    const double p = static_cast<double>(opt_step - warmup) /
                     std::max(1, total_opt_steps - warmup);
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + cos(p * 3.14159265358979323846)));
}

static void validate_attention_request(
    const NativeRequest& request,
    AttentionBackendKind attention_backend
) {
    if (attention_backend == AttentionBackendKind::ScalarFlash) {
        if (request.precision_profile == kBlackwellNvfp4PrecisionProfile) {
            throw std::runtime_error(
                "blackwell_nvfp4 requires a private advanced runtime package");
        }
        // scalar_flash's own kernel (flash_attn_forward/backward, attention.cu)
        // operates on plain BF16 Q/K/V pointers directly -- it has no FP8
        // dependency of its own. This gate only constrains the LINEAR GEMM
        // precision profile, an orthogonal setting; "legacy_fp8" was the
        // only value ever validated here historically (matching whichever
        // families used scalar_flash at the time), not a hardware/kernel
        // requirement. "legacy_bf16" added 2026-07-22 as a real, equally
        // valid alternative once a family's own FP8-vs-BF16 ablation says so
        // (see the Edge precision fix in remote_train_lambda.sh for the
        // methodology this follows).
        if (request.precision_profile != "legacy_fp8" &&
            request.precision_profile != "legacy_bf16" &&
            request.precision_profile != "ampere_fp8_packed") {
            throw std::runtime_error(
                "scalar_flash attention backend requires precision_profile=legacy_fp8, legacy_bf16, or ampere_fp8_packed");
        }
        if (request.model.heads > 0) {
            const int kv_heads = request.model.kv_heads > 0
                ? request.model.kv_heads : request.model.heads;
            if (kv_heads <= 0 || kv_heads > request.model.heads ||
                (request.model.heads % kv_heads) != 0) {
                throw std::runtime_error(
                    "scalar_flash attention requires kv_heads <= heads and heads divisible by kv_heads");
            }
            if (kv_heads != request.model.heads &&
                request.precision_profile != "legacy_bf16") {
                throw std::runtime_error(
                    "native GQA currently requires scalar_flash with precision_profile=legacy_bf16");
            }
        }
        return;
    }

    if (attention_backend == AttentionBackendKind::BlackwellMxf4Fp4) {
        throw std::runtime_error(
            "blackwell_mxf4_fp4 requires a private advanced runtime package");
    }

    if (attention_backend != AttentionBackendKind::HopperWgmmaPackedFp4 &&
        attention_backend != AttentionBackendKind::HopperWgmmaFp8) {
        return;
    }
    throw std::runtime_error(
        "advanced attention backends require a private advanced runtime package");
}

// ─── Sub-batch stream crossover (IDA_NATIVE_SUBBATCH_STREAMS=K) ──────────────
// K worker streams each run WHOLE micro-batches of the accumulation window
// concurrently (round-robin micro → worker), so one worker's scalar-attention
// backward (CUDA cores), another's GEMMs (tensor cores), and a third's
// elementwise/norm passes (HBM bandwidth) overlap instead of serializing —
// attacking the measured ~14% MFU at AI dims.  Weights are shared read-only
// within the window; every piece of MUTATING per-call state is per-worker:
// scratch buffers, the packed-FP4 per-layer delayed-scaling state, cuBLAS
// handle, and a full gradient set (GEMM beta=1 accumulation is not atomic —
// concurrent streams must not share accumulators).  Worker grads reduce into
// the main set at the window boundary, then the untouched norm/clip/optimizer
// path runs.  Gradient math is bit-order-different but semantically identical
// to sequential accumulation (addition reassociation only).
// v1 restriction: FP8-linear profiles (f8.on) fall back to K=1 — ActSlot
// delayed-scaling state is per-layer, not per-worker, and sharing it across
// concurrent streams is the same bug class the July-4 doc's Bug 2 fixed.
// The production recipe (hopper_bf16_packed_fp4) runs BF16 linears: f8 off.
struct SubbatchWorker {
    NativeArena arena{};   // shares device/pool with the main arena; own stream
    StepBuffers sb{};
    Fp8Ctx f8{};
    PackedFp4AttentionCtx fp4{};
    cublasHandle_t cublas{};
    LatticeGrads g{};
    uint32_t* d_tokens{};
    int32_t*  d_labels{};
    uint16_t* d_segs{};
    cudaEvent_t ev_done{};
};

static void accumulate_grads_into(
    LatticeGrads& dst, const LatticeGrads& src,
    const LatticeWeights& w, cudaStream_t s
) {
    const std::size_t H = w.hidden_size, I = w.intermediate_size;
    const std::size_t V = w.vocab_size;
    const std::size_t KV = static_cast<std::size_t>(w.kv_heads > 0 ? w.kv_heads : w.heads) *
        static_cast<std::size_t>(w.hidden_size / w.heads);
    auto acc = [&](float* d, const float* x, std::size_t n) {
        if (d && x && n) {
            k_acc_f32<<<ceildiv(n, 256), 256, 0, s>>>(d, x, n);
        }
    };
    auto acc_b = [&](__nv_bfloat16* d, const __nv_bfloat16* x, std::size_t n) {
        if (d && x && n) {
            // The data-parallel worker path is not used by the 1F1B profile,
            // but retain a correctly rounded BF16 reduction for it.
            k_inplace_add_bf16<<<ceildiv(n, 256), 256, 0, s>>>(d, x, n);
        }
    };
    acc(dst.g_embed,   src.g_embed,   V * H);
    acc(dst.g_position_embeddings, src.g_position_embeddings,
        static_cast<std::size_t>(w.max_position_embeddings) * H);
    acc(dst.g_fnorm,   src.g_fnorm,   H);
    acc(dst.g_fnorm_bias, src.g_fnorm_bias, H);
    acc(dst.g_lm_head, src.g_lm_head, V * H);
    if (w.lrss_enabled) {
        const std::size_t J = w.lrss_scales;
        acc(dst.g_lrss_query,   src.g_lrss_query,   H * H);
        acc(dst.g_lrss_key,     src.g_lrss_key,     H * H);
        acc(dst.g_lrss_gate_w,  src.g_lrss_gate_w,  H * 2 * H);
        acc(dst.g_lrss_gate_b,  src.g_lrss_gate_b,  H);
        acc(dst.g_lrss_log_tau, src.g_lrss_log_tau, J);
        acc(dst.g_lrss_scale_w, src.g_lrss_scale_w, J);
        if (w.lss_rank > 0) {
            acc(dst.g_lss_down, src.g_lss_down, static_cast<std::size_t>(w.lss_rank) *
                (static_cast<std::size_t>(2 * H) + static_cast<std::size_t>(w.pss_spike_joint_dim)));
            acc(dst.g_lss_up,   src.g_lss_up,   H * static_cast<std::size_t>(w.lss_rank));
        }
    }
    if (w.pss_pred_rank > 0) {
        const std::size_t n = H * static_cast<std::size_t>(w.pss_pred_rank);
        acc(dst.g_pss_pred_down, src.g_pss_pred_down, n);
        acc(dst.g_pss_pred_up,   src.g_pss_pred_up,   n);
    }
    for (int l = 0; l < w.num_layers; ++l) {
        auto& d = dst.layers[l]; auto& x = src.layers[l];
        // Worker reductions are not part of model-parallel execution; this
        // branch is retained for the legacy data-parallel helper and uses a
        // BF16 add kernel when its layer gradients are BF16.
        acc_b(d.g_anorm, x.g_anorm, H);
        acc_b(d.g_q, x.g_q, H * H); acc_b(d.g_k, x.g_k, H * KV);
        acc_b(d.g_v, x.g_v, H * KV); acc_b(d.g_o, x.g_o, H * H);
        acc(d.g_o_bias, x.g_o_bias, H);
        acc(d.g_attn_norm_bias, x.g_attn_norm_bias, H);
        acc(d.g_ffn_norm_bias, x.g_ffn_norm_bias, H);
        acc(d.g_ffn_in_bias, x.g_ffn_in_bias, I);
        acc(d.g_ffn_out_bias, x.g_ffn_out_bias, H);
        acc_b(d.g_fnorm, x.g_fnorm, H);
        acc_b(d.g_gate, x.g_gate, H * I); acc_b(d.g_up, x.g_up, H * I);
        acc_b(d.g_down, x.g_down, I * H);
        // MoE gradients, mirroring zero_lattice_grads below exactly. Before
        // this block, worker gradients for every MoE tensor were simply never
        // reduced into the main set: with IDA_NATIVE_SUBBATCH_STREAMS=K>1 on
        // an MoE body, every expert, the router and the pressure tensors
        // trained at roughly 1/K the effective batch while the dense tensors
        // got the full window, silently. The K>1 refusal covers f8 and LRSS
        // but never covered MoE (it now does, belt and braces).
        if (d.g_moe_expert_in && x.g_moe_expert_in) {
            const auto& lw = w.layers[l];
            const std::size_t I_e = static_cast<std::size_t>(lw.expert_intermediate_size);
            if (d.g_moe_trunk_in && x.g_moe_trunk_in) {
                acc_b(d.g_moe_trunk_in,  x.g_moe_trunk_in,  static_cast<std::size_t>(I) * H);
                acc_b(d.g_moe_trunk_out, x.g_moe_trunk_out, static_cast<std::size_t>(H) * I);
            }
            acc_b(d.g_moe_expert_in,  x.g_moe_expert_in,
                static_cast<std::size_t>(lw.num_experts) * I_e * H);
            acc_b(d.g_moe_expert_out, x.g_moe_expert_out,
                static_cast<std::size_t>(lw.num_experts) * H * I_e);
            acc_b(d.g_pressure_proj, x.g_pressure_proj,
                static_cast<std::size_t>(lw.num_routes) * H);
            acc_b(d.g_pressure_mod,  x.g_pressure_mod,
                H * static_cast<std::size_t>(lw.num_routes));
            acc_b(d.g_router_score,  x.g_router_score,
                static_cast<std::size_t>(lw.num_experts) * H);
            if (d.g_pressure_to_routes && x.g_pressure_to_routes) {
                acc_b(d.g_pressure_to_routes, x.g_pressure_to_routes,
                    static_cast<std::size_t>(lw.num_experts) * lw.num_routes);
            }
        }
    }
}

static void zero_lattice_grads(
    LatticeGrads& g, const LatticeWeights& w, cudaStream_t s
) {
    const std::size_t H = w.hidden_size, I = w.intermediate_size;
    const std::size_t V = w.vocab_size;
    const std::size_t KV = static_cast<std::size_t>(w.kv_heads > 0 ? w.kv_heads : w.heads) *
        static_cast<std::size_t>(w.hidden_size / w.heads);
    auto z = [&](float* p, std::size_t n) {
        if (p && n) {
            k_zeros_f32<<<ceildiv(n, 256), 256, 0, s>>>(p, n);
        }
    };
    auto zb = [&](__nv_bfloat16* p, std::size_t n) {
        if (p && n) {
            k_zeros_bf16<<<ceildiv(n, 256), 256, 0, s>>>(p, n);
        }
    };
    z(g.g_embed, V * H);
    z(g.g_position_embeddings, static_cast<std::size_t>(w.max_position_embeddings) * H);
    z(g.g_fnorm, H); z(g.g_fnorm_bias, H);
    z(g.g_lm_head, V * H);
    if (w.lrss_enabled) {
        const std::size_t J = w.lrss_scales;
        z(g.g_lrss_query, H * H); z(g.g_lrss_key, H * H);
        z(g.g_lrss_gate_w, H * 2 * H); z(g.g_lrss_gate_b, H);
        z(g.g_lrss_log_tau, J); z(g.g_lrss_scale_w, J);
        if (w.lss_rank > 0) {
            z(g.g_lss_down, static_cast<std::size_t>(w.lss_rank) *
                (static_cast<std::size_t>(2 * H) + static_cast<std::size_t>(w.pss_spike_joint_dim)));
            z(g.g_lss_up,   H * static_cast<std::size_t>(w.lss_rank));
        }
    }
    if (w.pss_pred_rank > 0) {
        const std::size_t n = H * static_cast<std::size_t>(w.pss_pred_rank);
        z(g.g_pss_pred_down, n);
        z(g.g_pss_pred_up,   n);
    }
    for (int l = 0; l < w.num_layers; ++l) {
        auto& gl = g.layers[l];
        zb(gl.g_anorm, H);
        zb(gl.g_q, H * H); zb(gl.g_k, H * KV); zb(gl.g_v, H * KV); zb(gl.g_o, H * H);
        zb(gl.g_fnorm, H);
        z(gl.g_q_bias, H); z(gl.g_k_bias, KV); z(gl.g_v_bias, KV);
        z(gl.g_o_bias, H); z(gl.g_attn_norm_bias, H);
        z(gl.g_ffn_norm_bias, H); z(gl.g_ffn_in_bias, I); z(gl.g_ffn_out_bias, H);
        zb(gl.g_gate, H * I); zb(gl.g_up, H * I); zb(gl.g_down, I * H);
        if (gl.g_moe_expert_in) {
            const auto& lw = w.layers[l];
            const std::size_t I_e = static_cast<std::size_t>(lw.expert_intermediate_size);
            if (gl.g_moe_trunk_in) {
                zb(gl.g_moe_trunk_in,  static_cast<std::size_t>(I) * H);
                zb(gl.g_moe_trunk_out, static_cast<std::size_t>(H) * I);
            }
            zb(gl.g_moe_expert_in,  static_cast<std::size_t>(lw.num_experts) * I_e * H);
            zb(gl.g_moe_expert_out, static_cast<std::size_t>(lw.num_experts) * H * I_e);
            zb(gl.g_pressure_proj, static_cast<std::size_t>(lw.num_routes) * H);
            zb(gl.g_pressure_mod,  H * static_cast<std::size_t>(lw.num_routes));
            zb(gl.g_router_score, static_cast<std::size_t>(lw.num_experts) * H);
            if (gl.g_pressure_to_routes) {
                zb(gl.g_pressure_to_routes, static_cast<std::size_t>(lw.num_experts) * lw.num_routes);
            }
        }
    }
}

// ─── LRSS scratch lifecycle ──────────────────────────────────────────────────

static LrssScratch* alloc_lrss_scratch(
    const LatticeWeights& w, int mb, NativeArena& arena
) {
    // Hard bounds: the kernels and scratch carry fixed 64-entry local arrays
    // (k_lrss_attn logits[64], refresh elapsed_h[64], LrssScratch::times[64]).
    // These are compile-time facts, not tunables — a config above them is
    // memory corruption, so it fails at allocation, loudly.
    if (w.lrss_anchors > 64 || w.lrss_scales > 64) {
        throw std::runtime_error(
            "LRSS config exceeds kernel bounds: anchors=" +
            std::to_string(w.lrss_anchors) + " scales=" +
            std::to_string(w.lrss_scales) + " (max 64 each)");
    }
    auto* s = new LrssScratch{};
    const std::size_t H = w.hidden_size;
    const std::size_t A = w.lrss_anchors;
    const std::size_t B = mb;
    const std::size_t J = w.lrss_scales;
    auto az = [&](std::size_t n) { return alloc_f32(n, arena); };
    s->ring     = az(A * H);
    s->bank     = az(A * H);
    s->elapsed  = az(A);
    s->pooled   = az(B * H);  s->n_content = az(B);
    s->q        = az(B * H);  s->k         = az(A * H);
    s->rel      = az(B * A);  s->attn      = az(J * B * A);
    s->sw       = az(J);      s->wsum      = az(B * A);
    s->mix      = az(B * H);  s->cat       = az(B * 2 * H);
    s->gpre     = az(B * H);  s->gate      = az(B * H);
    s->delta    = az(B * H);
    s->d_delta  = az(B * H);  s->dpre      = az(B * H);
    // dcat is reused as LSS's d_joint backward scratch (lrss_backward); when
    // PSS spike-joint is on, joint_dim > 2*H, so this buffer must be sized to
    // the wider use. The gate path above only ever touches its first B*2*H
    // elements, so this widening is invisible to it.
    const std::size_t joint_dim = static_cast<std::size_t>(2 * H) +
        static_cast<std::size_t>(w.pss_spike_joint_dim);
    s->d_mix    = az(B * H);  s->dcat      = az(B * joint_dim);
    s->d_pooled = az(B * H);  s->dwsum     = az(B * A);
    s->drel     = az(B * A);  s->dot_j     = az(J);
    s->dq       = az(B * H);  s->dk        = az(A * H);
    s->dq_in    = az(B * H);
    if (w.lss_rank > 0) {
        const std::size_t R = w.lss_rank;
        IDA_CUDA_CHECK(ida_malloc_async(&s->lss_anchor_q, H, arena.pool, arena.stream));
        s->lss_anchor_scale = az(1);
        s->lss_anchor  = az(H);
        s->lss_joint   = az(B * joint_dim);
        s->lss_hidden  = az(B * R);
        s->lss_recon   = az(B * H);
        s->lss_aux     = az(1);
        s->d_lss_recon  = az(B * H);
        s->d_lss_hidden = az(B * R);
        s->mix_bank_save= az(B * H);   // bank mix saved before injection (injection mode only)
        if (w.pss_spike_joint_dim > 0) {
            s->pss_spike = az(static_cast<std::size_t>(w.pss_spike_joint_dim));
            IDA_CUDA_CHECK(cudaMemsetAsync(
                s->pss_spike, 0,
                static_cast<std::size_t>(w.pss_spike_joint_dim) * sizeof(float),
                arena.stream));
        }
    }
    IDA_CUDA_CHECK(cudaMemsetAsync(s->ring, 0, A * H * sizeof(float), arena.stream));
    return s;
}

static void free_lrss_scratch(LrssScratch* s, NativeArena& arena) {
    if (!s) return;
    auto f = [&](float* p) { if (p) IDA_CUDA_CHECK(cudaFreeAsync(p, arena.stream)); };
    f(s->ring); f(s->bank); f(s->elapsed); f(s->pooled); f(s->n_content);
    f(s->q); f(s->k); f(s->rel); f(s->attn); f(s->sw); f(s->wsum);
    f(s->mix); f(s->cat); f(s->gpre); f(s->gate); f(s->delta);
    f(s->d_delta); f(s->dpre); f(s->d_mix); f(s->dcat); f(s->d_pooled);
    f(s->dwsum); f(s->drel); f(s->dot_j); f(s->dq); f(s->dk); f(s->dq_in);
    if (s->lss_anchor_q) IDA_CUDA_CHECK(cudaFreeAsync(s->lss_anchor_q, arena.stream));
    f(s->lss_anchor_scale); f(s->lss_anchor); f(s->lss_joint);
    f(s->lss_hidden); f(s->lss_recon); f(s->lss_aux);
    f(s->d_lss_recon); f(s->d_lss_hidden); f(s->mix_bank_save);
    f(s->pss_spike);
    delete s;
}
static void validate_model_contract_request(
    const NativeRequest& request, AttentionBackendKind attention_backend
) {
    if (request.model.moe_native_fp4) {
        throw std::runtime_error(
            "private packed-FP4 activation requires a private advanced runtime package");
    }
    if (request.model.generic_moe_shared_expert_width > 0) {
        throw std::runtime_error(
            "generic MoE shared-expert layouts require a separate validated contract");
    }
    const std::string& contract = request.model.architecture_contract.empty()
        ? request.architecture_contract : request.model.architecture_contract;
    if (contract != "hf_gpt2_native_v1") return;
    if (attention_backend != AttentionBackendKind::ScalarFlash ||
        request.precision_profile != "legacy_bf16")
        throw std::runtime_error("hf_gpt2_native_v1 requires scalar_flash with precision_profile=legacy_bf16");
    if (request.model.normalization_type != "layernorm")
        throw std::runtime_error("hf_gpt2_native_v1 requires normalization_type=layernorm");
    if (request.model.position_embedding_type != "learned_absolute" ||
        (request.model.activation_type != "gelu_new" && request.model.activation_type != "gelu"))
        throw std::runtime_error("hf_gpt2_native_v1 requires learned_absolute positions and GELU activation");
    if (request.model.max_position_embeddings <= 0)
        throw std::runtime_error("hf_gpt2_native_v1 requires max_position_embeddings > 0");
    if (request.input.sequence_length > request.model.max_position_embeddings)
        throw std::runtime_error("hf_gpt2_native_v1 position table is shorter than sequence_length");
    const int kv_heads = request.model.kv_heads > 0 ? request.model.kv_heads : request.model.heads;
    if (request.model.hidden_size <= 0 || request.model.heads <= 0 ||
        request.model.hidden_size % request.model.heads != 0 || kv_heads != request.model.heads)
        throw std::runtime_error("hf_gpt2_native_v1 requires dense equal Q/K/V head counts");
    if (!request.model.qkv_bias || !request.model.projection_bias)
        throw std::runtime_error("hf_gpt2_native_v1 requires QKV and projection bias fields");
    if (request.model.generic_moe_num_experts > 0 || request.model.num_cognitive_routes > 0 ||
        request.model.num_personality_experts > 0)
        throw std::runtime_error("hf_gpt2_native_v1 does not support MoE fields");
}


// ─── main training entry point ────────────────────────────────────────────────

BurnResult run_lattice_training(
    const NativeRequest& request,
    NativeArena& arena,
    ProgressCallback on_step
) {
    // MLPerf Open Division log: submission metadata + init interval first.
    MlLog ml;
    ml.open(request.output_dir / "mlperf_log.txt");
    ml.point("submission_benchmark", MlLog::q("llm_pretrain_ida_lattice_edge"));
    ml.point("submission_division",  MlLog::q("open"));
    ml.point("submission_org",       MlLog::q("HiddenCanopy"));
    ml.point("submission_platform",  MlLog::q("1xH100_80GB_HBM3"));
    ml.point("submission_status",    MlLog::q("research"));
    ml.point("cache_clear", "true");
    ml.begin("init_start");
    write_native_timeline_event(request, "native_init", "start");
    const auto native_init_t0 = std::chrono::steady_clock::now();

    const auto attention_backend = parse_attention_backend(request.attention_backend);
    validate_attention_request(request, attention_backend);
    validate_precision_state_request(request);
    validate_model_contract_request(request, attention_backend);
    // Cognitive-architecture port (2026-07-21): let the real config's
    // local_attention_window actually reach the engine instead of being
    // silently dropped by the head_dim heuristic (0 = unset -> -1 =
    // fall back to that heuristic / the env override).
    set_attn_window_request_override(
        request.model.local_attention_window > 0 ? request.model.local_attention_window : -1);
    // Native packed-FP4 MoE activation path: per-burn thread_local, set
    // once here from this burn's own request (see moe_native_fp4_enabled's
    // doc comment for why this isn't a raw env var).

    // 1. Load dataset
    HostDataset ds = load_host_dataset(request);

    // 2. Allocate model + optimizer + gradient buffers
    LatticeWeights  w   = allocate_lattice_weights(request, arena);
    LatticeOptState opt = allocate_lattice_opt(request, w, arena);
    LatticeGrads    g   = allocate_lattice_grads(w, arena);
    auto slots = build_param_slots(w, opt, g, /*wd=*/0.01f);

    // ── Exact resume (Phase 3, 2026-07-22) ──────────────────────────────────
    // Distinct from parent.init_from_model above (cross-lineage, weights-
    // only, always fresh optimizer/cursor state): resume_from_checkpoint is
    // THIS SAME BURN continuing after an interruption. A failure here is a
    // hard error, not a silent fresh-init fallback -- proceeding with zeroed
    // optimizer state while believing it resumed would silently corrupt
    // training dynamics in exactly the way this mechanism exists to prevent.
    ResumeState resume_state{};
    if (!request.parent.resume_from_checkpoint.empty()) {
        std::string err;
        int weights_cumulative_opt_steps = -1;
        if (!load_lattice_weights_safetensors(
                request.parent.resume_from_checkpoint, w, arena.stream, err,
                request.spec_hash, &weights_cumulative_opt_steps)) {
            throw std::runtime_error(
                "resume_from_checkpoint weights load failed (" +
                request.parent.resume_from_checkpoint.string() + "): " + err);
        }
        if (!load_lattice_opt_safetensors(
                request.parent.resume_from_checkpoint, w, opt, resume_state,
                arena.stream, err)) {
            throw std::runtime_error(
                "resume_from_checkpoint optimizer-state load failed (" +
                request.parent.resume_from_checkpoint.string() + "): " + err);
        }
        // Checkpoint-atomicity cross-check (2026-07-23): model.safetensors
        // and optimizer_state.safetensors are two separate atomic
        // tmp+rename writes (see save_lattice_weights_safetensors'
        // cumulative_opt_steps parameter) -- a kill between them leaves a
        // torn generation pair. Both -1 (either file predates this field)
        // skips the check; a real, tracked disagreement is a hard failure,
        // not a silent resume from mismatched state.
        if (weights_cumulative_opt_steps >= 0 &&
            weights_cumulative_opt_steps != resume_state.cumulative_opt_steps) {
            throw std::runtime_error(
                "resume_from_checkpoint generation mismatch: model.safetensors "
                "recorded cumulative_opt_steps=" +
                std::to_string(weights_cumulative_opt_steps) +
                " but optimizer_state.safetensors recorded " +
                std::to_string(resume_state.cumulative_opt_steps) +
                " (torn checkpoint pair -- the process likely crashed between "
                "the two saves; re-genesis or resume from an earlier, "
                "matched checkpoint generation instead)");
        }
        std::fprintf(stderr,
            "[ida_native_train] resumed: %s (cumulative_opt_steps=%d, "
            "cumulative_micro_steps=%d, dataset_cursor=%zu)\n",
            request.parent.resume_from_checkpoint.string().c_str(),
            resume_state.cumulative_opt_steps, resume_state.cumulative_micro_steps,
            resume_state.dataset_cursor);
    }
    const bool lrss_freeze = [] {
        const char* e = std::getenv("IDA_NATIVE_LRSS_FREEZE");
        return e && e[0] == '1';
    }();
    const float lrss_lr_scale = [] {
        const char* e = std::getenv("IDA_NATIVE_LRSS_LR_SCALE");
        return e ? std::max(0.0f, static_cast<float>(std::atof(e))) : 1.0f;
    }();
    // PSS Stage 2: the predictor's optimizer group otherwise silently shares
    // the trunk's own LR schedule (warmup/decay tuned for a ~1B-param model
    // already partway through training), not a freshly-initialized rank-64
    // auxiliary head with no pretrained prior. First GPU evidence
    // (2026-07-15) showed confidence flat and error rising over 4 windows
    // at scale 1.0 -- this is the first lever to try before concluding the
    // architecture itself is the problem. Default 1.0 (no behavior change).
    const float pss_pred_lr_scale = [&] {
        // Per-burn request field wins (worker-safe); env is the direct-launch
        // fallback. Negative request value = unset.
        if (request.pss_pred_lr_scale >= 0.0f) return request.pss_pred_lr_scale;
        const char* e = std::getenv("IDA_NATIVE_PSS_PRED_LR_SCALE");
        // e[0] (non-empty), not just e (non-null): a launcher's ${VAR:-}
        // default when unset passes through as a non-null empty string --
        // atof("") == 0.0, which would silently zero the predictor's LR
        // instead of leaving it at this default.
        return (e && e[0]) ? std::max(0.0f, static_cast<float>(std::atof(e))) : 1.0f;
    }();
    const bool optimizer_sr_seed_mix = [] {
        const char* e = std::getenv("IDA_NATIVE_SR_SEED_MIX");
        return e && e[0] == '1';
    }();
    const bool lrss_bias_debug = [] {
        const char* e = std::getenv("IDA_NATIVE_LRSS_BIAS_DEBUG");
        return e && e[0] == '1';
    }();
    unsigned lrss_update_mask = [] {
        const char* e = std::getenv("IDA_NATIVE_LRSS_UPDATE_MASK");
        // Gate-bias mutation alone moved the fixed packed-FP4 AI genesis
        // run from GN 4.99 to 131.78 on its second window. The first Adam
        // write itself was finite and correctly bounded at +/-1.00136e-4;
        // the failure is downstream quantized-path sensitivity, not an
        // optimizer overflow. Keep the parameter for checkpoint parity but
        // freeze only its update by default. Set 0x1f to reproduce the old
        // behavior explicitly.
        return e ? static_cast<unsigned>(std::strtoul(e, nullptr, 0)) : 0x0fu;
    }();
    // PSS Stage 2 (group 0x20): unconditionally included whenever the
    // predictor is enabled, regardless of IDA_NATIVE_LRSS_UPDATE_MASK --
    // it's architecturally independent of the LRSS bank this mask was built
    // to diagnose, and an operator forgetting to widen the mask should never
    // silently produce a predictor that never learns (dead weights, aux
    // loss frozen, confidence never calibrates).
    if (w.pss_pred_rank > 0) lrss_update_mask |= 0x20u;
    auto is_lrss_weight = [&](const ParamSlot& p) {
        return w.lrss_enabled &&
            (p.w == w.lrss_query || p.w == w.lrss_key ||
             p.w == w.lrss_gate_w || p.w == w.lrss_gate_b ||
             p.w == w.lrss_log_tau || p.w == w.lrss_scale_w ||
             p.w == w.lss_down || p.w == w.lss_up);
    };
    auto is_pss_pred_weight = [&](const ParamSlot& p) {
        return p.w == w.pss_pred_down || p.w == w.pss_pred_up;
    };
    auto lrss_weight_group = [&](const ParamSlot& p) -> unsigned {
        if (p.w == w.lrss_query || p.w == w.lrss_key) return 0x01u;
        if (p.w == w.lrss_gate_w) return 0x02u;
        if (p.w == w.lrss_log_tau || p.w == w.lrss_scale_w) return 0x04u;
        if (p.w == w.lss_down || p.w == w.lss_up) return 0x08u;
        if (p.w == w.lrss_gate_b) return 0x10u;
        if (p.w == w.pss_pred_down || p.w == w.pss_pred_up) return 0x20u;
        return 0u;
    };

    const int S  = ds.seq_len;
    int mb = request.input.batch_size > 0 ? request.input.batch_size
           : request.training.microbatch > 0 ? request.training.microbatch : 32;
    if (mb < 1) mb = 1;
    if (mb > ds.num_sequences) mb = std::max(1, ds.num_sequences);

    // Novelty accumulation curriculum (pass-1/genesis ramp, mirrors the
    // PyTorch trainer): accum starts at 1 and doubles at 7.5% / 15% / 20% /
    // 25% / 30% of data exposure, capped at the config ceiling.
    //
    // Queue-driven native ablations pass grad_accum_override explicitly.
    // That override is authoritative and must pin accumulation instead of
    // silently ramping through the genesis curriculum.
    const int accum_ceiling = request.training.grad_accumulation > 0
        ? request.training.grad_accumulation : 4;
    const bool fixed_accumulation = request.grad_accum_override > 0;

    // max_steps counts micro-steps (data exposure), matching the Python wrapper.
    int total_micro = request.training.max_steps > 0
        ? request.training.max_steps
        : (ds.num_sequences / mb) * 2;
    if (total_micro < 1) total_micro = 1;

    // max_samples caps total_micro (micro-batches). Round up to accum_ceiling
    // so at least one optimizer step fires — a canary that skips the optimizer
    // update cannot verify the full training chain.
    if (request.max_samples > 0) {
        const int accum_for_cap = std::max(1, accum_ceiling);
        const int capped = ((request.max_samples + accum_for_cap - 1) / accum_for_cap) * accum_for_cap;
        total_micro = std::min(total_micro, capped);
    }

    {
        ::ida_native::ontology::Context octx;
        octx.microbatch        = mb;
        octx.grad_accum        = accum_ceiling;
        octx.num_experts       = request.model.num_personality_experts;
        octx.top_k_experts     = request.model.top_k_experts;
        octx.seq_window        = request.model.local_attention_window;
        octx.precision_profile = request.precision_profile;
        octx.attention_backend = attention_backend_name(attention_backend);
        octx.optimizer         = request.optimizer_type;
        // No body_key field on the request; family/seat/version is the same
        // triple the queue scripts key a worker slot on, so records from two
        // co-resident bodies stay separable in one shared sink.
        octx.body_key = request.family + "/" + request.seat + "/" + request.version;
        octx.resident_bodies = -1;   // not yet plumbed from main.cpp's slot map
        ::ida_native::ontology::set_context(octx);
    }
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
    if (request.ontology_required &&
        (!::ida_native::ontology::enabled() || !::ida_native::ontology::sink_is_file())) {
        throw std::runtime_error(
            "ontology_required but the native trainer thread could not open the ontology sink");
    }
#endif

    auto accum_at = [&](int micro_idx) -> int {
        if (fixed_accumulation) return std::max(1, accum_ceiling);
        static const double kThresholds[] = {0.075, 0.15, 0.20, 0.25, 0.30};
        const double frac = static_cast<double>(micro_idx) /
                            static_cast<double>(total_micro);
        int a = 1;
        for (double t : kThresholds)
            if (frac >= t && a < accum_ceiling) a <<= 1;
        return std::min(a, std::max(1, accum_ceiling));
    };

    // Pre-simulate the curriculum to size the LR schedule in optimizer steps.
    int total_opt = 0;
    for (int m = 0; m < total_micro; ++total_opt) m += accum_at(m);
    total_opt = std::max(1, total_opt);

    const double base_lr = request.training.learning_rate > 0.0
        ? request.training.learning_rate : 3e-4;
    // Global L2 gradient-norm ceiling. Hardcoded 1.0 since Era 10; found
    // 2026-07-18 to be 100% SATURATED in production (fired every step of
    // every family, natural GN 3.4-10.9 vs ceiling 1.0, scale 0.03-0.36) --
    // meaning it is not a spike guard but a constant ~10x gradient rescale,
    // and because the scale varies step-to-step it randomly reweights Lion's
    // momentum-vs-gradient interpolation by up to ~6x per step. Env override
    // added for the staged 3-arm probe (probe_ledger.jsonl
    // clip_threshold_probe_plan_20260718): unset/empty = 1.0 (exact current
    // production behavior); a positive value raises the ceiling; 0 disables
    // clipping entirely (the grad_norm_abs_ceiling circuit breaker below is
    // separate and stays active either way). DO NOT change the production
    // default before that probe passes -- every observed dynamic of this
    // lineage was tuned with the saturated clip in the loop.
    // request.global_clip_override (per-burn, wrapper-supplied) wins over
    // the env var -- see request.hpp for why: a shared multi-tenant process
    // can't represent a per-family clip ceiling via its own fixed env.
    // Not cached (read fresh each call, matching the pre-existing pattern
    // here) since this function already ran per-burn, not per-process.
    const float clip = [&request] {
        if (request.global_clip_override >= 0.0f) {
            return request.global_clip_override > 0.0f
                ? request.global_clip_override
                : std::numeric_limits<float>::max();
        }
        const char* e = std::getenv("IDA_NATIVE_GLOBAL_CLIP");
        if (e && e[0]) {
            const float v = static_cast<float>(std::atof(e));
            return v > 0.0f ? v : std::numeric_limits<float>::max();
        }
        return 1.0f;
    }();
    // Circuit-breaker absolute ceiling: L2 clipping rescales the WHOLE
    // gradient tensor by clip/grad_norm.  When grad_norm is dominated by
    // one pathological outlier (a single accumulation micro-step gone
    // wrong, uncaught because clipping only runs once per optimizer step
    // — see the loop below), that rescale (~1e-15 at grad_norm~1e14)
    // doesn't just shrink the outlier into range — it crushes every OTHER
    // gradient element below FP32's useful precision floor, so the
    // "clipped" update is effectively noise concentrated in one corrupted
    // slot, applied every step.  !isfinite() alone doesn't catch this: an
    // astronomical value can still be finite.  Observed 2026-07-09: AI
    // mb=128 combos hit grad_norm ~1e14-1e15 (finite), clipped+applied on
    // every optimizer step, and the model ended up WORSE than untrained
    // genesis (loss returned to ln(vocab_size)).  Threshold sits well
    // above legitimate transient noise (~1e6, see the Era 11 fp8 doc's
    // "1.05e6 expected early-training noise") and well below any observed
    // catastrophic value.
    float grad_norm_abs_ceiling = 1e8f;
    if (const char* e = std::getenv("IDA_NATIVE_GRAD_NORM_ABS_CEILING")) {
        try { grad_norm_abs_ceiling = std::stof(e); } catch (...) {}
    }
    const float b1 = 0.9f, b2 = 0.999f, eps_a = 1e-8f;
    const bool use_lion = optimizer_uses_lion(request);
    const float lion_b1 = lion_beta1(request), lion_b2 = lion_beta2(request);
    const float lion_lr_mult = lion_lr_scale(request), lion_wd_mult = lion_wd_scale(request);

    // 3. Buffers
    StepBuffers sb = alloc_step_buffers(w, mb, S, arena);

    uint32_t* d_tokens = nullptr;
    int32_t*  d_labels = nullptr;
    uint16_t* d_segs   = nullptr;
    IDA_CUDA_CHECK(ida_malloc_async(&d_tokens, static_cast<std::size_t>(mb) * S * sizeof(uint32_t), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&d_labels, static_cast<std::size_t>(mb) * S * sizeof(int32_t),  arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&d_segs,   static_cast<std::size_t>(mb) * S * sizeof(uint16_t), arena.pool, arena.stream));
    sb.segs = d_segs;

    cublasHandle_t cublas{};
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, arena.stream));
    // CUBLAS_TF32_TENSOR_OP_MATH silently truncates to TF32 (10-bit mantissa)
    // inside every GEMM's tensor-core MMA despite CUBLAS_COMPUTE_32F being
    // requested — a fixed precision/reduction-order noise floor present in
    // every projection (Q/K/V/O, FFN gate/up/down), independent of which
    // attention backend runs. CUBLAS_PEDANTIC_MATH forces true deterministic
    // FP32 tensor-core math (slower, no silent TF32 fallback).
    // IDA_CUBLAS_PEDANTIC=0 selects CUBLAS_DEFAULT_MATH for speed ablation —
    // on the FP8 profiles most GEMMs run through cuBLASLt (unaffected by the
    // handle mode), so this only moves the BF16 fallback GEMMs.
    {
        const char* e = std::getenv("IDA_CUBLAS_PEDANTIC");
        if (e && e[0] == '0') {
            CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH));
        } else {
            CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH));
        }
    }

    // 3b. FP8 path (activation scratch sized for the largest quantized tensor)
    const std::size_t max_act = std::max({
        static_cast<std::size_t>(mb) * S * 3 * w.hidden_size,
        static_cast<std::size_t>(mb) * S * w.intermediate_size,
        static_cast<std::size_t>(sb.ce_chunk) * w.vocab_size});
    Fp8Ctx f8 = build_fp8_ctx(request, w, max_act, mb * S, arena);
    AmperePackedWeights ampere_packed = build_ampere_packed_weights(request, w, arena);
    g_ampere_packed_weights = &ampere_packed;
    PackedFp4AttentionCtx fp4_attn = build_packed_fp4_attention_ctx(request, sb, w.num_layers, arena);
    bind_fp8_weight_amax_slots(slots, f8, w);
    refresh_fp8_weights(f8, arena.stream, /*recompute_amax=*/true);

    // ── Sub-batch stream crossover workers (see SubbatchWorker above) ────────
    int subbatch_streams = [] {
        const char* e = std::getenv("IDA_NATIVE_SUBBATCH_STREAMS");
        const int k = e ? std::atoi(e) : 1;
        return std::min(std::max(k, 1), 8);
    }();
    bool moe_on = false;
    for (int l = 0; l < w.num_layers; ++l) moe_on |= (w.layers[l].num_experts > 0);
    const bool pss_on = w.pss_pred_rank > 0;
    if (subbatch_streams > 1 && (f8.on || w.lrss_enabled || moe_on || pss_on)) {
        // MoE is in this refusal as the second line of defence:
        // accumulate_grads_into now reduces the MoE tensors, but a worker
        // path that has never run under K>1 on an MoE body does not get
        // promoted to trusted by one code review. Decline loudly rather than
        // train an unverified configuration.
        std::fprintf(stderr,
            "[ida_native_train] subbatch streams: %s state cannot be shared "
            "across concurrent streams; falling back to 1 stream\n",
            f8.on ? "FP8-linear ActSlot"
                  : (w.lrss_enabled ? "LRSS anchor-ring"
                                    : (moe_on ? "MoE gradient reduction is unverified under K>1"
                                              : "PSS fence evidence is main-stream owned")));
        subbatch_streams = 1;
    }
    std::vector<SubbatchWorker> xworkers;   // index k-1 = worker k (0 = main)
    for (int k = 1; k < subbatch_streams; ++k) {
        SubbatchWorker wk{};
        wk.arena = arena;
        IDA_CUDA_CHECK(cudaStreamCreateWithFlags(&wk.arena.stream,
                                                 cudaStreamNonBlocking));
        wk.sb = alloc_step_buffers(w, mb, S, wk.arena);
        IDA_CUDA_CHECK(ida_malloc_async(&wk.d_tokens,
            static_cast<std::size_t>(mb) * S * sizeof(uint32_t), wk.arena.pool, wk.arena.stream));
        IDA_CUDA_CHECK(ida_malloc_async(&wk.d_labels,
            static_cast<std::size_t>(mb) * S * sizeof(int32_t),  wk.arena.pool, wk.arena.stream));
        IDA_CUDA_CHECK(ida_malloc_async(&wk.d_segs,
            static_cast<std::size_t>(mb) * S * sizeof(uint16_t), wk.arena.pool, wk.arena.stream));
        wk.sb.segs = wk.d_segs;
        CUBLAS_CHECK(cublasCreate(&wk.cublas));
        CUBLAS_CHECK(cublasSetStream(wk.cublas, wk.arena.stream));
        {
            const char* e = std::getenv("IDA_CUBLAS_PEDANTIC");
            CUBLAS_CHECK(cublasSetMathMode(wk.cublas,
                (e && e[0] == '0') ? CUBLAS_DEFAULT_MATH : CUBLAS_PEDANTIC_MATH));
        }
        wk.f8  = build_fp8_ctx(request, w, max_act, mb * S, wk.arena);
        wk.fp4 = build_packed_fp4_attention_ctx(request, wk.sb, w.num_layers,
                                                wk.arena);
        wk.g   = allocate_lattice_grads(w, wk.arena);
        IDA_CUDA_CHECK(cudaEventCreateWithFlags(&wk.ev_done,
                                                cudaEventDisableTiming));
        xworkers.push_back(wk);
    }
    cudaEvent_t ev_weights{};
    IDA_CUDA_CHECK(cudaEventCreateWithFlags(&ev_weights, cudaEventDisableTiming));
    if (subbatch_streams > 1) {
        std::fprintf(stderr,
            "[ida_native_train] subbatch stream crossover: %d concurrent "
            "workers, mb=%d each\n", subbatch_streams, mb);
    }

    // ── LRSS wiring (IDA_NATIVE_LRSS=1) ──────────────────────────────────────
    LrssParams lrss_params{};
    LrssGrads  lrss_gview{};
    LrssScratch* lrss_scr = nullptr;
    if (w.lrss_enabled) {
        lrss_params.query_w = w.lrss_query;   lrss_params.key_w   = w.lrss_key;
        lrss_params.gate_w  = w.lrss_gate_w;  lrss_params.gate_b  = w.lrss_gate_b;
        lrss_params.log_tau = w.lrss_log_tau; lrss_params.scale_w = w.lrss_scale_w;
        lrss_params.num_scales = w.lrss_scales;
        lrss_gview.query_w = g.g_lrss_query;   lrss_gview.key_w   = g.g_lrss_key;
        lrss_gview.gate_w  = g.g_lrss_gate_w;  lrss_gview.gate_b  = g.g_lrss_gate_b;
        lrss_gview.log_tau = g.g_lrss_log_tau; lrss_gview.scale_w = g.g_lrss_scale_w;
        if (w.lss_rank > 0) {
            lrss_params.lss_down = w.lss_down;
            lrss_params.lss_up   = w.lss_up;
            lrss_params.lss_rank = w.lss_rank;
            lrss_params.pss_spike_dim = w.pss_spike_joint_dim;
            lrss_gview.lss_down  = g.g_lss_down;
            lrss_gview.lss_up    = g.g_lss_up;
            // This was historically not wired to the env override, so the
            // struct default silently governed every run regardless of
            // IDA_NATIVE_LSS_AUX_WEIGHT, invalidating every "aux_weight=0"
            // ablation run before this fix (2026-07-11).
            if (const char* aw = std::getenv("IDA_NATIVE_LSS_AUX_WEIGHT")) {
                lrss_params.lss_aux_weight = std::atof(aw);
            }
        }
        lrss_scr = alloc_lrss_scratch(w, mb, arena);
        sb.lrss_p = &lrss_params;
        sb.lrss_s = lrss_scr;
        sb.lrss_g = &lrss_gview;
        std::fprintf(stderr,
            "[ida_native_train] LRSS multiscale memory: J=%d scales, A=%d "
            "anchors, +%.1f MiB params\n",
            w.lrss_scales, w.lrss_anchors,
            (2.0 * w.hidden_size * w.hidden_size + 2.0 * w.hidden_size * w.hidden_size)
                * sizeof(__nv_bfloat16) / (1024.0 * 1024.0));
    }

    const auto slot_names = build_param_slot_names(w);
    float* d_slot_normsq = alloc_f32(slots.size(), arena);
    // Parameter-norm companion to d_slot_normsq (weights were never measured).
    float* d_slot_wnormsq = alloc_f32(slots.size(), arena);
    std::vector<float> h_slot_wnormsq(slots.size(), 0.0f);
    if (act_row_clip_mode() == 1 && g_act_row_ema == nullptr) {
        g_act_row_ema = alloc_f32(2, arena);
        IDA_CUDA_CHECK(cudaMemsetAsync(g_act_row_ema, 0, 2 * sizeof(float), arena.stream));
        std::fprintf(stderr,
            "[ida_native_train] act_row_clip: EMA-adaptive ceiling "
            "(mult=%.2f beta=%.4f) -- replaces the fixed ceiling proven causal "
            "for MoE divergence 2026-07-30\n",
            act_row_clip_ema_mult(), act_row_clip_ema_beta());
    }
    if (rowclip_debug_enabled() && g_rowclip_stats == nullptr) {
        g_rowclip_stats = alloc_f32(16, arena);
        IDA_CUDA_CHECK(cudaMemsetAsync(g_rowclip_stats, 0, 16 * sizeof(float), arena.stream));
        IDA_CUDA_CHECK(cudaMemcpyToSymbolAsync(g_rowclip_dev_stats, &g_rowclip_stats,
                                               sizeof(float*), 0,
                                               cudaMemcpyHostToDevice, arena.stream));
        std::fprintf(stderr,
            "[ida_native_train] row-clip ladder probe ARMED "
            "(act/interlayer/dqkv/qk firing + ceiling vs max row norm)\n");
    }
    if (fp8_clip_debug_enabled() && g_fp8_clip_stats == nullptr) {
        g_fp8_clip_stats = alloc_f32(3, arena);
        IDA_CUDA_CHECK(cudaMemsetAsync(g_fp8_clip_stats, 0, 3 * sizeof(float), arena.stream));
        g_fp8_elem_stats = alloc_f32(2, arena);
        IDA_CUDA_CHECK(cudaMemsetAsync(g_fp8_elem_stats, 0, 2 * sizeof(float), arena.stream));
        std::fprintf(stderr,
            "[ida_native_train] FP8 clip probe ARMED (delayed-scaling saturation)\n");
    }
    std::vector<float> h_slot_normsq(slots.size(), 0.0f);
    // PSS Stage 1a: per-slot EMA-relative spike detection + adaptive clip.
    // Reuses the per-slot norms the dominant-slot argmax already reads back
    // to host every optimizer step -- no new device->host sync. A slot whose
    // norm spikes far above its own recent EMA gets clipped to
    // mult*EMA before the global L2 clip runs, so one slot's transient blow-up
    // (e.g. the gate-bias spike, mask 0x10) doesn't force every other slot's
    // step down through the shared global clip scale.
    // IDA_NATIVE_PSS_SLOT_CLIP=1 to enable (default off, lab flag).
    std::vector<float> pss_slot_norm_ema(slots.size(), 0.0f);
    const bool pss_slot_clip_enabled = [] {
        const char* e = std::getenv("IDA_NATIVE_PSS_SLOT_CLIP");
        return e && e[0] == '1';
    }();
    const float pss_slot_clip_mult = [] {
        const char* e = std::getenv("IDA_NATIVE_PSS_SLOT_CLIP_MULT");
        return e ? std::max(1.0f, static_cast<float>(std::atof(e))) : 4.0f;
    }();
    const float pss_spike_ema_beta = [] {
        const char* e = std::getenv("IDA_NATIVE_PSS_SPIKE_EMA_BETA");
        const float v = e ? static_cast<float>(std::atof(e)) : 0.9f;
        return std::min(0.999f, std::max(0.0f, v));
    }();
    constexpr float kPssSlotNormFloor = 1e-6f;
    const float embed_clip_threshold = embed_row_clip_threshold();
    const std::string pss_conditioning_mode =
        pss_policy_mode(request.pss_conditioning_mode, "IDA_NATIVE_PSS_CONDITIONING");
    const std::string pss_aux_normalize_mode =
        pss_policy_mode(request.pss_aux_normalize_mode, "IDA_NATIVE_PSS_AUX_NORMALIZE");

    // Optimizer Stage 3 v2: per-slot trust ratio state (Lion only, default
    // off -- see trust_ratio_enabled()'s comment). slot_trust_ratio starts
    // at 1.0 (no-op) for every slot and is only overwritten once a slot's
    // EMA has warmed past kPssSlotNormFloor, so cold start / an operator
    // enabling the flag mid-run never sees an undefined multiplier.
    std::vector<float> slot_trust_ratio(slots.size(), 1.0f);

    // 3c. Metrics: per-step JSONL + optimizer summary arrays.
    // The per-burn training_metrics.jsonl is the canonical evidence file for
    // dashboards/HF.  The repo-level native stream is append-only for live
    // backfills and never replaces the per-burn record.
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
    std::ofstream shared_metrics_out;
    std::ofstream burn_metrics_out;
    {
        const auto mpath = request.repo_root / "artifacts" / "telemetry" /
                           "native_training_metrics.jsonl";
        std::error_code ec;
        std::filesystem::create_directories(mpath.parent_path(), ec);
        shared_metrics_out.open(mpath, std::ios::app);
    }
    {
        const auto mpath = request.analytics_path.empty()
            ? request.output_dir / "training_metrics.jsonl"
            : request.analytics_path;
        std::error_code ec;
        std::filesystem::create_directories(mpath.parent_path(), ec);
        burn_metrics_out.open(mpath, std::ios::trunc);
    }
    if (request.analytics_required &&
        (!shared_metrics_out.is_open() || !burn_metrics_out.is_open())) {
        throw std::runtime_error("analytics_required but native metrics streams could not be opened");
    }
#endif
    auto utc_now = []() {
        char buf[32];
        const std::time_t t = std::time(nullptr);
        std::tm tm_utc{};
#if defined(_WIN32)
        gmtime_s(&tm_utc, &t);
#else
        gmtime_r(&t, &tm_utc);
#endif
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
        return std::string(buf);
    };
    std::vector<float> loss_hist, lr_hist, gn_hist, global_clip_scale_hist;
    std::vector<float> embed_clip_max_preclip_hist, dominant_slot_norm_hist, dominant_slot_frac_hist;
    std::vector<int> global_clip_fired_hist, embed_clip_fired_hist, embed_clip_rows_hist;
    float loss_ema = -1.0f;
    std::size_t examples_seen = 0;
    const auto burn_wall_start = std::chrono::steady_clock::now();

    // MLPerf: full hyperparameter disclosure (Open Division), then run_start.
    {
        const int warmup = std::max(1, total_opt / 50);
        ml.point("seed", std::to_string(request.seed),
                 "{\"seat\": \"" + request.seat + "\", \"version\": \"" + request.version + "\"}");
        ml.point("global_batch_size", std::to_string(mb * accum_ceiling));
        ml.point("gradient_accumulation_steps", std::to_string(accum_ceiling));
        // opt_name and the hyperparameters beneath it must reflect the
        // ACTUAL optimizer this burn ran, not an assumed default -- this
        // MLPerf evidence file is a durable submission artifact, and a
        // Lion burn disclosing "adamw" + AdamW's beta/epsilon constants
        // (which Lion's own kernel never reads) would misrepresent exactly
        // the run it claims to describe.
        if (optimizer_uses_lion(request)) {
            ml.point("opt_name", MlLog::q("lion"));
            ml.point("opt_base_learning_rate", std::to_string(base_lr));
            ml.point("opt_end_learning_rate", std::to_string(base_lr * 0.1));
            ml.point("opt_learning_rate_warmup_steps", std::to_string(warmup));
            ml.point("opt_learning_rate_decay_schedule", MlLog::q("cosine"));
            ml.point("opt_lion_beta_1", std::to_string(lion_b1));
            ml.point("opt_lion_beta_2", std::to_string(lion_b2));
            ml.point("opt_lion_lr_scale", std::to_string(lion_lr_mult));
            ml.point("opt_lion_weight_decay_scale", std::to_string(lion_wd_mult));
            ml.point("opt_trust_ratio_enabled", MlLog::q(trust_ratio_enabled(request) ? "true" : "false"));
        }
        ml.point("opt_gradient_clip_norm", std::to_string(clip));
        ml.point("max_sequence_length", std::to_string(S));
        ml.point("train_samples", std::to_string(ds.num_sequences));
        ml.point("eval_samples", "0");
        // Open Division model disclosure (custom architecture)
        ml.point("model_architecture", MlLog::q("ida_lattice"));
        ml.point("model_hidden_size", std::to_string(w.hidden_size));
        ml.point("model_num_layers", std::to_string(w.num_layers));
        ml.point("model_num_heads", std::to_string(w.heads));
        ml.point("model_num_kv_heads", std::to_string(w.kv_heads > 0 ? w.kv_heads : w.heads));
        ml.point("model_intermediate_size", std::to_string(w.intermediate_size));
        ml.point("model_vocab_size", std::to_string(w.vocab_size));
        ml.point("precision", MlLog::q(runtime_precision_label(request)));
        ml.point("accumulation_curriculum",
                 MlLog::q("1x->2x@7.5%->4x@15% of exposure, ceiling " +
                          std::to_string(accum_ceiling)));
        ml.end("init_stop");
        write_native_timeline_event(request, "native_init", "end",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - native_init_t0).count());
        ml.begin("run_start");
        write_native_timeline_event(request, "native_train_loop", "start");
        ml.begin("block_start",
                 "{\"first_epoch_num\": 0, \"epoch_count\": 1}");
    }
    const auto native_train_loop_t0 = std::chrono::steady_clock::now();
    int ml_epoch = 0;

    cudaEvent_t ev_start{}, ev_stop{};
    IDA_CUDA_CHECK(cudaEventCreate(&ev_start));
    IDA_CUDA_CHECK(cudaEventCreate(&ev_stop));
    IDA_CUDA_CHECK(cudaEventRecord(ev_start, arena.stream));
    attn_bwd_health_reset(arena.stream);

    auto last_hb = std::chrono::steady_clock::now();
    constexpr long long kHbMs = 5000;

    std::size_t total_tokens = 0;
    float  last_loss      = 0.0f;
    float  last_grad_norm = 0.0f;
    double last_lr        = 0.0;
    bool   last_global_clip_fired = false;
    float  last_global_clip_scale = 1.0f;
    bool   last_embed_clip_fired = false;
    float  last_embed_clip_max_preclip_norm = 0.0f;
    int    last_embed_clipped_rows = 0;
    std::string last_dominant_grad_slot;
    float  last_dominant_grad_slot_norm = 0.0f;
    float  last_dominant_grad_slot_frac = 0.0f;
    // PSS predictor's own gradient norm (pred_down + pred_up, L2-combined),
    // reported unconditionally every step -- see the computation site for
    // why this exists (dominant_grad_slot alone silently discards it).
    float  last_pss_predictor_grad_norm = 0.0f;
    float  last_pss_predictor_grad_frac = 0.0f;
    // Decoupled-accumulation visibility (2026-08-14): without these, nothing
    // in the evidence stream can distinguish "this step accumulated into the
    // PSS window" from "this step applied the window average" -- the code
    // could be silently wrong (or silently right) with zero way to tell from
    // training_metrics.jsonl alone.
    int    last_pss_window_count = 0;
    bool   last_pss_window_applied = false;
    float  last_pss_predictor_momentum_norm = 0.0f;
    // Second/third-ranked gradient slots (2026-08-14): dominant_grad_slot
    // alone only ever shows the single winner every step -- any OTHER slot's
    // real gradient share (LRSS, LSS, a specific layer) is silently
    // discarded unless it happens to win outright. Same blind spot class as
    // the PSS-specific fix above, generalized: rank 2 and 3 close most of
    // the gap cheaply (top-3 covers "who's actually competing" without a
    // full per-step dump of every slot).
    std::string last_dominant_grad_slot_2, last_dominant_grad_slot_3;
    float  last_dominant_grad_slot_2_norm = 0.0f, last_dominant_grad_slot_2_frac = 0.0f;
    float  last_dominant_grad_slot_3_norm = 0.0f, last_dominant_grad_slot_3_frac = 0.0f;
    int    last_lss_feedback_skip_tail = 0;
    float  last_lss_feedback_residual_scale = 1.0f;
    float  last_lss_aux = 0.0f;
    float  last_lss_recon_norm = 0.0f;
    float  last_lss_target_norm = 0.0f;
    float  last_lss_relative_rmse = 0.0f;
    float  last_support_transition_scale = 1.0f;
    float  prev_lss_target_norm = 0.0f;
    float  prev_lss_relative_rmse = 0.0f;
    int    lss_rmse_non_improve_windows = 0;
    int    global_clip_fired_steps = 0;
    int    embed_clip_fired_steps = 0;
    int    embed_clipped_rows_total = 0;
    float  last_pss_spike_ratio_max = 0.0f;
    std::string last_pss_spike_slot;
    int    pss_slot_clip_fired_steps = 0;
    float  last_pss_confidence = 0.0f;
    float  last_pss_pred_err = 0.0f;
    float  last_pss_int2_agreement = 0.0f;
    std::uint32_t last_pss_covered = 0;
    std::size_t last_pss_n = 0;
    std::uint32_t last_pss_int2_matched = 0;
    std::uint32_t last_pss_int2_scored = 0;
    int last_pss_scored_micros = 0;
    float last_pss_int2_inv_rms = 0.0f;
    float last_pss_confidence_min = 0.0f;
    float last_pss_confidence_max = 0.0f;
    float last_pss_blend_delta_rms = 0.0f;
    float last_pss_aux_weight = 0.0f;
    float last_pss_aux_denom = 0.0f;
    bool last_pss_aux_normalize = false;
    bool last_pss_conditioning_active = false;
    bool last_pss_override_active = false;
    float last_pss_effective_confidence = 0.0f;
    std::string last_pss_governor_event{"none"};
    // Magnitude-bucketed err/hit-rate diagnostic -- see pss_mag_bucket.
    // err_ratio[b] mirrors pred_err's definition (err_sq/target_sq) but
    // scoped to bucket b only; hit_rate[b] mirrors confidence's definition
    // (covered/count) the same way. Lets pred_err's aggregate-improving vs
    // confidence's aggregate-declining trends be checked against the SAME
    // per-magnitude-decile data instead of only the two aggregate scalars.
    // Single-GPU path only (run_lattice_training_model_parallel is not
    // wired up here -- k_pss_pred_score is called from one shared site,
    // but only this function's evidence readback populates these).
    std::array<float, kPssMagBuckets> last_pss_mag_bucket_err_ratio{};
    std::array<float, kPssMagBuckets> last_pss_mag_bucket_hit_rate{};
    std::array<std::uint32_t, kPssMagBuckets> last_pss_mag_bucket_count{};
    // Exact resume (Phase 3): seeded from resume_state when
    // resume_from_checkpoint was set above, else both stay at the fresh-
    // genesis default of 0. Seeding micro_done alone is sufficient to fix
    // BOTH the curriculum ramp (accum_at(micro_done) below) and the loop's
    // own exit condition (micro_done < total_micro) -- no separate offset
    // parameter needed in accum_at() itself, since it already operates
    // directly on whatever micro_done is at call time.
    int    micro_done     = resume_state.cumulative_micro_steps;
    int    skipped        = 0;
    int    seq_cursor     = static_cast<int>(resume_state.dataset_cursor);

    // Per-slot GPU buffers for async D→D loss/clip-stats copies — avoids a
    // cudaStreamSynchronize after every micro-batch.  One sync per optimizer step.
    const int kMaxMicroAccum = std::max(1, accum_ceiling);
    float* d_micro_losses = nullptr;
    float* d_micro_clip_stats = nullptr;
    IDA_CUDA_CHECK(ida_malloc_async(&d_micro_losses,
        static_cast<std::size_t>(kMaxMicroAccum) * sizeof(float), arena.pool, arena.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&d_micro_clip_stats,
        static_cast<std::size_t>(kMaxMicroAccum) * 2 * sizeof(float), arena.pool, arena.stream));

    // Periodic async weight persistence: every N optimizer steps the full
    // weight body is staged into pinned DDR5 on a side stream and written to
    // model.safetensors by a background thread (tmp+rename, crash-atomic).
    // Default off (0) — end-of-burn save is unchanged; production burns set
    // this for crash-consistent durability at ~90 ms/save training cost.
    const int ckpt_every_opt_steps = [] {
        // Era 13 default: 100 (crash-consistent periodic saves, ~90 ms
        // visible each on the AI body).  =0 restores end-of-burn-only.
        const char* e = std::getenv("IDA_NATIVE_CKPT_EVERY_OPT_STEPS");
        return e ? std::max(0, std::atoi(e)) : 100;
    }();
    AsyncWeightSaver weight_saver;

    // Exact resume: seeded so the LR schedule (lr_at(opt_step, total_opt,
    // ...) below) continues from the right point in the warmup/cosine
    // curve -- ++opt_step at the top of the loop makes the first resumed
    // iteration exactly cumulative_opt_steps+1, matching a fresh run's
    // first iteration being exactly 1.
    int opt_step = resume_state.cumulative_opt_steps;
    while (micro_done < total_micro) {
        ++opt_step;
        if (w.lrss_enabled && w.lss_rank > 0) {
            const float lss_aux_weight_base = lrss_params.lss_aux_weight;
            lrss_params.lss_aux_weight = lss_aux_weight_base * last_support_transition_scale;
        }
        sb.support_transition_scale = last_support_transition_scale;
        for (auto& wk : xworkers) {
            wk.sb.support_transition_scale = last_support_transition_scale;
        }
        const auto step_wall_start = std::chrono::steady_clock::now();
        const int accum_now = accum_at(micro_done);
        sb.pss_scored_micros = 0;
        sb.pss_last_blend_frac = 0.0f;
        sb.pss_aux_weight_used = 0.0f;
        sb.pss_aux_normalize_used = false;
        sb.pss_conditioning_active = false;
        if (sb.pss_confidence_minmax) {
            IDA_CUDA_CHECK(cudaSetDevice(arena.device_id));
            k_pss_reset_confidence_minmax<<<1, 1, 0, arena.stream>>>(
                sb.pss_confidence_minmax);
        }
        for (auto& wk : xworkers) {
            wk.sb.pss_scored_micros = 0;
            wk.sb.pss_last_blend_frac = 0.0f;
            wk.sb.pss_aux_weight_used = 0.0f;
            wk.sb.pss_aux_normalize_used = false;
            wk.sb.pss_conditioning_active = false;
            if (wk.sb.pss_confidence_minmax) {
                IDA_CUDA_CHECK(cudaSetDevice(wk.arena.device_id));
                k_pss_reset_confidence_minmax<<<1, 1, 0, wk.arena.stream>>>(
                    wk.sb.pss_confidence_minmax);
            }
        }
        IDA_CUDA_CHECK(cudaSetDevice(arena.device_id));
        // ── Micro-batch loop: accumulate gradients ───────────────────────────
        float loss_sum = 0.0f;
        int   micros_this_step = 0;
        bool  step_embed_clip_fired = false;
        float step_embed_clip_max_preclip_norm = 0.0f;
        int   step_embed_clipped_rows = 0;
        // Crossover fence: workers must not read weights until the previous
        // window's optimizer step (and FP8 refresh + grad zeroing) on the
        // main stream is complete.
        if (!xworkers.empty()) {
            IDA_CUDA_CHECK(cudaEventRecord(ev_weights, arena.stream));
            for (auto& wk : xworkers)
                IDA_CUDA_CHECK(cudaStreamWaitEvent(wk.arena.stream, ev_weights, 0));
        }
        for (int a = 0; a < accum_now && micro_done < total_micro; ++a) {
            // Round-robin micro-batches over the crossover workers; index 0
            // is the main stream's own context.
            const int widx = xworkers.empty()
                ? 0 : a % (static_cast<int>(xworkers.size()) + 1);
            NativeArena&           xarena  = widx ? xworkers[widx-1].arena  : arena;
            StepBuffers&           xsb     = widx ? xworkers[widx-1].sb     : sb;
            Fp8Ctx&                xf8     = widx ? xworkers[widx-1].f8     : f8;
            PackedFp4AttentionCtx& xfp4    = widx ? xworkers[widx-1].fp4    : fp4_attn;
            cublasHandle_t         xcublas = widx ? xworkers[widx-1].cublas : cublas;
            LatticeGrads&          xg      = widx ? xworkers[widx-1].g      : g;
            uint32_t* xd_tokens = widx ? xworkers[widx-1].d_tokens : d_tokens;
            int32_t*  xd_labels = widx ? xworkers[widx-1].d_labels : d_labels;
            uint16_t* xd_segs   = widx ? xworkers[widx-1].d_segs   : d_segs;
            xsb.lss_feedback_completed_optimizer_steps = opt_step - 1;

            int batch_start = seq_cursor % ds.num_sequences;
            int actual_mb   = std::min(mb, ds.num_sequences - batch_start);
            if (actual_mb < mb && ds.num_sequences >= mb) {
                batch_start = 0; actual_mb = mb;
            }
            seq_cursor = (batch_start + actual_mb) % ds.num_sequences;

            const std::size_t off    = static_cast<std::size_t>(batch_start) * S;
            const std::size_t nelems = static_cast<std::size_t>(actual_mb) * S;

            IDA_CUDA_CHECK(cudaMemcpyAsync(xd_tokens, ds.tokens.data() + off,
                nelems * sizeof(uint32_t), cudaMemcpyHostToDevice, xarena.stream));
            IDA_CUDA_CHECK(cudaMemcpyAsync(xd_labels, ds.labels.data() + off,
                nelems * sizeof(int32_t), cudaMemcpyHostToDevice, xarena.stream));
            IDA_CUDA_CHECK(cudaMemcpyAsync(xd_segs, ds.segs.data() + off,
                nelems * sizeof(uint16_t), cudaMemcpyHostToDevice, xarena.stream));

            xsb.B = actual_mb;   // clamp: never touch uninitialized rows

            // LRSS ring choreography (K=1 guaranteed by the crossover guard):
            // freeze this step's bank view BEFORE the forward consumes it;
            // record this step's anchor AFTER the backward is enqueued — the
            // recompute-hazard rule from the contract doc.
            if (lrss_scr)
                lrss_refresh_bank(*lrss_scr, w.lrss_anchors, w.hidden_size,
                                  micro_done + 1, xarena.stream);

            forward(xcublas, attention_backend, xf8, xfp4, request, w, xd_tokens, xsb, xarena);
            backward_accumulate(xcublas, attention_backend, xf8, xfp4, request, w, xg,
                                xd_tokens, xd_labels, xsb, xarena,
                                pss_pred_aux_weight(request));

            if (lrss_scr)
                lrss_record_anchor(*lrss_scr, w.lrss_anchors, w.hidden_size,
                                   micro_done + 1, xarena.stream);
            last_lss_feedback_skip_tail = xsb.lss_feedback_skip_tail;
            last_lss_feedback_residual_scale =
                xsb.lss_feedback_skip_tail > 0 ? lss_feedback_residual_scale() : 1.0f;
            last_lss_aux =
                (xsb.lrss_s && xsb.lrss_s->lss_aux_valid)
                    ? xsb.lrss_s->lss_last_aux : 0.0f;

            // Async D→D: stage loss and clip stats into per-slot GPU buffers.
            // One cudaStreamSynchronize fires at the end of the inner loop,
            // not after every micro-batch.
            IDA_CUDA_CHECK(cudaMemcpyAsync(d_micro_losses + a, xsb.loss,
                sizeof(float), cudaMemcpyDeviceToDevice, xarena.stream));
            if (embed_clip_threshold > 0.0f) {
                IDA_CUDA_CHECK(cudaMemcpyAsync(d_micro_clip_stats + a * 2,
                    xsb.embed_clip_stats, 2 * sizeof(float),
                    cudaMemcpyDeviceToDevice, xarena.stream));
            }

            total_tokens  += nelems;
            examples_seen += actual_mb;
            ++micro_done;
            ++micros_this_step;

            // Intra-accumulation heartbeat — prevents supervisor stall when
            // large batch×accum makes the first optimizer step take >900 s
            // (e.g. AI: 64 micro-batches × ~15 s each = ~960 s).
            if (on_step) {
                auto _now = std::chrono::steady_clock::now();
                long long _ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    _now - last_hb).count();
                if (_ms >= kHbMs) {
                    cudaEvent_t _hb{};
                    IDA_CUDA_CHECK(cudaEventCreate(&_hb));
                    IDA_CUDA_CHECK(cudaEventRecord(_hb, arena.stream));
                    IDA_CUDA_CHECK(cudaEventSynchronize(_hb));
                    if (xsb.lrss_s && xsb.lrss_s->lss_aux_copy_pending) {
                        xsb.lrss_s->lss_aux_copy_pending = 0;
                        xsb.lrss_s->lss_aux_valid = 1;
                    }
                    if (xsb.lrss_s && xsb.lrss_s->lss_aux_valid)
                        last_lss_aux = xsb.lrss_s->lss_last_aux;
                    // Read back losses so far for heartbeat; don't touch loss_sum
                    // (batch readback happens after the inner loop).
                    const int _nm = std::min(micros_this_step, kMaxMicroAccum);
                    std::vector<float> _lbuf(_nm);
                    IDA_CUDA_CHECK(cudaMemcpy(_lbuf.data(), d_micro_losses,
                        static_cast<std::size_t>(_nm) * sizeof(float),
                        cudaMemcpyDeviceToHost));
                    float _hb_loss_sum = 0.0f;
                    for (int _i = 0; _i < _nm; ++_i) _hb_loss_sum += _lbuf[_i];
                    float _ms_gpu = 0.0f;
                    IDA_CUDA_CHECK(cudaEventElapsedTime(&_ms_gpu, ev_start, _hb));
                    IDA_CUDA_CHECK(cudaEventDestroy(_hb));
                    ProgressReport _rep{};
                    _rep.micro_step           = micro_done;
                    _rep.optimizer_step       = opt_step - 1;
                    _rep.active_grad_accum    = micros_this_step;
                    _rep.requested_grad_accum = accum_ceiling;
                    _rep.tokens               = total_tokens;
                    _rep.elapsed_s            = _ms_gpu / 1000.0;
                    _rep.loss                 = _hb_loss_sum / std::max(1, micros_this_step);
                    _rep.loss_ema             = loss_ema;
                    _rep.grad_norm            = last_grad_norm;
                    _rep.lr                   = static_cast<float>(last_lr);
                    _rep.effective_batch      = mb * std::max(1, micros_this_step);
                    _rep.skipped_steps        = skipped;
                    _rep.global_grad_clip_steps       = global_clip_fired_steps;
                    _rep.embed_row_clip_steps         = embed_clip_fired_steps;
                    _rep.embed_row_clipped_rows_total = embed_clipped_rows_total;
                    _rep.lss_feedback_skip_tail = last_lss_feedback_skip_tail;
                    _rep.lss_feedback_residual_scale = last_lss_feedback_residual_scale;
                    _rep.lss_aux = last_lss_aux;
                    _rep.lss_recon_norm = last_lss_recon_norm;
                    _rep.lss_target_norm = last_lss_target_norm;
                    _rep.lss_relative_rmse = last_lss_relative_rmse;
                    _rep.support_transition_scale = last_support_transition_scale;
                    // PSS telemetry: this intra-accumulation heartbeat fires every
                    // ~kHbMs throughout a single (potentially 600s+) accumulation
                    // window, far more often than the once-per-optimizer-step fence
                    // that computes these values. Without this, _rep's default
                    // construction zeroes them in the LIVE status file between real
                    // updates -- the JSONL (written only at the fence) and the final
                    // result.final_* summary were never affected, but real-time
                    // monitoring was seeing stale zeros for most of every window.
                    _rep.pss_slot_clip_steps = pss_slot_clip_fired_steps;
                    _rep.pss_spike_ratio_max = last_pss_spike_ratio_max;
                    _rep.pss_spike_slot = last_pss_spike_slot;
                    _rep.pss_confidence = last_pss_confidence;
                    _rep.pss_pred_err = last_pss_pred_err;
                    _rep.pss_int2_agreement = last_pss_int2_agreement;
                    _rep.pss_engaged_frac = sb.pss_engaged_frac;
                    _rep.pss_covered = last_pss_covered;
                    _rep.pss_n = last_pss_n;
                    _rep.pss_int2_matched = last_pss_int2_matched;
                    _rep.pss_int2_scored = last_pss_int2_scored;
                    _rep.pss_scored_micros = last_pss_scored_micros;
                    _rep.pss_int2_inv_rms = last_pss_int2_inv_rms;
                    _rep.pss_confidence_min = last_pss_confidence_min;
                    _rep.pss_confidence_max = last_pss_confidence_max;
                    _rep.pss_blend_delta_rms = last_pss_blend_delta_rms;
                    _rep.pss_aux_weight = last_pss_aux_weight;
                    _rep.pss_aux_denom = last_pss_aux_denom;
                    _rep.pss_aux_normalize = last_pss_aux_normalize;
                    _rep.pss_conditioning_active = last_pss_conditioning_active;
                    _rep.pss_override_active = last_pss_override_active;
                    _rep.pss_effective_confidence = last_pss_effective_confidence;
                    _rep.pss_governor_event = last_pss_governor_event;
                    _rep.pss_conditioning_mode = pss_conditioning_mode;
                    _rep.pss_aux_normalize_mode = pss_aux_normalize_mode;
                    _rep.fp8_active           = f8.on;
                    _rep.attention_backend    = attention_backend_name(attention_backend);
                    _rep.precision_profile    = request.precision_profile;
                    try { on_step(_rep); } catch (...) {}
                    last_hb = _now;
                }
            }
        }
        // Crossover join: main stream waits for every worker's window work,
        // then folds their gradient sets into the main one.  The norm/clip/
        // optimizer path below runs on the reduced grads unchanged.
        if (!xworkers.empty()) {
            for (auto& wk : xworkers) {
                IDA_CUDA_CHECK(cudaEventRecord(wk.ev_done, wk.arena.stream));
                IDA_CUDA_CHECK(cudaStreamWaitEvent(arena.stream, wk.ev_done, 0));
            }
            for (auto& wk : xworkers)
                accumulate_grads_into(g, wk.g, w, arena.stream);
        }
        // Batch D→H readback: one sync per optimizer step instead of one per micro-batch.
        IDA_CUDA_CHECK(cudaStreamSynchronize(arena.stream));
        if (sb.lrss_s && sb.lrss_s->lss_aux_copy_pending) {
            sb.lrss_s->lss_aux_copy_pending = 0;
            sb.lrss_s->lss_aux_valid = 1;
        }
        if (sb.lrss_s && sb.lrss_s->lss_aux_valid)
            last_lss_aux = sb.lrss_s->lss_last_aux;
        {
            const int _nm = std::min(micros_this_step, kMaxMicroAccum);
            std::vector<float> _lbuf(_nm);
            IDA_CUDA_CHECK(cudaMemcpy(_lbuf.data(), d_micro_losses,
                static_cast<std::size_t>(_nm) * sizeof(float), cudaMemcpyDeviceToHost));
            for (int _i = 0; _i < _nm; ++_i) loss_sum += _lbuf[_i];
            if (embed_clip_threshold > 0.0f) {
                std::vector<float> _cbuf(_nm * 2);
                IDA_CUDA_CHECK(cudaMemcpy(_cbuf.data(), d_micro_clip_stats,
                    static_cast<std::size_t>(_nm) * 2 * sizeof(float), cudaMemcpyDeviceToHost));
                for (int _i = 0; _i < _nm; ++_i) {
                    const int clipped_rows = static_cast<int>(_cbuf[_i * 2]);
                    if (clipped_rows > 0) {
                        step_embed_clip_fired = true;
                        step_embed_clipped_rows += clipped_rows;
                        step_embed_clip_max_preclip_norm =
                            std::max(step_embed_clip_max_preclip_norm, _cbuf[_i * 2 + 1]);
                    }
                }
            }
        }
        last_loss = loss_sum / std::max(1, micros_this_step);
        loss_ema  = (loss_ema < 0.0f) ? last_loss : 0.95f * loss_ema + 0.05f * last_loss;

        // ── Mean over accumulated micro-steps ────────────────────────────────
        if (micros_this_step > 1) {
            const float inv = 1.0f / static_cast<float>(micros_this_step);
            for (auto& p : slots)
                slot_scale_gradient(p, inv, arena.stream);
        }
        k_zeros_f32<<<ceildiv(slots.size(), 256), 256, 0, arena.stream>>>(d_slot_normsq, slots.size());
        for (std::size_t i = 0; i < slots.size(); ++i)
            slot_sq_sum_acc(slots[i], d_slot_normsq + i, arena.stream);

        // ── Global L2 gradient norm ──────────────────────────────────────────
        k_zeros_f32<<<1, 1, 0, arena.stream>>>(sb.norm_acc, 1);
        for (auto& p : slots)
            slot_sq_sum_acc(p, sb.norm_acc, arena.stream);
        float sq = 0.0f;
        IDA_CUDA_CHECK(cudaStreamSynchronize(arena.stream));
        IDA_CUDA_CHECK(cudaMemcpy(&sq, sb.norm_acc, sizeof(float), cudaMemcpyDeviceToHost));
        IDA_CUDA_CHECK(cudaMemcpy(
            h_slot_normsq.data(),
            d_slot_normsq,
            slots.size() * sizeof(float),
            cudaMemcpyDeviceToHost));
        // PSS Stage 2: read back the tail-layer predictor's shadow-scoring
        // accumulators from the last micro-step of this window, riding the
        // same host sync as the grad-norm readback above -- no new sync.
        // All PSS fence scalars share one contiguous device surface so the
        // fence needs one D2H read, not one blocking read per scalar.
        // Snapshot semantics (last micro-step, not window-averaged), same as
        // dominant_grad_slot below.
        if (sb.pss_fence_metrics) {
            PssFenceEvidence evidence{};
            IDA_CUDA_CHECK(cudaMemcpy(
                &evidence, sb.pss_fence_metrics, sizeof(evidence), cudaMemcpyDeviceToHost));
            const PssFenceMetrics& metrics = evidence.metrics;
            const float int2_inv_rms = evidence.int2_inv_rms;
            const float confidence_minmax[2] = {
                evidence.confidence_min, evidence.confidence_max};
            const float err_sq = metrics.err_sq;
            const float target_sq = metrics.target_sq;
            const unsigned int covered = metrics.covered;
            const unsigned int int2_matched = metrics.int2_matched;
            const unsigned int int2_scored = metrics.int2_scored;
            const std::size_t n = static_cast<std::size_t>(sb.B) * sb.S * sb.H;
            last_pss_covered = covered;
            last_pss_n = n;
            last_pss_int2_matched = int2_matched;
            last_pss_int2_scored = int2_scored;
            last_pss_scored_micros = sb.pss_scored_micros;
            last_pss_int2_inv_rms = int2_inv_rms;
            last_pss_confidence_min =
                confidence_minmax[0] <= confidence_minmax[1] ? confidence_minmax[0] : 0.0f;
            last_pss_confidence_max =
                confidence_minmax[0] <= confidence_minmax[1] ? confidence_minmax[1] : 0.0f;
            last_pss_confidence = (n > 0)
                ? (static_cast<float>(static_cast<double>(covered) / static_cast<double>(n)))
                : 0.0f;
            last_pss_pred_err = (target_sq > 1e-12f) ? (err_sq / target_sq) : 0.0f;
            last_pss_int2_agreement = int2_scored > 0
                ? static_cast<float>(static_cast<double>(int2_matched) /
                                     static_cast<double>(int2_scored))
                : 0.0f;
            last_pss_blend_delta_rms =
                (sb.pss_last_blend_frac > 0.0f && n > 0 && std::isfinite(err_sq))
                    ? sb.pss_last_blend_frac *
                          std::sqrt(std::max(0.0f, err_sq / static_cast<float>(n)))
                    : 0.0f;
            last_pss_aux_weight = sb.pss_aux_weight_used;
            last_pss_aux_normalize = sb.pss_aux_normalize_used;
            last_pss_aux_denom =
                (last_pss_aux_normalize && std::isfinite(target_sq) && target_sq > 1e-12f)
                    ? target_sq : static_cast<float>(n);
            last_pss_conditioning_active = sb.pss_conditioning_active;
            last_pss_effective_confidence = last_pss_confidence;
            last_pss_override_active = false;
            last_pss_governor_event = "none";
            if (sb.pss_mag_bucket_err_sq) {
                std::array<float, kPssMagBuckets> h_err_sq{}, h_target_sq{};
                std::array<std::uint32_t, kPssMagBuckets> h_covered{};
                IDA_CUDA_CHECK(cudaMemcpy(h_err_sq.data(), sb.pss_mag_bucket_err_sq,
                    kPssMagBuckets * sizeof(float), cudaMemcpyDeviceToHost));
                IDA_CUDA_CHECK(cudaMemcpy(h_target_sq.data(), sb.pss_mag_bucket_target_sq,
                    kPssMagBuckets * sizeof(float), cudaMemcpyDeviceToHost));
                IDA_CUDA_CHECK(cudaMemcpy(h_covered.data(), sb.pss_mag_bucket_covered,
                    kPssMagBuckets * sizeof(float), cudaMemcpyDeviceToHost));
                IDA_CUDA_CHECK(cudaMemcpy(last_pss_mag_bucket_count.data(), sb.pss_mag_bucket_count,
                    kPssMagBuckets * sizeof(float), cudaMemcpyDeviceToHost));
                for (int b = 0; b < kPssMagBuckets; ++b) {
                    last_pss_mag_bucket_err_ratio[b] =
                        (h_target_sq[b] > 1e-12f) ? (h_err_sq[b] / h_target_sq[b]) : 0.0f;
                    last_pss_mag_bucket_hit_rate[b] =
                        (last_pss_mag_bucket_count[b] > 0)
                            ? static_cast<float>(static_cast<double>(h_covered[b]) /
                                                  static_cast<double>(last_pss_mag_bucket_count[b]))
                            : 0.0f;
                }
            }
        }
        // PSS Stage 4: governor state machine. Ramps engagement up by
        // pss_engage_step() per window once confidence sits at or above
        // pss_engage_in(), capped at pss_engage_max(); snaps to 0 the
        // instant confidence drops below pss_engage_out() (no ramp-down --
        // pull-out is immediate, nudge-in is gradual, the asymmetry is
        // deliberate). Between the two thresholds is a dead zone: hold
        // whatever the current fraction is (hysteresis -- this is what
        // prevents chatter at a single boundary). Requires
        // pss_engage_min_optimizer_steps() windows of history first, same
        // gate shape as the LSS feedback controller uses.
        if (pss_governor_enabled() && sb.pss_pred_hidden) {
            float conf = last_pss_confidence;
            float override_conf = 0.0f;
            last_pss_override_active = pss_confidence_override(&override_conf);
            if (last_pss_override_active) conf = override_conf;
            last_pss_effective_confidence = conf;
            if (opt_step >= pss_engage_min_optimizer_steps()) {
                const float previous_frac = sb.pss_engaged_frac;
                if (conf >= pss_engage_in()) {
                    sb.pss_engaged_frac = std::min(
                        sb.pss_engaged_frac + pss_engage_step(), pss_engage_max());
                    if (sb.pss_engaged_frac > previous_frac)
                        last_pss_governor_event = "engage";
                } else if (conf < pss_engage_out()) {
                    sb.pss_engaged_frac = 0.0f;
                    if (previous_frac > 0.0f)
                        last_pss_governor_event = "disengage";
                }
            }
            if (last_pss_governor_event != "none") {
                std::fprintf(stderr,
                    "[pss-governor] opt_step=%d event=%s confidence=%.6g "
                    "override=%s engaged_frac=%.6g\n",
                    opt_step, last_pss_governor_event.c_str(),
                    last_pss_effective_confidence,
                    last_pss_override_active ? "true" : "false",
                    sb.pss_engaged_frac);
            }
        }
        float grad_norm = sqrtf(sq);
        last_dominant_grad_slot.clear();
        last_dominant_grad_slot_norm = 0.0f;
        last_dominant_grad_slot_frac = 0.0f;
        double dominant_slot_sq = 0.0;
        std::size_t dominant_slot_idx = 0;
        for (std::size_t i = 0; i < h_slot_normsq.size(); ++i) {
            if (h_slot_normsq[i] > dominant_slot_sq) {
                dominant_slot_sq = h_slot_normsq[i];
                dominant_slot_idx = i;
            }
        }
        if (!slot_names.empty() && dominant_slot_idx < slot_names.size()) {
            last_dominant_grad_slot = slot_names[dominant_slot_idx];
            last_dominant_grad_slot_norm = std::sqrt(dominant_slot_sq);
            last_dominant_grad_slot_frac =
                (sq > 0.0f) ? static_cast<float>(dominant_slot_sq / static_cast<double>(sq)) : 0.0f;
        }
        // Rank 2 and 3: same mark-and-exclude scan, run twice more, so a
        // non-dominant slot's real share is visible instead of silently
        // discarded (see StepBuffers comment for why this matters).
        last_dominant_grad_slot_2.clear(); last_dominant_grad_slot_3.clear();
        last_dominant_grad_slot_2_norm = last_dominant_grad_slot_2_frac = 0.0f;
        last_dominant_grad_slot_3_norm = last_dominant_grad_slot_3_frac = 0.0f;
        {
            std::size_t excluded[2] = {dominant_slot_idx, dominant_slot_idx};
            for (int rank = 2; rank <= 3; ++rank) {
                double rank_sq = 0.0;
                std::size_t rank_idx = 0;
                bool found = false;
                for (std::size_t i = 0; i < h_slot_normsq.size(); ++i) {
                    if (i == excluded[0] || (rank == 3 && i == excluded[1])) continue;
                    if (!found || h_slot_normsq[i] > rank_sq) {
                        rank_sq = h_slot_normsq[i]; rank_idx = i; found = true;
                    }
                }
                if (found && rank_idx < slot_names.size()) {
                    const float norm = std::sqrt(rank_sq);
                    const float frac = (sq > 0.0f) ? static_cast<float>(rank_sq / static_cast<double>(sq)) : 0.0f;
                    if (rank == 2) {
                        last_dominant_grad_slot_2 = slot_names[rank_idx];
                        last_dominant_grad_slot_2_norm = norm;
                        last_dominant_grad_slot_2_frac = frac;
                        excluded[1] = rank_idx;
                    } else {
                        last_dominant_grad_slot_3 = slot_names[rank_idx];
                        last_dominant_grad_slot_3_norm = norm;
                        last_dominant_grad_slot_3_frac = frac;
                    }
                }
            }
        }
        // PSS predictor's own gradient norm, reported EVERY step regardless
        // of whether it happens to be the dominant slot. Before this, the
        // only per-slot signal that ever reached the evidence stream was
        // whichever slot won the max() above -- "embed" every single step
        // in every run seen so far -- so the predictor's actual gradient
        // magnitude was silently discarded on every step it didn't win,
        // i.e. always. That made "is PSS confidence flat because the
        // predictor isn't learning, or for some other reason" unanswerable
        // from the evidence alone. L2-combine pred_down + pred_up into one
        // number (both are the same logical predictor); if this comes out
        // non-trivial and moving, the predictor IS receiving real gradient
        // and confidence's own flatness needs a different explanation.
        {
            double pss_pred_sq = 0.0;
            for (std::size_t i = 0; i < slot_names.size() && i < h_slot_normsq.size(); ++i) {
                if (slot_names[i] == "pss.pred_down" || slot_names[i] == "pss.pred_up")
                    pss_pred_sq += h_slot_normsq[i];
            }
            last_pss_predictor_grad_norm = std::sqrt(pss_pred_sq);
            last_pss_predictor_grad_frac =
                (sq > 0.0f) ? static_cast<float>(pss_pred_sq / static_cast<double>(sq)) : 0.0f;
        }
        // PSS predictor's Lion MOMENTUM norm, not just its gradient. A
        // predictor can have healthy incoming gradient (above) while its
        // momentum stays dead/saturated -- gradient alone can't distinguish
        // "receiving signal" from "actually moving". OptStateTensor holds
        // either f32 or bf16 (never both), matching the same dispatch every
        // other momentum read in this file already uses.
        if (w.pss_pred_rank > 0) {
            const std::size_t pss_n = static_cast<std::size_t>(w.hidden_size) * w.pss_pred_rank;
            IDA_CUDA_CHECK(cudaMemsetAsync(sb.norm_acc, 0, sizeof(float), arena.stream));
            if (opt.m_pss_pred_dn.bf16) {
                sq_sum_acc_bf16(opt.m_pss_pred_dn.bf16, pss_n, sb.norm_acc, arena.stream);
                sq_sum_acc_bf16(opt.m_pss_pred_up.bf16, pss_n, sb.norm_acc, arena.stream);
            } else if (opt.m_pss_pred_dn.f32) {
                sq_sum_acc_f32(opt.m_pss_pred_dn.f32, pss_n, sb.norm_acc, arena.stream);
                sq_sum_acc_f32(opt.m_pss_pred_up.f32, pss_n, sb.norm_acc, arena.stream);
            }
            float mom_sq = 0.0f;
            IDA_CUDA_CHECK(cudaStreamSynchronize(arena.stream));
            IDA_CUDA_CHECK(cudaMemcpy(&mom_sq, sb.norm_acc, sizeof(float), cudaMemcpyDeviceToHost));
            last_pss_predictor_momentum_norm = std::sqrt(std::max(0.0f, mom_sq));
        }

        // ── PSS Stage 1a: per-slot spike detection + adaptive clip ──────────
        // Runs on the same host-side per-slot norms read back above. Tracks
        // an EMA per slot; a slot whose current norm exceeds
        // pss_slot_clip_mult * EMA is scaled down to that ceiling (when
        // enabled), and sq/grad_norm are corrected in place so the global
        // clip decision below sees the post-adaptive-clip norm. The EMA
        // itself always updates (even when the flag is off) so a later
        // enable has a warm baseline instead of a cold one.
        {
            float spike_ratio_max = 0.0f;
            std::string spike_slot;
            const float trust_lo = trust_ratio_lo(request), trust_hi = trust_ratio_hi(request);
            // IDA_NATIVE_TRUST_LR_NORM (2026-07-19, default off): normalize the
            // slot norms feeding the ratio/EMA by the LR schedule. Under Lion
            // the equilibrium gradient norm scales ~1/lr (measured r=-0.94
            // within production burns), so a trailing EMA of RAW slot norm can
            // never track the schedule-driven rise -- current/EMA sits
            // chronically >1 (production spike_ratio_max means: edge 4.3 /
            // ai 5.9 / swift 2.2) and trust = clamp(1/ratio) damps the
            // mass-carrying slots to ~0.2-0.25 for most of every burn.
            // Bisect evidence (probe_ledger trust_ratio_edge_loss_cost_
            // 20260719): that chronic damping cost Edge ~0.29 nats at 8
            // steps. slot_norm * (lr_now/base_lr) is approximately
            // schedule-invariant, so the EMA-relative ratio goes back to
            // detecting TRUE anomalies -- deviation from the slot's own
            // schedule-corrected baseline -- which is the original design
            // intent of both trust v2 and the Stage-1a spike signal.
            static const bool trust_lr_norm = [] {
                const char* e = std::getenv("IDA_NATIVE_TRUST_LR_NORM");
                return e && e[0] == '1';
            }();
            const float lr_norm_mult = trust_lr_norm
                ? static_cast<float>(lr_at(opt_step, total_opt, base_lr) / base_lr)
                : 1.0f;
            for (std::size_t i = 0; i < h_slot_normsq.size(); ++i) {
                const float raw_norm = std::sqrt(std::max(0.0f, h_slot_normsq[i]));
                const float slot_norm = raw_norm * lr_norm_mult;
                const float ema = pss_slot_norm_ema[i];
                if (ema > kPssSlotNormFloor) {
                    const float ratio = slot_norm / ema;
                    if (ratio > spike_ratio_max) {
                        spike_ratio_max = ratio;
                        spike_slot = (i < slot_names.size()) ? slot_names[i] : "";
                    }
                    // Optimizer Stage 3 v2 (2026-07-16 redesign, Lion only):
                    // trust_i = clamp(ema_i / (current_i + eps), lo, hi) --
                    // i.e. clamp(1/ratio, lo, hi). Compares this slot's
                    // CURRENT norm against ITS OWN EMA history, reusing the
                    // ratio PSS Stage 1a already computes here -- free, no
                    // new kernel pass. Normal slot (ratio~1) -> trust~1, no
                    // damping. Spiking slot (ratio>>1) -> trust<<1, damped
                    // for exactly this slot, this step. v1 (||w_i||/sqrt(n_i),
                    // see probe_ledger.jsonl optimizer_stage3_trust_ratio_
                    // CORRECTION) was an architectural constant that could
                    // not discriminate a spiking slot from a normal one --
                    // this EMA-relative form is slot-selective by
                    // construction, unlike v1.
                    if (use_lion && trust_ratio_enabled(request)) {
                        slot_trust_ratio[i] = std::min(trust_hi, std::max(trust_lo, 1.0f / ratio));
                    }
                    if (pss_slot_clip_enabled && ratio > pss_slot_clip_mult &&
                        slot_norm > kPssSlotNormFloor) {
                        // scale is a dimensionless ratio of two same-scale
                        // quantities (both LR-normalized when the flag is on),
                        // but sq and h_slot_normsq stay in RAW norm units --
                        // convert via raw_norm, never mix scales.
                        const float scale = (pss_slot_clip_mult * ema) / slot_norm;
                        const float raw_target = raw_norm * scale;
                        slot_scale_gradient(slots[i], scale, arena.stream);
                        sq += raw_target * raw_target - raw_norm * raw_norm;
                        h_slot_normsq[i] = raw_target * raw_target;
                        ++pss_slot_clip_fired_steps;
                    }
                }
                // EMA tracks the same (possibly LR-normalized) scale as the
                // ratio numerator, or the comparison is meaningless.
                pss_slot_norm_ema[i] = pss_spike_ema_beta * ema +
                    (1.0f - pss_spike_ema_beta) *
                        std::sqrt(std::max(0.0f, h_slot_normsq[i])) * lr_norm_mult;
            }
            last_pss_spike_ratio_max = spike_ratio_max;
            last_pss_spike_slot = spike_slot;
            grad_norm = std::sqrt(std::max(0.0f, sq));
        }
        last_grad_norm = grad_norm;

        // ── PSS Stage 1b: spike-ratio bucket vector → LSS conditioning ──────
        // Condenses the per-slot ratios above (already computed, no extra
        // pass over device memory) into kPssSpikeBuckets coarse buckets and
        // stages them into sb.lrss_s->pss_spike for the LSS joint input. This
        // publishes at most once per optimizer step -- the same one-sync
        // budget as the dominant-slot readback -- and micro-steps within the
        // next window read the same (one-step-stale) vector, matching the
        // anchor ring's existing lag pattern. Only runs when the checkpoint
        // was actually allocated with the wider joint (pss_spike_joint_dim >
        // 0); otherwise sb.lrss_s->pss_spike is null and this is a no-op.
        if (sb.lrss_s && sb.lrss_s->pss_spike) {
            auto ends_with = [](const std::string& s, const char* suf) {
                const std::size_t n = std::strlen(suf);
                return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
            };
            auto starts_with = [](const std::string& s, const char* pre) {
                const std::size_t n = std::strlen(pre);
                return s.size() >= n && s.compare(0, n, pre) == 0;
            };
            // Bucket order must match docs/predictive-supersampling-int2.md:
            // embed, lm_head, norms, attn-qk, attn-v/o, ffn, lrss-bank, lss-head.
            auto bucket_for = [&](const std::string& name) -> int {
                if (name == "embed") return 0;
                if (name == "lm_head") return 1;
                if (name == "final_norm" || ends_with(name, "attn_norm") ||
                    ends_with(name, "ffn_norm")) return 2;
                if (name == "lrss.query" || name == "lrss.key" ||
                    ends_with(name, "q_proj") || ends_with(name, "k_proj")) return 3;
                if (ends_with(name, "v_proj") || ends_with(name, "o_proj")) return 4;
                if (ends_with(name, "gate_proj") || ends_with(name, "up_proj") ||
                    ends_with(name, "down_proj")) return 5;
                if (starts_with(name, "lss.")) return 7;
                if (starts_with(name, "lrss.")) return 6;
                return -1;
            };
            float bucket_ratio[kPssSpikeBuckets] = {0.0f};
            for (std::size_t i = 0; i < h_slot_normsq.size() && i < slot_names.size(); ++i) {
                const int b = bucket_for(slot_names[i]);
                if (b < 0) continue;
                const float slot_norm = std::sqrt(std::max(0.0f, h_slot_normsq[i]));
                const float ema = pss_slot_norm_ema[i];
                const float ratio = (ema > kPssSlotNormFloor) ? (slot_norm / ema) : 0.0f;
                bucket_ratio[b] = std::max(bucket_ratio[b], ratio);
            }
            IDA_CUDA_CHECK(cudaMemcpyAsync(
                sb.lrss_s->pss_spike, bucket_ratio,
                sizeof(bucket_ratio), cudaMemcpyHostToDevice, arena.stream));
            sb.lrss_s->pss_spike_valid = 1;
        }

        // Support-transition controller. Restored 2026-07-31: 0a4ebd28
        // replaced this block with the rowclip/fp8-clip/slot-trace
        // instrumentation below rather than adding alongside it, silently
        // dropping the whole feature (IDA_NATIVE_SUPPORT_TRANSITION, 7 refs,
        // and support_transition_scale, 17 refs, went to ZERO) while the flag
        // stayed set in every probe and production launch shell. The two are
        // independent per-optimizer-step blocks; both belong here.
        if (support_transition_enabled() && w.lss_rank > 0) {
            const float prev_target = prev_lss_target_norm;
            const float target_growth = (prev_target > 1.0e-12f && last_lss_target_norm > 0.0f)
                ? (last_lss_target_norm / prev_target) : 1.0f;
            const bool rmse_improving =
                prev_lss_relative_rmse > 0.0f &&
                last_lss_relative_rmse < prev_lss_relative_rmse;
            if (prev_lss_relative_rmse > 0.0f && last_lss_relative_rmse > 0.0f) {
                lss_rmse_non_improve_windows = rmse_improving
                    ? 0 : lss_rmse_non_improve_windows + 1;
            }
            const bool support_dominant =
                (last_dominant_grad_slot.rfind("pss.", 0) == 0 ||
                 last_dominant_grad_slot.rfind("lss.", 0) == 0 ||
                 last_dominant_grad_slot.rfind("lrss.", 0) == 0) &&
                last_dominant_grad_slot_frac > support_transition_dominant_frac_max();
            const bool growth_pressure =
                prev_target > 0.0f &&
                target_growth > support_transition_target_growth_max() &&
                lss_rmse_non_improve_windows >= 2;
            const bool recover_ready =
                prev_target > 0.0f &&
                target_growth < support_transition_target_growth_max() &&
                (last_lss_relative_rmse <= support_transition_recover_rmse() || rmse_improving) &&
                !support_dominant;
            if (growth_pressure || support_dominant) {
                last_support_transition_scale = std::max(
                    support_transition_floor(),
                    last_support_transition_scale * support_transition_step_down());
            } else if (recover_ready) {
                last_support_transition_scale = std::min(
                    1.0f,
                    last_support_transition_scale * support_transition_step_up());
            }
            // Guarded, exactly as in bf381aeb -- an unguarded assignment would
            // let a zero/invalid reading poison prev_* and corrupt the growth
            // comparison on the following step.
            if (last_lss_target_norm > 0.0f) {
                prev_lss_target_norm = last_lss_target_norm;
            }
            if (last_lss_relative_rmse > 0.0f) {
                prev_lss_relative_rmse = last_lss_relative_rmse;
            }
        }

        if (rowclip_debug_enabled() && g_rowclip_stats != nullptr) {
            float rs[16] = {0.0f};
            IDA_CUDA_CHECK(cudaStreamSynchronize(arena.stream));
            IDA_CUDA_CHECK(cudaMemcpy(rs, g_rowclip_stats, 16 * sizeof(float),
                                      cudaMemcpyDeviceToHost));
            static const char* kSiteName[4] =
                {"act_row", "interlayer", "dqkv_row", "qk_row"};
            for (int si = 0; si < 4; ++si) {
                const float* r = rs + si * 4;
                if (r[1] <= 0.0f) continue;
                std::fprintf(stderr,
                    "[rowclip] opt_step=%d site=%s clipped_rows=%.0f total_rows=%.0f "
                    "frac=%.6f max_row_norm=%g ceiling=%g headroom=%.3f\n",
                    opt_step, kSiteName[si], r[0], r[1], r[0] / r[1], r[2], r[3],
                    (r[3] > 0.0f) ? (r[2] / r[3]) : 0.0f);
            }
            IDA_CUDA_CHECK(cudaMemsetAsync(g_rowclip_stats, 0, 16 * sizeof(float),
                                           arena.stream));
        }

        // FP8 delayed-scaling saturation report. Runs EVERY optimizer step
        // (not only on spikes) because silent clamping is precisely what does
        // NOT announce itself in grad_norm — __NV_SATFINITE conversion yields
        // no inf/nan, so a run can clip heavily while every existing counter,
        // including nonfinite, stays clean.
        if (fp8_clip_debug_enabled() && g_fp8_clip_stats != nullptr) {
            float cs[3] = {0.0f, 0.0f, 0.0f};
            IDA_CUDA_CHECK(cudaStreamSynchronize(arena.stream));
            IDA_CUDA_CHECK(cudaMemcpy(cs, g_fp8_clip_stats, 3 * sizeof(float),
                                      cudaMemcpyDeviceToHost));
            std::fprintf(stderr,
                "[fp8-clip] FP8 clip probe REPORT opt_step=%d clipped_calls=%.0f "
                "total_calls=%.0f clipped_frac=%.6f max_overflow_ratio=%.4f\n",
                opt_step, cs[0], cs[2],
                (cs[2] > 0.0f) ? (cs[0] / cs[2]) : 0.0f, cs[1]);
            if (g_fp8_elem_stats != nullptr) {
                float es[2] = {0.0f, 0.0f};
                IDA_CUDA_CHECK(cudaMemcpy(es, g_fp8_elem_stats, 2 * sizeof(float),
                                          cudaMemcpyDeviceToHost));
                std::fprintf(stderr,
                    "[fp8-clip-elem] opt_step=%d clipped_elems=%.0f total_elems=%.0f "
                    "elem_clipped_frac=%.8f\n",
                    opt_step, es[0], es[1], (es[1] > 0.0f) ? (es[0] / es[1]) : 0.0f);
                IDA_CUDA_CHECK(cudaMemsetAsync(g_fp8_elem_stats, 0, 2 * sizeof(float),
                                               arena.stream));
            }
            IDA_CUDA_CHECK(cudaMemsetAsync(g_fp8_clip_stats, 0, 3 * sizeof(float),
                                           arena.stream));
        }

        // Unconditional per-slot trace: WEIGHT norm and GRAD norm for every
        // slot, every slot_trace_every optimizer steps, regardless of health.
        // The dump below only fires when grad_norm > 1e6, so healthy runs are
        // invisible at slot granularity -- which is exactly the baseline you
        // need to recognise abnormality. Weight norms answer whether observed
        // activation growth is driven by the parameters themselves.
        if (slot_trace_enabled() && (opt_step == 1 || opt_step % slot_trace_every() == 0)) {
            k_zeros_f32<<<ceildiv(slots.size(), 256), 256, 0, arena.stream>>>(
                d_slot_wnormsq, slots.size());
            for (std::size_t i = 0; i < slots.size(); ++i)
                sq_sum_acc_bf16(slots[i].w, slots[i].n, d_slot_wnormsq + i, arena.stream);
            IDA_CUDA_CHECK(cudaMemcpyAsync(h_slot_wnormsq.data(), d_slot_wnormsq,
                                           slots.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost, arena.stream));
            IDA_CUDA_CHECK(cudaStreamSynchronize(arena.stream));
            for (std::size_t i = 0; i < slots.size() && i < slot_names.size(); ++i) {
                const float wn = std::sqrt(std::max(0.0f, h_slot_wnormsq[i]));
                const float gn = std::sqrt(std::max(0.0f, h_slot_normsq[i]));
                std::fprintf(stderr,
                    "[slot-trace] opt_step=%d slot=%s w_norm=%.6g g_norm=%.6g "
                    "g_over_w=%.6g n=%zu\n",
                    opt_step, slot_names[i].c_str(), wn, gn,
                    (wn > 0.0f) ? (gn / wn) : 0.0f, slots[i].n);
            }
        }

        // Per-component breakdown when the global (post-accum-average,
        // pre-clip) norm spikes — localizes which weight tensor's gradient
        // is actually exploding instead of only seeing the aggregate.
        if (gradnorm_debug_enabled() && grad_norm > 1e6f) {
            std::fprintf(stderr,
                "[gradnorm-debug] opt_step=%d global_grad_norm=%g SPIKE, per-slot breakdown:\n",
                opt_step, grad_norm);
            for (std::size_t i = 0; i < slots.size() && i < slot_names.size(); ++i) {
                const float slot_norm = std::sqrt(std::max(0.0f, h_slot_normsq[i]));
                if (slot_norm > 1.0f) {
                    std::fprintf(stderr, "  %-20s norm=%g\n", slot_names[i].c_str(), slot_norm);
                }
                if (i == 0 && slot_norm > 1.0f) {
                    // embed: break down further by vocab row.
                    debug_print_embed_row_breakdown(slots[i].g, w.vocab_size, w.hidden_size, arena.stream);
                }
            }
        }

        // LR-accumulation coupling (linear-scaling rule).  base_lr is tuned
        // for the CEILING effective batch (mb × accum_ceiling); a window that
        // accumulated fewer micro-batches takes a proportionally smaller
        // step.  Root-caused 2026-07-11 from the curriculum-ramp A/B: at
        // accum=1 the uncoupled LR is a 64×-oversized per-sample step —
        // healthy loss 10.5→3.0 in six steps, then geometric divergence
        // (~2.2×/step, embed → L0.v_proj) that the global clip converts
        // into pure clip-noise updates.  Also closes the epoch-seam kick:
        // a partial final window (44/64 micros) now takes 44/64 LR.
        // IDA_NATIVE_LR_ACCUM_COUPLING=0 restores the uncoupled schedule.
        const bool lr_accum_coupling = [] {
            const char* e = std::getenv("IDA_NATIVE_LR_ACCUM_COUPLING");
            return !e || e[0] == '1';
        }();
        double lr_t = lr_at(opt_step, total_opt, base_lr);
        if (lr_accum_coupling && accum_ceiling > 0) {
            lr_t *= static_cast<double>(micros_this_step) /
                    std::max(1, accum_ceiling);
        }
        last_lr = lr_t;
        last_embed_clip_fired = step_embed_clip_fired;
        last_embed_clip_max_preclip_norm = step_embed_clip_max_preclip_norm;
        last_embed_clipped_rows = step_embed_clipped_rows;
        if (step_embed_clip_fired) {
            ++embed_clip_fired_steps;
            embed_clipped_rows_total += step_embed_clipped_rows;
        }
        last_global_clip_fired = false;
        last_global_clip_scale = 1.0f;

        // #44: per-expert dW norms, paired with [expert-util]'s per-expert
        // routing share. An expert with ~0 share and a real norm here means
        // gradient is reaching an expert that never routed -- which would make
        // DENSE the buggy side and selection the fix, not the other way round.
        if (expert_grad_util_enabled()) {
            const int E_report = request.model.num_personality_experts;
            for (std::size_t i = 0; i < slots.size() && i < slot_names.size(); ++i) {
                const std::string& nm = slot_names[i];
                const bool is_expert_slot =
                    // Guard covers BOTH suffix compares below, so it must use
                    // the longer one's length (17, "moe.expert_fc_out") or
                    // that compare underflows nm.size()-17 for a 16-char nm
                    // that fails the first (short-circuited) compare --
                    // std::string::compare throws out_of_range rather than
                    // corrupting memory, but it's still a live bug, not a
                    // theoretical one. Latent: no real parameter name is
                    // exactly 16 chars long here (they all carry a layer/
                    // module prefix), so this has never actually thrown.
                    nm.size() >= 17 &&
                    (nm.compare(nm.size() - 16, 16, "moe.expert_fc_in") == 0 ||
                     nm.compare(nm.size() - 17, 17, "moe.expert_fc_out") == 0);
                if (!is_expert_slot) continue;
                moe_expert_grad_report(nm.c_str(), slots[i].g,
                                       static_cast<long long>(slots[i].n),
                                       E_report, opt_step, arena.stream);
            }
        }

        bool step_update_skipped = false;
        if (!std::isfinite(grad_norm) || grad_norm > grad_norm_abs_ceiling) {
            // Circuit breaker: skip the update, discard this step's gradients.
            // Absolute-ceiling branch (not just non-finite) — see the
            // grad_norm_abs_ceiling comment above for why "finite but
            // astronomical" is just as destructive as inf/nan here.
            ++skipped;
            step_update_skipped = true;
            if (gradnorm_debug_enabled() && std::isfinite(grad_norm)) {
                std::fprintf(stderr,
                    "[gradnorm-debug] opt_step=%d global_grad_norm=%g exceeds "
                    "absolute ceiling %g — SKIPPING (not clipping)\n",
                    opt_step, grad_norm, grad_norm_abs_ceiling);
            }
        } else {
            if (grad_norm > clip) {
                last_global_clip_fired = true;
                const float scale = clip / grad_norm;
                last_global_clip_scale = scale;
                ++global_clip_fired_steps;
                for (auto& p : slots)
                    slot_scale_gradient(p, scale, arena.stream);
            }

            // Decoupled PSS accumulation window (see StepBuffers::pss_window_
            // accum_down/up for the full design comment). Deliberately here:
            // after the mean-over-micro-steps AND the global clip above, so
            // the accumulator receives the exact same fully-processed
            // gradient the trunk itself would have applied this window --
            // never a pre-clip value. Lives inside this "not skipped" branch
            // on purpose: a discarded (skipped) window contributes nothing
            // to any slot's optimizer state, and PSS's cross-window
            // accumulation should honor that same discard, not silently
            // fold in a step the trunk itself threw away.
            const bool pss_at_boundary = (w.pss_pred_rank > 0) &&
                (++sb.pss_window_count >= pss_accum_steps());
            last_pss_window_count = sb.pss_window_count;
            last_pss_window_applied = pss_at_boundary;
            if (w.pss_pred_rank > 0) {
                const std::size_t pss_n = static_cast<std::size_t>(w.hidden_size) * w.pss_pred_rank;
                k_accumulate_f32<<<ceildiv(pss_n, 256), 256, 0, arena.stream>>>(
                    sb.pss_window_accum_down, g.g_pss_pred_down, pss_n);
                k_accumulate_f32<<<ceildiv(pss_n, 256), 256, 0, arena.stream>>>(
                    sb.pss_window_accum_up, g.g_pss_pred_up, pss_n);
            }

            for (std::size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
                auto& p = slots[slot_idx];
                // Isolation gate only: retain LRSS forward/backward and its
                // contribution to global norm/clip, but suppress support-head
                // optimizer mutation to locate the first-window transition.
                if (lrss_freeze && is_lrss_weight(p)) continue;
                const unsigned support_group = lrss_weight_group(p);
                if (support_group != 0u && (lrss_update_mask & support_group) == 0u)
                    continue;
                // PSS decoupled cadence: skip this slot entirely (weights,
                // Lion momentum untouched) until the accumulator reaches its
                // window. At the boundary, overwrite p.g with the true
                // cross-window average before the normal optimizer call
                // below -- everything after this check is unchanged from the
                // pss_accum_steps()==1 (default) path.
                if (is_pss_pred_weight(p)) {
                    if (!pss_at_boundary) continue;
                    float* accum = (p.w == w.pss_pred_down)
                        ? sb.pss_window_accum_down : sb.pss_window_accum_up;
                    IDA_CUDA_CHECK(cudaMemcpyAsync(p.g, accum, p.n * sizeof(float),
                        cudaMemcpyDeviceToDevice, arena.stream));
                    k_scale_f32<<<ceildiv(p.n, 256), 256, 0, arena.stream>>>(
                        p.g, p.n, 1.0f / static_cast<float>(sb.pss_window_count));
                }
                const float slot_lr = static_cast<float>(lr_t) *
                    (is_lrss_weight(p) ? lrss_lr_scale :
                     is_pss_pred_weight(p) ? pss_pred_lr_scale : 1.0f) *
                    ((is_lrss_weight(p) || is_pss_pred_weight(p))
                        ? last_support_transition_scale : 1.0f) *
                    ((use_lion && trust_ratio_enabled(request)) ? slot_trust_ratio[slot_idx] : 1.0f);
                // The legacy default seed 0 applies the same rounding
                // threshold to equal indices in every tensor and every step.
                // Mix stable slot/step identities to retain exact replay while
                // restoring stochastic rounding across parameters and time.
                const unsigned sr_seed = optimizer_sr_seed_mix
                    ? (static_cast<unsigned>(opt_step) * 0x9e3779b9u) ^
                      (static_cast<unsigned>(slot_idx + 1) * 0x85ebca6bu)
                    : 0u;
                if (p.fp8_amax) {
                    IDA_CUDA_CHECK(cudaMemsetAsync(p.fp8_amax, 0, sizeof(float), arena.stream));
                }
                const bool debug_gate_bias = lrss_bias_debug &&
                    p.w == w.lrss_gate_b && opt_step <= 2;
                const bool have_slot_name = slot_idx < slot_names.size();
                if (have_slot_name) nvtxRangePushA(slot_names[slot_idx].c_str());
                const std::size_t chunk_capacity = p.g_bf16
                    ? sb.grad_update_scratch_n : p.n;
                if (chunk_capacity == 0) {
                    throw std::runtime_error("empty optimizer staging capacity");
                }
                if (debug_gate_bias) {
                    std::fprintf(stderr,
                        "[lrss-bias-debug] opt_step=%d lr=%.9g preclip_norm=%.9g "
                        "global_clip_scale=%.9g\n",
                        opt_step, slot_lr,
                        std::sqrt(std::max(0.0f, h_slot_normsq[slot_idx])),
                        last_global_clip_scale);
                    debug_print_bf16_vector_stats(
                        opt_step, "weight.pre", p.w, p.n, arena.stream);
                }
                for (std::size_t offset = 0; offset < p.n; offset += chunk_capacity) {
                    const std::size_t chunk_n = std::min(chunk_capacity, p.n - offset);
                    float* optimizer_grad = p.g ? p.g + offset : sb.grad_update_scratch;
                    if (p.g_bf16) {
                        cast_bf16_to_f32(p.g_bf16 + offset, sb.grad_update_scratch,
                                         chunk_n, arena.stream);
                        optimizer_grad = sb.grad_update_scratch;
                    }
                    if (debug_gate_bias && offset == 0) {
                        debug_print_f32_vector_stats(
                            opt_step, "grad.clipped", optimizer_grad, chunk_n, arena.stream);
                    }
                    if (use_lion) {
                        // Stage 2a: v is allocated but intentionally unused here
                        // (see optimizer_uses_lion comment) -- Lion is a single-
                        // moment optimizer by construction. lr/wd multipliers are
                        // separate from AdamW's slot_lr/p.wd -- Lion needs a
                        // materially different scale, probed via env, not
                        // inherited from the AdamW-tuned config values.
                        if (p.m.bf16) {
                            lion_step(
                                p.w + offset, p.m.bf16 + offset, optimizer_grad, chunk_n,
                                slot_lr * lion_lr_mult, lion_b1, lion_b2,
                                p.wd * lion_wd_mult,
                                arena.stream, p.fp8_amax, sr_seed
                            );
                        } else {
                            lion_step(
                                p.w + offset, p.m.f32 + offset, optimizer_grad, chunk_n,
                                slot_lr * lion_lr_mult, lion_b1, lion_b2,
                                p.wd * lion_wd_mult,
                                arena.stream, p.fp8_amax, sr_seed
                            );
                        }
                    }
#if IDA_NATIVE_ENABLE_ADAMW
                    else if (p.m.bf16) {
                        adamw_step(
                            p.w + offset, p.m.bf16 + offset, p.v.bf16 + offset,
                            optimizer_grad, chunk_n, slot_lr, b1, b2, eps_a, p.wd,
                            opt_step - skipped, arena.stream, p.fp8_amax, sr_seed
                        );
                    } else {
                        adamw_step(
                            p.w + offset, p.m.f32 + offset, p.v.f32 + offset,
                            optimizer_grad, chunk_n, slot_lr, b1, b2, eps_a, p.wd,
                            opt_step - skipped, arena.stream, p.fp8_amax, sr_seed
                        );
                    }
#else
                    else {
                        throw std::runtime_error(
                            "Adam and AdamW are disabled in the native public runtime");
                    }
#endif
                }
                if (have_slot_name) nvtxRangePop();
                if (debug_gate_bias) {
                    debug_print_bf16_vector_stats(
                        opt_step, "weight.post", p.w, p.n, arena.stream);
                    if (p.m.bf16) {
                        debug_print_bf16_vector_stats(
                            opt_step, "moment.m", p.m.bf16, p.n, arena.stream);
                        // Lion has no second moment -- v is unallocated
                        // (null) when use_lion skipped it (Stage 2b).
                        if (!use_lion) {
                            debug_print_bf16_vector_stats(
                                opt_step, "moment.v", p.v.bf16, p.n, arena.stream);
                        }
                    } else {
                        debug_print_f32_vector_stats(
                            opt_step, "moment.m", p.m.f32, p.n, arena.stream);
                        if (!use_lion) {
                            debug_print_f32_vector_stats(
                                opt_step, "moment.v", p.v.f32, p.n, arena.stream);
                        }
                    }
                }
            }
            // Weights changed: refresh the FP8 caches.
            refresh_fp8_weights(f8, arena.stream, /*recompute_amax=*/false);
            refresh_ampere_packed_weights(ampere_packed, arena.stream);

            if (ckpt_every_opt_steps > 0 &&
                opt_step % ckpt_every_opt_steps == 0) {
                std::string save_err;
                if (!begin_async_weight_save(request, w, arena.stream,
                                             weight_saver, save_err, opt_step)) {
                    // Durability is best-effort mid-burn; never kill training.
                    std::fprintf(stderr,
                        "[ida_native_train] async checkpoint save failed "
                        "(opt_step %d): %s\n", opt_step, save_err.c_str());
                }
                // Exact resume (Phase 3): optimizer state + cursor, synchronous
                // (no async variant yet -- a real, disclosed training stall
                // proportional to optimizer-state size, same cadence as the
                // weight saver above). Without this, a crash between periodic
                // checkpoints would have weights but no matching optimizer/
                // cursor state to resume from, defeating the point of exact
                // resume for the crash-mid-burn case specifically. Best-effort
                // like the weight save: never kill training on failure.
                resume_state.cumulative_opt_steps = opt_step;
                resume_state.cumulative_micro_steps = micro_done;
                resume_state.dataset_cursor = static_cast<std::size_t>(seq_cursor);
                if (!save_lattice_opt_safetensors(request, w, opt, resume_state,
                                                  arena.stream, save_err)) {
                    std::fprintf(stderr,
                        "[ida_native_train] optimizer-state checkpoint save "
                        "failed (opt_step %d): %s\n", opt_step, save_err.c_str());
                }
            }

            // Reset the PSS cross-window accumulator once its window fired.
            // Deliberately OUTSIDE the per-slot loop above (but still inside
            // this "not skipped" branch, where pss_at_boundary lives):
            // pss_at_boundary is shared by both PSS slots (pred_down,
            // pred_up) in that loop, and resetting mid-loop after processing
            // only one of them would make the second see a wrong
            // (already-cleared) window_count.
            if (pss_at_boundary) {
                const std::size_t pss_n = static_cast<std::size_t>(w.hidden_size) * w.pss_pred_rank;
                k_zeros_f32<<<ceildiv(pss_n, 256), 256, 0, arena.stream>>>(
                    sb.pss_window_accum_down, pss_n);
                k_zeros_f32<<<ceildiv(pss_n, 256), 256, 0, arena.stream>>>(
                    sb.pss_window_accum_up, pss_n);
                sb.pss_window_count = 0;
            }
        }

        // ── Zero gradients for the next accumulation window ──────────────────
        for (auto& p : slots) {
            if (p.g_bf16) {
                k_zeros_bf16<<<ceildiv(p.n, 256), 256, 0, arena.stream>>>(p.g_bf16, p.n);
            } else {
                k_zeros_f32<<<ceildiv(p.n, 256), 256, 0, arena.stream>>>(p.g, p.n);
            }
        }
        // Worker grad sets zero on the main stream: the next window's worker
        // kernels are already fenced behind ev_weights (recorded after this).
        for (auto& wk : xworkers)
            zero_lattice_grads(wk.g, w, arena.stream);

        // ── Metrics ──────────────────────────────────────────────────────────
        loss_hist.push_back(last_loss);
        lr_hist.push_back(static_cast<float>(lr_t));
        gn_hist.push_back(last_grad_norm);
        global_clip_fired_hist.push_back(last_global_clip_fired ? 1 : 0);
        global_clip_scale_hist.push_back(last_global_clip_scale);
        embed_clip_fired_hist.push_back(last_embed_clip_fired ? 1 : 0);
        embed_clip_rows_hist.push_back(last_embed_clipped_rows);
        embed_clip_max_preclip_hist.push_back(last_embed_clip_max_preclip_norm);
        dominant_slot_norm_hist.push_back(last_dominant_grad_slot_norm);
        dominant_slot_frac_hist.push_back(last_dominant_grad_slot_frac);
        {
            ::ida_native::ontology::StepOutcome so;
            so.opt_step       = opt_step;
            so.loss           = last_loss;
            so.wallclock_s    = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - step_wall_start).count();
            so.tokens         = total_tokens;
            so.update_skipped = step_update_skipped;
            so.clip_fired     = last_global_clip_fired;
            so.grad_norm      = last_grad_norm;
            ::ida_native::ontology::observe_step(so);
        }
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
        if (shared_metrics_out.is_open() || burn_metrics_out.is_open()) {
            const auto now = std::chrono::steady_clock::now();
            const double mono = std::chrono::duration<double>(now - burn_wall_start).count();
            const double step_ms = std::chrono::duration<double, std::milli>(
                now - step_wall_start).count();
            const double epoch = ds.num_sequences > 0
                ? static_cast<double>(examples_seen) / ds.num_sequences : 0.0;
            // Training-derived floats can genuinely go non-finite during a
            // real divergence (this session's own MoE NaN bug, for one) --
            // %f renders NaN/Inf as bare lowercase "nan"/"inf"/"-inf",
            // invalid JSON that even Python's stdlib json module rejects
            // (only the capitalized NaN/Infinity tokens are accepted as an
            // extension). Pre-format each as a %s argument instead: "null"
            // for non-finite, the same fixed-precision text otherwise.
            auto fnum = [](double v, int prec) -> std::string {
                if (!std::isfinite(v)) return "null";
                char b[40];
                std::snprintf(b, sizeof(b), "%.*f", prec, v);
                return std::string(b);
            };
            const std::string s_loss = fnum(last_loss, 6);
            const std::string s_loss_ema = fnum(loss_ema, 6);
            const std::string s_lr = fnum(lr_t, 8);
            const std::string s_grad_norm = fnum(last_grad_norm, 6);
            const std::string s_global_clip_scale = fnum(last_global_clip_scale, 8);
            const std::string s_embed_clip_threshold = fnum(embed_clip_threshold, 6);
            const std::string s_embed_clip_max_preclip = fnum(last_embed_clip_max_preclip_norm, 6);
            const std::string s_dom_norm = fnum(last_dominant_grad_slot_norm, 6);
            const std::string s_dom_frac = fnum(last_dominant_grad_slot_frac, 6);
            const std::string s_pss_predictor_grad_norm = fnum(last_pss_predictor_grad_norm, 6);
            const std::string s_pss_predictor_grad_frac = fnum(last_pss_predictor_grad_frac, 6);
            const std::string s_pss_predictor_momentum_norm = fnum(last_pss_predictor_momentum_norm, 6);
            const std::string s_dom2_norm = fnum(last_dominant_grad_slot_2_norm, 6);
            const std::string s_dom2_frac = fnum(last_dominant_grad_slot_2_frac, 6);
            const std::string s_dom3_norm = fnum(last_dominant_grad_slot_3_norm, 6);
            const std::string s_dom3_frac = fnum(last_dominant_grad_slot_3_frac, 6);
            const std::string s_lss_residual_scale = fnum(last_lss_feedback_residual_scale, 6);
            const std::string s_lss_aux = fnum(last_lss_aux, 6);
            const std::string s_lss_recon_norm = fnum(last_lss_recon_norm, 6);
            const std::string s_lss_target_norm = fnum(last_lss_target_norm, 6);
            const std::string s_lss_relative_rmse = fnum(last_lss_relative_rmse, 6);
            const std::string s_support_transition_scale = fnum(last_support_transition_scale, 6);
            const std::string s_pss_spike_ratio = fnum(last_pss_spike_ratio_max, 6);
            const std::string s_pss_confidence = fnum(last_pss_confidence, 6);
            const std::string s_pss_pred_err = fnum(last_pss_pred_err, 6);
            const std::string s_pss_int2_agreement = fnum(last_pss_int2_agreement, 6);
            const std::string s_pss_engaged_frac = fnum(sb.pss_engaged_frac, 6);
            const std::string s_pss_int2_inv_rms = fnum(last_pss_int2_inv_rms, 8);
            const std::string s_pss_confidence_min = fnum(last_pss_confidence_min, 6);
            const std::string s_pss_confidence_max = fnum(last_pss_confidence_max, 6);
            const std::string s_pss_blend_delta_rms = fnum(last_pss_blend_delta_rms, 6);
            const std::string s_pss_aux_weight = fnum(last_pss_aux_weight, 8);
            const std::string s_pss_aux_denom = fnum(last_pss_aux_denom, 8);
            const std::string s_pss_effective_confidence = fnum(last_pss_effective_confidence, 6);
            const std::string s_pss_conditioning_mode = json_escape(pss_conditioning_mode);
            const std::string s_pss_aux_normalize_mode = json_escape(pss_aux_normalize_mode);
            auto json_float_array = [fnum](const std::array<float, kPssMagBuckets>& arr) {
                std::string s = "[";
                for (int i = 0; i < kPssMagBuckets; ++i) {
                    if (i) s += ",";
                    s += fnum(arr[i], 6);
                }
                s += "]";
                return s;
            };
            const std::string s_pss_mag_bucket_err_ratio =
                json_float_array(last_pss_mag_bucket_err_ratio);
            const std::string s_pss_mag_bucket_hit_rate =
                json_float_array(last_pss_mag_bucket_hit_rate);
            char line[6144];
            std::snprintf(line, sizeof(line),
                "{\"timestamp\": \"%s\", \"monotonic_s\": %.3f, \"seat\": \"%s\", "
                "\"family\": \"%s\", \"version\": \"%s\", \"attention_backend\": \"%s\", "
                "\"precision_profile\": \"%s\", \"global_step\": %d, "
                "\"optimizer_step\": %d, \"micro_step\": %d, \"examples_seen\": %zu, "
                "\"tokens_seen\": %zu, \"epoch\": %.4f, \"loss\": %s, "
                "\"loss_ema\": %s, \"learning_rate\": %s, \"grad_norm\": %s, "
                "\"global_grad_clip_fired\": %s, \"global_grad_clip_scale\": %s, "
                "\"embed_row_clip_enabled\": %s, \"embed_row_clip_fired\": %s, "
                "\"embed_row_clip_threshold\": %s, \"embed_row_clip_max_preclip_norm\": %s, "
                "\"embed_row_clipped_rows\": %d, \"dominant_grad_slot\": \"%s\", "
                "\"dominant_grad_slot_norm\": %s, \"dominant_grad_slot_frac\": %s, "
                "\"pss_predictor_grad_norm\": %s, \"pss_predictor_grad_frac\": %s, "
                "\"pss_predictor_momentum_norm\": %s, "
                "\"pss_window_count\": %d, \"pss_window_applied\": %s, "
                "\"dominant_grad_slot_2\": \"%s\", \"dominant_grad_slot_2_norm\": %s, \"dominant_grad_slot_2_frac\": %s, "
                "\"dominant_grad_slot_3\": \"%s\", \"dominant_grad_slot_3_norm\": %s, \"dominant_grad_slot_3_frac\": %s, "
                "\"lss_feedback_skip_tail\": %d, \"lss_feedback_residual_scale\": %s, \"lss_aux\": %s, "
                "\"lss_recon_norm\": %s, \"lss_target_norm\": %s, \"lss_relative_rmse\": %s, "
                "\"support_transition_scale\": %s, "
                "\"pss_slot_clip_steps\": %d, \"pss_spike_ratio_max\": %s, \"pss_spike_slot\": \"%s\", "
                "\"pss_confidence\": %s, \"pss_pred_err\": %s, "
                "\"pss_int2_agreement\": %s, \"pss_engaged_frac\": %s, "
                "\"pss_covered\": %u, \"pss_n\": %zu, "
                "\"pss_int2_matched\": %u, \"pss_int2_scored\": %u, "
                "\"pss_scored_micros\": %d, \"pss_int2_inv_rms\": %s, "
                "\"pss_confidence_min\": %s, \"pss_confidence_max\": %s, "
                "\"pss_blend_delta_rms\": %s, \"pss_aux_weight\": %s, "
                "\"pss_aux_denom\": %s, \"pss_aux_normalize\": %s, "
                "\"pss_conditioning_active\": %s, \"pss_override_active\": %s, "
                "\"pss_effective_confidence\": %s, \"pss_governor_event\": \"%s\", "
                "\"pss_conditioning_mode\": \"%s\", \"pss_aux_normalize_mode\": \"%s\", "
                "\"pss_mag_bucket_err_ratio\": %s, \"pss_mag_bucket_hit_rate\": %s, "
                "\"step_duration_ms\": %.1f, \"active_grad_accum\": %d, "
                "\"requested_grad_accum\": %d, \"effective_batch\": %d, "
                "\"event\": \"native_step\"}",
                utc_now().c_str(), mono,
                request.seat.c_str(), request.family.c_str(), request.version.c_str(),
                attention_backend_name(attention_backend), request.precision_profile.c_str(),
                micro_done, opt_step, micro_done, examples_seen, total_tokens, epoch,
                s_loss.c_str(), s_loss_ema.c_str(), s_lr.c_str(), s_grad_norm.c_str(),
                last_global_clip_fired ? "true" : "false",
                s_global_clip_scale.c_str(),
                embed_clip_threshold > 0.0f ? "true" : "false",
                last_embed_clip_fired ? "true" : "false",
                s_embed_clip_threshold.c_str(),
                s_embed_clip_max_preclip.c_str(),
                last_embed_clipped_rows,
                last_dominant_grad_slot.c_str(),
                s_dom_norm.c_str(),
                s_dom_frac.c_str(),
                s_pss_predictor_grad_norm.c_str(),
                s_pss_predictor_grad_frac.c_str(),
                s_pss_predictor_momentum_norm.c_str(),
                last_pss_window_count,
                last_pss_window_applied ? "true" : "false",
                last_dominant_grad_slot_2.c_str(),
                s_dom2_norm.c_str(),
                s_dom2_frac.c_str(),
                last_dominant_grad_slot_3.c_str(),
                s_dom3_norm.c_str(),
                s_dom3_frac.c_str(),
                last_lss_feedback_skip_tail,
                s_lss_residual_scale.c_str(),
                s_lss_aux.c_str(),
                s_lss_recon_norm.c_str(),
                s_lss_target_norm.c_str(),
                s_lss_relative_rmse.c_str(),
                s_support_transition_scale.c_str(),
                pss_slot_clip_fired_steps,
                s_pss_spike_ratio.c_str(),
                last_pss_spike_slot.c_str(),
                s_pss_confidence.c_str(),
                s_pss_pred_err.c_str(),
                s_pss_int2_agreement.c_str(),
                s_pss_engaged_frac.c_str(),
                last_pss_covered,
                last_pss_n,
                last_pss_int2_matched,
                last_pss_int2_scored,
                last_pss_scored_micros,
                s_pss_int2_inv_rms.c_str(),
                s_pss_confidence_min.c_str(),
                s_pss_confidence_max.c_str(),
                s_pss_blend_delta_rms.c_str(),
                s_pss_aux_weight.c_str(),
                s_pss_aux_denom.c_str(),
                last_pss_aux_normalize ? "true" : "false",
                last_pss_conditioning_active ? "true" : "false",
                last_pss_override_active ? "true" : "false",
                s_pss_effective_confidence.c_str(),
                last_pss_governor_event.c_str(),
                s_pss_conditioning_mode.c_str(),
                s_pss_aux_normalize_mode.c_str(),
                s_pss_mag_bucket_err_ratio.c_str(),
                s_pss_mag_bucket_hit_rate.c_str(),
                step_ms,
                micros_this_step,
                accum_ceiling,
                mb * micros_this_step);
            if (shared_metrics_out.is_open()) {
                shared_metrics_out << line << "\n";
                shared_metrics_out.flush();
            }
            if (burn_metrics_out.is_open()) {
                burn_metrics_out << line << "\n";
                burn_metrics_out.flush();
            }
        }
#endif

        // MLPerf: per-step train_loss + throughput; epoch block boundaries.
        {
            const std::string step_meta =
                "{\"step_num\": " + std::to_string(opt_step) +
                ", \"samples_count\": " + std::to_string(examples_seen) + "}";
            ml.point("train_loss", std::to_string(last_loss), step_meta);
            ml.point("grad_norm", std::to_string(last_grad_norm), step_meta);
            const int cur_epoch = ds.num_sequences > 0
                ? static_cast<int>(examples_seen / ds.num_sequences) : 0;
            if (cur_epoch > ml_epoch) {
                ml.end("block_stop",
                       "{\"first_epoch_num\": " + std::to_string(ml_epoch) +
                       ", \"samples_count\": " + std::to_string(examples_seen) + "}");
                ml_epoch = cur_epoch;
                if (micro_done < total_micro)
                    ml.begin("block_start",
                             "{\"first_epoch_num\": " + std::to_string(ml_epoch) +
                             ", \"epoch_count\": 1}");
            }
        }

        // ── Heartbeat ────────────────────────────────────────────────────────
        if (on_step) {
            auto now = std::chrono::steady_clock::now();
            long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now - last_hb).count();
            if (opt_step == 1 || ms >= kHbMs || micro_done >= total_micro) {
                cudaEvent_t hb{};
                IDA_CUDA_CHECK(cudaEventCreate(&hb));
                IDA_CUDA_CHECK(cudaEventRecord(hb, arena.stream));
                IDA_CUDA_CHECK(cudaEventSynchronize(hb));
                float ms_gpu = 0.0f;
                IDA_CUDA_CHECK(cudaEventElapsedTime(&ms_gpu, ev_start, hb));
                IDA_CUDA_CHECK(cudaEventDestroy(hb));
                ProgressReport rep{};
                rep.micro_step      = micro_done;
                rep.optimizer_step  = opt_step;
                rep.active_grad_accum = micros_this_step;
                rep.requested_grad_accum = accum_ceiling;
                rep.tokens          = total_tokens;
                rep.elapsed_s       = ms_gpu / 1000.0;
                rep.loss            = last_loss;
                rep.loss_ema        = loss_ema;
                rep.grad_norm       = last_grad_norm;
                rep.lr              = static_cast<float>(lr_t);
                rep.effective_batch = mb * micros_this_step;
                rep.skipped_steps   = skipped;
                rep.global_grad_clip_steps = global_clip_fired_steps;
                rep.embed_row_clip_steps = embed_clip_fired_steps;
                rep.embed_row_clipped_rows_total = embed_clipped_rows_total;
                rep.global_grad_clip_fired = last_global_clip_fired;
                rep.global_grad_clip_scale = last_global_clip_scale;
                rep.embed_row_clip_fired = last_embed_clip_fired;
                rep.embed_row_clip_threshold = embed_clip_threshold;
                rep.embed_row_clip_max_preclip_norm = last_embed_clip_max_preclip_norm;
                rep.embed_row_clipped_rows = last_embed_clipped_rows;
                rep.dominant_grad_slot = last_dominant_grad_slot;
                rep.dominant_grad_slot_norm = last_dominant_grad_slot_norm;
                rep.dominant_grad_slot_frac = last_dominant_grad_slot_frac;
                rep.lss_feedback_skip_tail = last_lss_feedback_skip_tail;
                rep.lss_feedback_residual_scale = last_lss_feedback_residual_scale;
                rep.lss_aux = last_lss_aux;
                rep.lss_recon_norm = last_lss_recon_norm;
                rep.lss_target_norm = last_lss_target_norm;
                rep.lss_relative_rmse = last_lss_relative_rmse;
                rep.support_transition_scale = last_support_transition_scale;
                rep.pss_slot_clip_steps = pss_slot_clip_fired_steps;
                rep.pss_spike_ratio_max = last_pss_spike_ratio_max;
                rep.pss_spike_slot = last_pss_spike_slot;
                rep.pss_confidence = last_pss_confidence;
                rep.pss_pred_err = last_pss_pred_err;
                rep.pss_int2_agreement = last_pss_int2_agreement;
                rep.pss_engaged_frac = sb.pss_engaged_frac;
                rep.pss_covered = last_pss_covered;
                rep.pss_n = last_pss_n;
                rep.pss_int2_matched = last_pss_int2_matched;
                rep.pss_int2_scored = last_pss_int2_scored;
                rep.pss_scored_micros = last_pss_scored_micros;
                rep.pss_int2_inv_rms = last_pss_int2_inv_rms;
                rep.pss_confidence_min = last_pss_confidence_min;
                rep.pss_confidence_max = last_pss_confidence_max;
                rep.pss_blend_delta_rms = last_pss_blend_delta_rms;
                rep.pss_aux_weight = last_pss_aux_weight;
                rep.pss_aux_denom = last_pss_aux_denom;
                rep.pss_aux_normalize = last_pss_aux_normalize;
                rep.pss_conditioning_active = last_pss_conditioning_active;
                rep.pss_override_active = last_pss_override_active;
                rep.pss_effective_confidence = last_pss_effective_confidence;
                rep.pss_governor_event = last_pss_governor_event;
                rep.pss_conditioning_mode = pss_conditioning_mode;
                rep.pss_aux_normalize_mode = pss_aux_normalize_mode;
                rep.fp8_active      = f8.on;
                rep.attention_backend = attention_backend_name(attention_backend);
                rep.precision_profile = request.precision_profile;
                try { on_step(rep); } catch (...) {}
                last_hb = now;
            }
        }
    }

    IDA_CUDA_CHECK(cudaEventRecord(ev_stop, arena.stream));
    IDA_CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float elapsed_ms = 0.0f;
    IDA_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    // MLPerf: close the final block and the run.
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
    {
        const double final_elapsed =
            std::max(0.000001, static_cast<double>(elapsed_ms) / 1000.0);
        ml.end("block_stop",
               "{\"first_epoch_num\": " + std::to_string(ml_epoch) +
               ", \"samples_count\": " + std::to_string(examples_seen) + "}");
        ml.point("train_samples_processed", std::to_string(examples_seen));
        ml.point("tokens_processed", std::to_string(total_tokens));
        ml.point("throughput_tokens_per_second",
                 std::to_string(static_cast<double>(total_tokens) / final_elapsed));
        ml.point("skipped_optimizer_steps", std::to_string(skipped));
        ml.end("run_stop",
               std::string("{\"status\": \"") +
               (skipped == 0 && std::isfinite(last_loss) ? "success" : "aborted") +
               "\", \"final_loss\": " + std::to_string(last_loss) +
               ", \"optimizer_steps\": " + std::to_string(opt_step) + "}");
        write_native_timeline_event(request, "native_train_loop", "end",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - native_train_loop_t0).count());
    }

    // Optimizer summary: full per-step arrays (telemetry contract + per-student copy)
    {
        std::ostringstream js;
        auto arr = [&](const std::vector<float>& v) {
            js << "[";
            for (std::size_t i = 0; i < v.size(); ++i)
                js << (i ? ", " : "") << v[i];
            js << "]";
        };
        auto arr_int = [&](const std::vector<int>& v) {
            js << "[";
            for (std::size_t i = 0; i < v.size(); ++i)
                js << (i ? ", " : "") << v[i];
            js << "]";
        };
        js << "{\n  \"timestamp\": \"" << utc_now() << "\",\n"
           << "  \"seat\": \"" << request.seat << "\",\n"
           << "  \"family\": \"" << request.family << "\",\n"
           << "  \"version\": \"" << request.version << "\",\n"
           << "  \"backend\": \"native\",\n"
           << "  \"attention_backend\": \"" << attention_backend_name(attention_backend) << "\",\n"
           << "  \"precision_profile\": \"" << request.precision_profile << "\",\n"
           << "  \"fp8_active\": " << (f8.on ? "true" : "false") << ",\n"
           << "  \"optimizer_steps\": " << opt_step << ",\n"
           << "  \"skipped_steps\": " << skipped << ",\n"
           << "  \"loss_delta\": "
           << (loss_hist.empty() ? 0.0f : loss_hist.front() - loss_hist.back()) << ",\n"
           << "  \"loss_by_step\": ";  arr(loss_hist);
        js << ",\n  \"lr_by_step\": ";  arr(lr_hist);
        js << ",\n  \"grad_norm_by_step\": ";  arr(gn_hist);
        js << ",\n  \"global_grad_clip_fired_by_step\": "; arr_int(global_clip_fired_hist);
        js << ",\n  \"global_grad_clip_scale_by_step\": "; arr(global_clip_scale_hist);
        js << ",\n  \"embed_row_clip_fired_by_step\": "; arr_int(embed_clip_fired_hist);
        js << ",\n  \"embed_row_clipped_rows_by_step\": "; arr_int(embed_clip_rows_hist);
        js << ",\n  \"embed_row_clip_max_preclip_norm_by_step\": "; arr(embed_clip_max_preclip_hist);
        js << ",\n  \"dominant_grad_slot_norm_by_step\": "; arr(dominant_slot_norm_hist);
        js << ",\n  \"dominant_grad_slot_frac_by_step\": "; arr(dominant_slot_frac_hist);
        js << "\n}\n";
        const std::string payload = js.str();
        for (const auto& p : {
                 request.repo_root / "artifacts" / "telemetry" / "native_optimizer_summary.json",
                 request.output_dir / "optimizer_summary.json"}) {
            std::error_code ec;
            std::filesystem::create_directories(p.parent_path(), ec);
            std::ofstream out(p, std::ios::trunc);
            if (out) out << payload;
        }
    }
#endif

    // ── Durable weight body ───────────────────────────────────────────────────
    // Written before any teardown so a failure here still leaves the model in
    // device memory for diagnosis.  IDA_NATIVE_SAVE_WEIGHTS=0 is an ablation
    // escape hatch (verification harness runs that would otherwise write
    // gigabytes per iteration); default is on — a burn that trains and then
    // discards its weights is a metadata husk by construction.
    {
        // Any in-flight periodic async save must land (or fail) before the
        // final authoritative write below replaces the same file.
        drain_async_weight_saver(weight_saver);
        if (weight_saver.saves_completed > 0 || weight_saver.saves_skipped > 0) {
            std::fprintf(stderr,
                "[ida_native_train] periodic async checkpoints: %d written, "
                "%d skipped (writer busy)\n",
                weight_saver.saves_completed, weight_saver.saves_skipped);
        }
        const char* sw = std::getenv("IDA_NATIVE_SAVE_WEIGHTS");
        const bool save_enabled = !(sw && sw[0] == '0');
        if (save_enabled) {
            write_native_timeline_event(request, "native_checkpoint_save", "start");
            const auto ckpt_t0 = std::chrono::steady_clock::now();
            std::string err;
            if (!save_lattice_weights_safetensors(request, w, arena.stream, err, opt_step)) {
                throw std::runtime_error("weight save failed: " + err);
            }
            write_native_timeline_event(request, "native_checkpoint_save", "end",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - ckpt_t0).count());
            std::fprintf(stderr,
                "[ida_native_train] weights saved: %s (%.1f MiB)\n",
                (request.output_dir / "model.safetensors").string().c_str(),
                static_cast<double>(w.total_bytes) / (1024.0 * 1024.0));
            // Exact resume (Phase 3): final optimizer state + cursor,
            // end-of-burn. A failure here is a WARN, not fatal -- the burn
            // itself completed successfully and its weights are already
            // durable; a missing optimizer_state.safetensors only means a
            // FUTURE resume_from_checkpoint against this exact directory
            // would fail (loudly, via load_lattice_opt_safetensors' own
            // fail-closed check), not that this burn's own result is lost.
            resume_state.cumulative_opt_steps = opt_step;
            resume_state.cumulative_micro_steps = micro_done;
            resume_state.dataset_cursor = static_cast<std::size_t>(seq_cursor);
            std::string opt_save_err;
            if (!save_lattice_opt_safetensors(request, w, opt, resume_state,
                                              arena.stream, opt_save_err)) {
                std::fprintf(stderr,
                    "[ida_native_train] end-of-burn optimizer-state save "
                    "failed (non-fatal, weights already saved): %s\n",
                    opt_save_err.c_str());
            }
        }
        // PSS predictor standalone state: deliberately OUTSIDE the
        // IDA_NATIVE_SAVE_WEIGHTS gate — probes run SAVE_WEIGHTS=0 and are
        // exactly the runs that need the head's accumulated training
        // persisted. Failure is a WARN, never fatal: an auxiliary head must
        // not wedge a completed burn. Request field wins over env — see the
        // load-side comment above for the worker-caching rationale.
        {
            const std::string out_path_str = !request.pss_pred_state_out.empty()
                ? request.pss_pred_state_out
                : [] {
                      const char* e = std::getenv("IDA_NATIVE_PSS_PRED_STATE_OUT");
                      return std::string(e ? e : "");
                  }();
            const char* pss_out = out_path_str.c_str();
            if (pss_out && pss_out[0] && w.pss_pred_rank > 0) {
                write_native_timeline_event(request, "native_pss_state_save", "start");
                const auto pss_save_t0 = std::chrono::steady_clock::now();
                const int cumulative = w.pss_pred_prior_opt_steps + opt_step;
                std::string err;
                if (!save_pss_pred_state_safetensors(
                        request, w, pss_out, cumulative, arena.stream, err)) {
                    std::fprintf(stderr,
                        "[ida_native_train] WARN: pss-pred state save failed: %s\n",
                        err.c_str());
                } else {
                    std::fprintf(stderr,
                        "[ida_native_train] pss-pred state saved: %s "
                        "(cumulative_opt_steps=%d)\n",
                        pss_out, cumulative);
                }
                write_native_timeline_event(request, "native_pss_state_save", "end",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - pss_save_t0).count());
            }
        }
    }

    // Cleanup
    for (auto& wk : xworkers) {
        IDA_CUDA_CHECK(cudaStreamSynchronize(wk.arena.stream));
        IDA_CUDA_CHECK(cudaFreeAsync(wk.d_segs,   wk.arena.stream));
        IDA_CUDA_CHECK(cudaFreeAsync(wk.d_tokens, wk.arena.stream));
        IDA_CUDA_CHECK(cudaFreeAsync(wk.d_labels, wk.arena.stream));
        CUBLAS_CHECK(cublasDestroy(wk.cublas));
        free_fp8_ctx(wk.f8, wk.arena);
        free_packed_fp4_attention_ctx(wk.fp4, wk.arena);
        free_step_buffers(wk.sb, wk.arena);
        free_lattice_grads(wk.g, w, wk.arena);
        IDA_CUDA_CHECK(cudaStreamSynchronize(wk.arena.stream));
        IDA_CUDA_CHECK(cudaEventDestroy(wk.ev_done));
        IDA_CUDA_CHECK(cudaStreamDestroy(wk.arena.stream));
    }
    IDA_CUDA_CHECK(cudaEventDestroy(ev_weights));
    if (lrss_scr) {
        sb.lrss_p = nullptr; sb.lrss_s = nullptr; sb.lrss_g = nullptr;
        free_lrss_scratch(lrss_scr, arena);
        lrss_scr = nullptr;
    }
    IDA_CUDA_CHECK(cudaFreeAsync(d_segs, arena.stream));
    IDA_CUDA_CHECK(cudaFreeAsync(d_tokens, arena.stream));
    IDA_CUDA_CHECK(cudaFreeAsync(d_labels, arena.stream));
    IDA_CUDA_CHECK(cudaFreeAsync(d_slot_normsq, arena.stream));
    IDA_CUDA_CHECK(cudaFreeAsync(d_slot_wnormsq, arena.stream));
    IDA_CUDA_CHECK(cudaFreeAsync(d_micro_losses, arena.stream));
    IDA_CUDA_CHECK(cudaFreeAsync(d_micro_clip_stats, arena.stream));
    IDA_CUDA_CHECK(cudaEventDestroy(ev_stop));
    IDA_CUDA_CHECK(cudaEventDestroy(ev_start));
    CUBLAS_CHECK(cublasDestroy(cublas));
    const bool fp8_was_on = f8.on;
    free_fp8_ctx(f8, arena);
    g_ampere_packed_weights = nullptr;
    free_ampere_packed_weights(ampere_packed, arena);
    free_packed_fp4_attention_ctx(fp4_attn, arena);
    free_step_buffers(sb, arena);
    free_lattice_grads(g, w, arena);
    free_lattice_opt(opt, w, arena);
    free_lattice_weights(w, arena);

    const double elapsed_s = std::max(0.000001, static_cast<double>(elapsed_ms) / 1000.0);
    BurnResult result{};
    result.global_step       = micro_done;
    result.optimizer_steps   = opt_step;
    result.tokens_processed  = total_tokens;
    result.elapsed_seconds   = elapsed_s;
    result.tokens_per_second = static_cast<double>(total_tokens) / elapsed_s;
    result.device_bytes_touched = w.total_bytes;
    ::ida_native::gemm_trace::dump(
        ::ida_native::ontology::sink(), elapsed_s,
        ::ida_native::ontology::context_epoch());

    result.final_loss        = last_loss;
    result.final_loss_ema    = loss_ema;
    result.final_grad_norm   = last_grad_norm;
    result.final_lr          = static_cast<float>(last_lr);
    result.skipped_steps     = skipped;
    result.global_grad_clip_steps = global_clip_fired_steps;
    result.embed_row_clip_steps = embed_clip_fired_steps;
    result.embed_row_clipped_rows_total = embed_clipped_rows_total;
    result.final_global_grad_clip_fired = last_global_clip_fired;
    result.final_global_grad_clip_scale = last_global_clip_scale;
    result.final_embed_row_clip_fired = last_embed_clip_fired;
    result.final_embed_row_clip_threshold = embed_clip_threshold;
    result.final_embed_row_clip_max_preclip_norm = last_embed_clip_max_preclip_norm;
    result.final_embed_row_clipped_rows = last_embed_clipped_rows;
    result.final_dominant_grad_slot = last_dominant_grad_slot;
    result.final_dominant_grad_slot_norm = last_dominant_grad_slot_norm;
    result.final_dominant_grad_slot_frac = last_dominant_grad_slot_frac;
    result.final_pss_predictor_grad_norm = last_pss_predictor_grad_norm;
    result.final_pss_predictor_grad_frac = last_pss_predictor_grad_frac;
    result.final_lss_feedback_skip_tail = last_lss_feedback_skip_tail;
    result.final_lss_feedback_residual_scale = last_lss_feedback_residual_scale;
    result.final_lss_aux = last_lss_aux;
    result.final_lss_recon_norm = last_lss_recon_norm;
    result.final_lss_target_norm = last_lss_target_norm;
    result.final_lss_relative_rmse = last_lss_relative_rmse;
    result.final_support_transition_scale = last_support_transition_scale;
    result.final_pss_slot_clip_steps = pss_slot_clip_fired_steps;
    result.final_pss_spike_ratio_max = last_pss_spike_ratio_max;
    result.final_pss_spike_slot = last_pss_spike_slot;
    result.final_pss_confidence = last_pss_confidence;
    result.final_pss_pred_err = last_pss_pred_err;
    result.final_pss_int2_agreement = last_pss_int2_agreement;
    result.final_pss_engaged_frac = sb.pss_engaged_frac;
    result.final_pss_covered = last_pss_covered;
    result.final_pss_n = last_pss_n;
    result.final_pss_int2_matched = last_pss_int2_matched;
    result.final_pss_int2_scored = last_pss_int2_scored;
    result.final_pss_scored_micros = last_pss_scored_micros;
    result.final_pss_int2_inv_rms = last_pss_int2_inv_rms;
    result.final_pss_confidence_min = last_pss_confidence_min;
    result.final_pss_confidence_max = last_pss_confidence_max;
    result.final_pss_blend_delta_rms = last_pss_blend_delta_rms;
    result.final_pss_aux_weight = last_pss_aux_weight;
    result.final_pss_aux_denom = last_pss_aux_denom;
    result.final_pss_aux_normalize = last_pss_aux_normalize;
    result.final_pss_conditioning_active = last_pss_conditioning_active;
    result.final_pss_override_active = last_pss_override_active;
    result.final_pss_effective_confidence = last_pss_effective_confidence;
    result.final_pss_governor_event = last_pss_governor_event;
    result.final_pss_conditioning_mode = pss_conditioning_mode;
    result.final_pss_aux_normalize_mode = pss_aux_normalize_mode;
    result.fp8_active        = fp8_was_on;
    result.attention_backend = attention_backend_name(attention_backend);
    result.precision_profile = request.precision_profile;
    attn_bwd_health_read(&result.attn_p_zero_frac, &result.attn_lse_drift_max,
                         arena.stream);
    return result;
}

BurnResult run_lattice_training_model_parallel(
    const NativeRequest& request,
    NativeArena& first_stage,
    NativeArena& second_stage,
    ProgressCallback on_step
) {
    const auto& devices = request.device.model_parallel_devices;
    if (devices.size() != 2 || devices[0] == devices[1]) {
        throw std::runtime_error(
            "native model parallel requires exactly two distinct device ordinals");
    }
    if (first_stage.device_id != devices[0] || second_stage.device_id != devices[1]) {
        throw std::runtime_error(
            "native model-parallel arena devices do not match the accepted request");
    }
    const char* pss_enabled = std::getenv("IDA_NATIVE_PSS_PRED");
    if ((request.pss_pred_rank > 0 || (pss_enabled && pss_enabled[0] == '1')) &&
        pss_state_persistence_requested()) {
        throw std::runtime_error(
            "native model-parallel execution cannot persist the standalone PSS predictor; "
            "pin the PSS resume probe to a single GPU");
    }
    if (!request.parent.resume_from_checkpoint.empty() ||
        !request.pss_pred_state_init_path.empty() ||
        !request.pss_pred_state_out.empty()) {
        throw std::runtime_error(
            "native model-parallel execution supports parent-weight lineage but "
            "not exact resume or standalone PSS state persistence");
    }
    const char* subbatch_env = std::getenv("IDA_NATIVE_SUBBATCH_STREAMS");
    if (subbatch_env && std::atoi(subbatch_env) > 1) {
        throw std::runtime_error(
            "native model-parallel execution does not compose with subbatch "
            "stream crossover; set IDA_NATIVE_SUBBATCH_STREAMS=1");
    }
    if (act_row_clip_mode() == 1 || rowclip_debug_enabled() || fp8_clip_debug_enabled()) {
        throw std::runtime_error(
            "native model-parallel execution rejects process-global adaptive or "
            "debug clip state; use the fixed clip path for the initial canary");
    }
    if (trust_ratio_enabled(request)) {
        throw std::runtime_error(
            "native model-parallel execution requires Lion trust ratio off until "
            "per-stage EMA state is included in the peer-pipeline receipt");
    }
    const bool pss_slot_clip = [] {
        const char* e = std::getenv("IDA_NATIVE_PSS_SLOT_CLIP");
        return e && e[0] == '1';
    }();
    if (pss_slot_clip) {
        throw std::runtime_error(
            "native model-parallel execution rejects PSS slot clip until the "
            "per-stage slot-EMA evidence is merged into one receipt");
    }
    if (const char* e = std::getenv("IDA_NATIVE_PSS_SPIKE_JOINT"); e && e[0] == '1') {
        throw std::runtime_error(
            "native model-parallel execution rejects the PSS spike joint until "
            "its feature buckets include both pipeline stages");
    }

    const std::string requested_peer_transport =
        request.device.peer_transport.empty() ? "auto" : request.device.peer_transport;
    if (requested_peer_transport != "auto" &&
        requested_peer_transport != "cuda_p2p" &&
        requested_peer_transport != "host_staged") {
        throw std::runtime_error(
            "native model-parallel peer_transport must be auto, cuda_p2p, or host_staged");
    }
    std::string peer_transport_used = requested_peer_transport;
    if (requested_peer_transport == "host_staged") {
        std::fprintf(stderr,
            "[ida_native_train] peer-pipeline: using explicit host-staged "
            "boundary transport (NVLink/CUDA P2P not required)\n");
    } else {
        try {
            enable_bidirectional_cuda_peer_access(first_stage.device_id, second_stage.device_id);
            peer_transport_used = "cuda_p2p";
        } catch (const std::exception& peer_enable_error) {
            if (requested_peer_transport == "cuda_p2p") {
                throw std::runtime_error(
                    std::string("native model-parallel peer_transport=cuda_p2p unavailable: ") +
                    peer_enable_error.what());
            }
            peer_transport_used = "host_staged";
            std::fprintf(stderr,
                "[ida_native_train] peer-pipeline: %s -- using "
                "host-staged boundary copies (slower, correctness-preserving)\n",
                peer_enable_error.what());
        }
    }
    const auto attention_backend = parse_attention_backend(request.attention_backend);
    validate_attention_request(request, attention_backend);
    validate_model_contract_request(request, attention_backend);
    validate_precision_state_request(request);
    set_attn_window_request_override(
        request.model.local_attention_window > 0 ? request.model.local_attention_window : -1);

    const int global_layers = request.model.layers;
    const int split_layer = request.device.pipeline_split_layer > 0
        ? request.device.pipeline_split_layer : global_layers / 2;
    if (split_layer <= 0 || split_layer >= global_layers) {
        throw std::runtime_error(
            "native model-parallel pipeline_split_layer must be inside the model depth");
    }

    HostDataset ds = load_host_dataset(request);
    const int S = ds.seq_len;
    int mb = request.input.batch_size > 0 ? request.input.batch_size
           : request.training.microbatch > 0 ? request.training.microbatch : 32;
    mb = std::max(1, std::min(mb, std::max(1, ds.num_sequences)));
    const int accum_ceiling = request.training.grad_accumulation > 0
        ? request.training.grad_accumulation : 4;
    const bool fixed_accumulation = request.grad_accum_override > 0;
    int total_micro = request.training.max_steps > 0
        ? request.training.max_steps : std::max(1, (ds.num_sequences / mb) * 2);
    if (request.max_samples > 0) {
        const int cap_accum = std::max(1, accum_ceiling);
        const int capped = ((request.max_samples + cap_accum - 1) / cap_accum) * cap_accum;
        total_micro = std::min(total_micro, capped);
    }
    const auto accum_at = [&](int micro_idx) {
        if (fixed_accumulation) return std::max(1, accum_ceiling);
        static const double thresholds[] = {0.075, 0.15, 0.20, 0.25, 0.30};
        const double frac = static_cast<double>(micro_idx) / std::max(1, total_micro);
        int accum = 1;
        for (double threshold : thresholds) {
            if (frac >= threshold && accum < accum_ceiling) accum <<= 1;
        }
        return std::min(accum, std::max(1, accum_ceiling));
    };
    int total_opt = 0;
    for (int micro = 0; micro < total_micro; ++total_opt) micro += accum_at(micro);
    total_opt = std::max(1, total_opt);

    const double base_lr = request.training.learning_rate > 0.0
        ? request.training.learning_rate : 3e-4;
    const float clip = [&request] {
        if (request.global_clip_override >= 0.0f) {
            return request.global_clip_override > 0.0f
                ? request.global_clip_override : std::numeric_limits<float>::max();
        }
        const char* e = std::getenv("IDA_NATIVE_GLOBAL_CLIP");
        if (e && e[0]) {
            const float parsed = static_cast<float>(std::atof(e));
            return parsed > 0.0f ? parsed : std::numeric_limits<float>::max();
        }
        return 1.0f;
    }();
    float grad_norm_abs_ceiling = 1e8f;
    if (const char* e = std::getenv("IDA_NATIVE_GRAD_NORM_ABS_CEILING")) {
        try { grad_norm_abs_ceiling = std::stof(e); } catch (...) {}
    }
    const bool use_lion = optimizer_uses_lion(request);
    const float lion_b1 = lion_beta1(request);
    const float lion_b2 = lion_beta2(request);
    const float lion_lr_mult = lion_lr_scale(request);
    const float lion_wd_mult = lion_wd_scale(request);
    const float adam_b1 = 0.9f, adam_b2 = 0.999f, adam_eps = 1e-8f;
    const bool lr_accum_coupling = [] {
        const char* e = std::getenv("IDA_NATIVE_LR_ACCUM_COUPLING");
        return !e || e[0] == '1';
    }();
    const bool lrss_freeze = [] {
        const char* e = std::getenv("IDA_NATIVE_LRSS_FREEZE");
        return e && e[0] == '1';
    }();
    const float lrss_lr_scale = [] {
        const char* e = std::getenv("IDA_NATIVE_LRSS_LR_SCALE");
        return e ? std::max(0.0f, static_cast<float>(std::atof(e))) : 1.0f;
    }();
    const float pss_pred_lr_scale = request.pss_pred_lr_scale >= 0.0f
        ? request.pss_pred_lr_scale
        : [] {
              const char* e = std::getenv("IDA_NATIVE_PSS_PRED_LR_SCALE");
              return (e && e[0]) ? std::max(0.0f, static_cast<float>(std::atof(e))) : 1.0f;
          }();
    unsigned lrss_update_mask = [] {
        const char* e = std::getenv("IDA_NATIVE_LRSS_UPDATE_MASK");
        return e ? static_cast<unsigned>(std::strtoul(e, nullptr, 0)) : 0x0fu;
    }();
    const bool optimizer_sr_seed_mix = [] {
        const char* e = std::getenv("IDA_NATIVE_SR_SEED_MIX");
        return e && e[0] == '1';
    }();

    auto set_device = [](const NativeArena& arena) {
        IDA_CUDA_CHECK(cudaSetDevice(arena.device_id));
    };
    auto configure_cublas = [](NativeArena& arena) {
        cublasHandle_t handle{};
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetStream(handle, arena.stream));
        const char* e = std::getenv("IDA_CUBLAS_PEDANTIC");
        CUBLAS_CHECK(cublasSetMathMode(
            handle, (e && e[0] == '0') ? CUBLAS_DEFAULT_MATH : CUBLAS_PEDANTIC_MATH));
        return handle;
    };

    LatticeWeights w0{}, w1{};
    LatticeOptState opt0{}, opt1{};
    LatticeGrads g0{}, g1{};
    StepBuffers sb0{}, sb1{};
    Fp8Ctx f80{}, f81{};
    PackedFp4AttentionCtx fp40{}, fp41{};
    AmperePackedWeights ampere0{}, ampere1{};
    cublasHandle_t cublas0{}, cublas1{};
    uint32_t* tokens0{}; uint32_t* tokens1{};
    int32_t* labels1{};
    uint16_t* segs0{}; uint16_t* segs1{};
    float* normsq0{}; float* normsq1{};
    // ── 1F1B second slot for stage0 only ─────────────────────────────────
    // Stage1 (dev1) stays single-buffered: its own stream already processes
    // microbatches strictly in order with no gaps to fill, and its LRSS
    // anchor bank / FP8 ActSlot delayed-scaling state is only ever touched
    // by that one in-order stream. Stage0 (dev0) is the one that needs a
    // second live instance so its forward for microbatch i+1 can be issued
    // (and actually run, concurrently with dev1's fwd+bwd on microbatch i)
    // before its own backward for microbatch i has consumed microbatch i's
    // saved-activation ring / FP8 ActSlot scale_snapshot. Duplicating via
    // the same allocators used for the primary slot (alloc_step_buffers /
    // build_fp8_ctx / build_packed_fp4_attention_ctx) rather than hand-
    // copying StepBuffers' ~80 fields, so nothing gets missed. The FP8
    // WEIGHT-quantization cache inside f80b is redundant (weights only
    // change at the optimizer-step boundary, refreshed on both instances
    // there) -- only its ACTIVATION ActSlots are load-bearing here, but
    // duplicating the whole ctx is what the allocator gives us.
    StepBuffers sb0b{};
    Fp8Ctx f80b{};
    PackedFp4AttentionCtx fp40b{};
    uint32_t* tokens0b{};
    uint16_t* segs0b{};
    cudaEvent_t ev_fwd0_ready[2]{};
    cudaEvent_t ev_bwd1_ready[2]{};
    LrssScratch* lrss1{};
    LrssParams lrss_params1{};
    LrssGrads lrss_grads1{};

    set_device(first_stage);
    w0 = allocate_lattice_weights_shard(
        request, first_stage, LatticeShardSpec{0, split_layer, true, false});
    ampere0 = build_ampere_packed_weights(request, w0, first_stage);
    g_ampere_packed_weights = &ampere0;
    opt0 = allocate_lattice_opt(request, w0, first_stage);
    g0 = allocate_lattice_grads(w0, first_stage);
    auto slots0 = build_param_slots(w0, opt0, g0, 0.01f);
    sb0 = alloc_step_buffers(w0, mb, S, first_stage);
    IDA_CUDA_CHECK(ida_malloc_async(&tokens0, static_cast<std::size_t>(mb) * S * sizeof(uint32_t),
                                    first_stage.pool, first_stage.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&segs0, static_cast<std::size_t>(mb) * S * sizeof(uint16_t),
                                    first_stage.pool, first_stage.stream));
    sb0.segs = segs0;
    cublas0 = configure_cublas(first_stage);
    const std::size_t max_act0 = std::max({
        static_cast<std::size_t>(mb) * S * 3 * w0.hidden_size,
        static_cast<std::size_t>(mb) * S * w0.intermediate_size,
        static_cast<std::size_t>(sb0.ce_chunk) * w0.vocab_size});
    f80 = build_fp8_ctx(request, w0, max_act0, mb * S, first_stage);
    fp40 = build_packed_fp4_attention_ctx(request, sb0, w0.num_layers, first_stage);
    bind_fp8_weight_amax_slots(slots0, f80, w0);
    refresh_fp8_weights(f80, first_stage.stream, /*recompute_amax=*/true);
    normsq0 = alloc_f32(std::max<std::size_t>(1, slots0.size()), first_stage);

    const bool peer_1f1b = [] {
        // Default OFF (2026-08-13): confirmed via a clean standalone repro
        // that duplicating stage0's StepBuffers/Fp8Ctx/PackedFp4AttentionCtx
        // for real cross-GPU overlap costs more VRAM at full production
        // scale (mb=64) than this box's ~27GB of headroom past the base
        // ~53GB single-buffered footprint -- OOMs at trainer.cu's normsq
        // alloc, solo, zero contention. Confirmed correctness-safe and
        // OOM-free with IDA_NATIVE_PEER_1F1B=1 passed directly to the
        // binary; the queue's own env-propagation chain was NOT reliably
        // forwarding a newly added queue_overrides.env var to the spawned
        // subprocess (separate, still-open issue) -- flipping the default
        // here needs no env var and can't be defeated by that gap.
        const char* e = std::getenv("IDA_NATIVE_PEER_1F1B");
        return e && e[0] == '1';
    }();
    if (peer_1f1b) {
        sb0b = alloc_step_buffers(w0, mb, S, first_stage);
        IDA_CUDA_CHECK(ida_malloc_async(&tokens0b,
            static_cast<std::size_t>(mb) * S * sizeof(uint32_t),
            first_stage.pool, first_stage.stream));
        IDA_CUDA_CHECK(ida_malloc_async(&segs0b,
            static_cast<std::size_t>(mb) * S * sizeof(uint16_t),
            first_stage.pool, first_stage.stream));
        sb0b.segs = segs0b;
        f80b = build_fp8_ctx(request, w0, max_act0, mb * S, first_stage);
        fp40b = build_packed_fp4_attention_ctx(request, sb0b, w0.num_layers, first_stage);
        // Weight-side quantization only: no optimizer slot binding for f80b
        // (bind_fp8_weight_amax_slots wires the *single* slots0/optimizer
        // path, which must stay pointed at f80's amax buffers, not f80b's --
        // see update_stage's fp8_amax usage below). f80b's activation
        // ActSlots are what this second buffer set actually needs; its
        // weight cache is refreshed in lockstep with f80's at each
        // optimizer-step boundary purely so forward reads a consistent
        // quantized weight, never read via slots0.
        refresh_fp8_weights(f80b, first_stage.stream, /*recompute_amax=*/true);
        IDA_CUDA_CHECK(cudaEventCreateWithFlags(&ev_fwd0_ready[0], cudaEventDisableTiming));
        IDA_CUDA_CHECK(cudaEventCreateWithFlags(&ev_fwd0_ready[1], cudaEventDisableTiming));
    }

    set_device(second_stage);
    if (peer_1f1b) {
        // Must be created with second_stage current: this pair is recorded
        // on second_stage.stream (below) and, without CUDA P2P (consumer
        // Blackwell -- see cuda_peer_copy_async's comment in arena.cu),
        // cudaEventRecord against a stream on a device other than the one
        // active at cudaEventCreate time fails with "invalid resource
        // handle". ev_fwd0_ready above has no such mismatch: it is created
        // on first_stage and always recorded on first_stage.stream.
        IDA_CUDA_CHECK(cudaEventCreateWithFlags(&ev_bwd1_ready[0], cudaEventDisableTiming));
        IDA_CUDA_CHECK(cudaEventCreateWithFlags(&ev_bwd1_ready[1], cudaEventDisableTiming));
    }
    w1 = allocate_lattice_weights_shard(
        request, second_stage, LatticeShardSpec{split_layer, global_layers, false, true});
    ampere1 = build_ampere_packed_weights(request, w1, second_stage);
    g_ampere_packed_weights = &ampere1;
    opt1 = allocate_lattice_opt(request, w1, second_stage);
    g1 = allocate_lattice_grads(w1, second_stage);
    auto slots1 = build_param_slots(w1, opt1, g1, 0.01f);
    sb1 = alloc_step_buffers(w1, mb, S, second_stage);
    IDA_CUDA_CHECK(ida_malloc_async(&tokens1, static_cast<std::size_t>(mb) * S * sizeof(uint32_t),
                                    second_stage.pool, second_stage.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&labels1, static_cast<std::size_t>(mb) * S * sizeof(int32_t),
                                    second_stage.pool, second_stage.stream));
    IDA_CUDA_CHECK(ida_malloc_async(&segs1, static_cast<std::size_t>(mb) * S * sizeof(uint16_t),
                                    second_stage.pool, second_stage.stream));
    sb1.segs = segs1;
    cublas1 = configure_cublas(second_stage);
    const std::size_t max_act1 = std::max({
        static_cast<std::size_t>(mb) * S * 3 * w1.hidden_size,
        static_cast<std::size_t>(mb) * S * w1.intermediate_size,
        static_cast<std::size_t>(sb1.ce_chunk) * w1.vocab_size});
    f81 = build_fp8_ctx(request, w1, max_act1, mb * S, second_stage);
    fp41 = build_packed_fp4_attention_ctx(request, sb1, w1.num_layers, second_stage);
    bind_fp8_weight_amax_slots(slots1, f81, w1);
    refresh_fp8_weights(f81, second_stage.stream, /*recompute_amax=*/true);
    normsq1 = alloc_f32(std::max<std::size_t>(1, slots1.size()), second_stage);

    if (w1.lrss_enabled) {
        lrss_params1.query_w = w1.lrss_query;   lrss_params1.key_w = w1.lrss_key;
        lrss_params1.gate_w = w1.lrss_gate_w;    lrss_params1.gate_b = w1.lrss_gate_b;
        lrss_params1.log_tau = w1.lrss_log_tau;  lrss_params1.scale_w = w1.lrss_scale_w;
        lrss_params1.num_scales = w1.lrss_scales;
        lrss_grads1.query_w = g1.g_lrss_query;   lrss_grads1.key_w = g1.g_lrss_key;
        lrss_grads1.gate_w = g1.g_lrss_gate_w;   lrss_grads1.gate_b = g1.g_lrss_gate_b;
        lrss_grads1.log_tau = g1.g_lrss_log_tau; lrss_grads1.scale_w = g1.g_lrss_scale_w;
        if (w1.lss_rank > 0) {
            lrss_params1.lss_down = w1.lss_down; lrss_params1.lss_up = w1.lss_up;
            lrss_params1.lss_rank = w1.lss_rank;
            lrss_params1.pss_spike_dim = w1.pss_spike_joint_dim;
            lrss_grads1.lss_down = g1.g_lss_down; lrss_grads1.lss_up = g1.g_lss_up;
            if (const char* e = std::getenv("IDA_NATIVE_LSS_AUX_WEIGHT")) {
                lrss_params1.lss_aux_weight = std::atof(e);
            }
        }
        lrss1 = alloc_lrss_scratch(w1, mb, second_stage);
        sb1.lrss_p = &lrss_params1;
        sb1.lrss_s = lrss1;
        sb1.lrss_g = &lrss_grads1;
    }
    if (w1.pss_pred_rank > 0) lrss_update_mask |= 0x20u;
    // Keep the unscaled auxiliary coefficient immutable. The support
    // controller applies its transition factor at the same optimizer-step
    // boundary as the single-device runner, then the next window consumes it.
    const float lss_aux_weight_base = lrss_params1.lss_aux_weight;

    {
        ::ida_native::ontology::Context octx;
        octx.microbatch = mb;
        octx.grad_accum = accum_ceiling;
        octx.num_experts = request.model.num_personality_experts;
        octx.top_k_experts = request.model.top_k_experts;
        octx.seq_window = request.model.local_attention_window;
        octx.precision_profile = request.precision_profile;
        octx.attention_backend = attention_backend_name(attention_backend);
        octx.optimizer = request.optimizer_type;
        octx.body_key = request.family + "/" + request.seat + "/" + request.version;
        octx.resident_bodies = 1;
        ::ida_native::ontology::set_context(octx);
    }
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
    if (request.ontology_required &&
        (!::ida_native::ontology::enabled() || !::ida_native::ontology::sink_is_file())) {
        throw std::runtime_error(
            "ontology_required but the model-parallel trainer could not open the ontology sink");
    }
#endif

#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
    std::ofstream shared_metrics;
    std::ofstream burn_metrics;
    {
        const auto metrics_path = request.repo_root / "artifacts" / "telemetry" /
                                  "native_training_metrics.jsonl";
        std::error_code ec;
        std::filesystem::create_directories(metrics_path.parent_path(), ec);
        shared_metrics.open(metrics_path, std::ios::app);
    }
    {
        std::error_code ec;
        std::filesystem::create_directories(request.output_dir, ec);
        const auto metrics_path = request.analytics_path.empty()
            ? request.output_dir / "training_metrics.jsonl"
            : request.analytics_path;
        burn_metrics.open(metrics_path, std::ios::trunc);
    }
    if (request.analytics_required &&
        (!shared_metrics.is_open() || !burn_metrics.is_open())) {
        throw std::runtime_error("analytics_required but model-parallel metrics streams could not be opened");
    }
#endif
    set_device(first_stage);
    attn_bwd_health_reset(first_stage.stream);
    set_device(second_stage);
    attn_bwd_health_reset(second_stage.stream);
    const auto started = std::chrono::steady_clock::now();
    const std::size_t weight_bytes = w0.total_bytes + w1.total_bytes;
    std::size_t peer_forward_bytes = 0;
    std::size_t peer_backward_bytes = 0;
    std::size_t total_tokens = 0;
    std::size_t examples_seen = 0;
    int micro_done = 0;
    int opt_step = 0;
    int skipped = 0;
    int seq_cursor = 0;
    float loss_ema = -1.0f;
    float last_loss = 0.0f;
    float last_grad_norm = 0.0f;
    float last_lr = 0.0f;
    bool last_clip_fired = false;
    float last_clip_scale = 1.0f;
    bool last_embed_clip_fired = false;
    float last_embed_clip_max = 0.0f;
    int last_embed_clip_rows = 0;
    int clip_steps = 0;
    int embed_clip_steps = 0;
    int embed_clip_rows_total = 0;
    std::string dominant_slot;
    float dominant_slot_norm = 0.0f;
    float dominant_slot_frac = 0.0f;
    // Mirrors last_pss_predictor_grad_norm/_frac in run_lattice_training --
    // see that computation site's comment. Model-parallel splits slots
    // across two stages (names0/host_norms0, names1/host_norms1), so both
    // must be searched.
    float pss_predictor_grad_norm = 0.0f;
    float pss_predictor_grad_frac = 0.0f;
    float last_lss_aux = 0.0f;
    int last_lss_skip_tail = 0;
    float last_lss_residual_scale = 1.0f;
    float last_lss_recon_norm = 0.0f;
    float last_lss_target_norm = 0.0f;
    float last_lss_relative_rmse = 0.0f;
    float prev_lss_target_norm = 0.0f;
    float prev_lss_relative_rmse = 0.0f;
    int lss_rmse_non_improve_windows = 0;
    float last_pss_confidence = 0.0f;
    float last_pss_err = 0.0f;
    float last_pss_int2 = 0.0f;
    std::uint32_t last_pss_covered = 0;
    std::size_t last_pss_n = 0;
    std::uint32_t last_pss_int2_matched = 0;
    std::uint32_t last_pss_int2_scored = 0;
    int last_pss_scored_micros = 0;
    float last_pss_int2_inv_rms = 0.0f;
    float last_pss_confidence_min = 0.0f;
    float last_pss_confidence_max = 0.0f;
    float last_pss_blend_delta_rms = 0.0f;
    float last_pss_aux_weight = 0.0f;
    float last_pss_aux_denom = 0.0f;
    bool last_pss_aux_normalize = false;
    bool last_pss_conditioning_active = false;
    bool last_pss_override_active = false;
    float last_pss_effective_confidence = 0.0f;
    std::string last_pss_governor_event{"none"};
    float support_transition_scale = 1.0f;
    const float embed_clip_threshold = embed_row_clip_threshold();
    const std::string pss_conditioning_mode =
        pss_policy_mode(request.pss_conditioning_mode, "IDA_NATIVE_PSS_CONDITIONING");
    const std::string pss_aux_normalize_mode =
        pss_policy_mode(request.pss_aux_normalize_mode, "IDA_NATIVE_PSS_AUX_NORMALIZE");
    const auto names0 = build_param_slot_names(w0);
    const auto names1 = build_param_slot_names(w1);

    auto collect_norms = [&](NativeArena& arena, StepBuffers& sb,
                             const std::vector<ParamSlot>& slots, float* d_slot_normsq,
                             std::vector<float>& host) {
        set_device(arena);
        IDA_CUDA_CHECK(cudaMemsetAsync(d_slot_normsq, 0, slots.size() * sizeof(float), arena.stream));
        IDA_CUDA_CHECK(cudaMemsetAsync(sb.norm_acc, 0, sizeof(float), arena.stream));
        for (std::size_t i = 0; i < slots.size(); ++i) {
            slot_sq_sum_acc(slots[i], d_slot_normsq + i, arena.stream);
            slot_sq_sum_acc(slots[i], sb.norm_acc, arena.stream);
        }
        IDA_CUDA_CHECK(cudaStreamSynchronize(arena.stream));
        float sq = 0.0f;
        IDA_CUDA_CHECK(cudaMemcpy(&sq, sb.norm_acc, sizeof(float), cudaMemcpyDeviceToHost));
        host.resize(slots.size());
        if (!host.empty()) {
            IDA_CUDA_CHECK(cudaMemcpy(host.data(), d_slot_normsq,
                                      host.size() * sizeof(float), cudaMemcpyDeviceToHost));
        }
        return sq;
    };
    auto scale_slots = [&](NativeArena& arena, const std::vector<ParamSlot>& slots, float scale) {
        set_device(arena);
        for (const auto& slot : slots) {
            slot_scale_gradient(slot, scale, arena.stream);
        }
    };
    auto zero_slots = [&](NativeArena& arena, LatticeGrads& grads, const LatticeWeights& weights) {
        set_device(arena);
        zero_lattice_grads(grads, weights, arena.stream);
    };
    auto is_lrss_weight = [](const ParamSlot& slot, const LatticeWeights& weights) {
        return weights.lrss_enabled &&
            (slot.w == weights.lrss_query || slot.w == weights.lrss_key ||
             slot.w == weights.lrss_gate_w || slot.w == weights.lrss_gate_b ||
             slot.w == weights.lrss_log_tau || slot.w == weights.lrss_scale_w ||
             slot.w == weights.lss_down || slot.w == weights.lss_up);
    };
    auto is_pss_weight = [](const ParamSlot& slot, const LatticeWeights& weights) {
        return slot.w == weights.pss_pred_down || slot.w == weights.pss_pred_up;
    };
    auto support_group = [](const ParamSlot& slot, const LatticeWeights& weights) -> unsigned {
        if (slot.w == weights.lrss_query || slot.w == weights.lrss_key) return 0x01u;
        if (slot.w == weights.lrss_gate_w) return 0x02u;
        if (slot.w == weights.lrss_log_tau || slot.w == weights.lrss_scale_w) return 0x04u;
        if (slot.w == weights.lss_down || slot.w == weights.lss_up) return 0x08u;
        if (slot.w == weights.lrss_gate_b) return 0x10u;
        if (slot.w == weights.pss_pred_down || slot.w == weights.pss_pred_up) return 0x20u;
        return 0u;
    };
    auto update_stage = [&](NativeArena& arena, LatticeWeights& weights, Fp8Ctx& f8,
                            AmperePackedWeights& ampere,
                            StepBuffers& sb,
                            std::vector<ParamSlot>& slots,
                            const std::vector<std::string>& names, int seed_offset,
                            int optimizer_step, int skipped_steps, float lr) {
        set_device(arena);
        g_ampere_packed_weights = &ampere;
        for (std::size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
            auto& slot = slots[slot_idx];
            if (lrss_freeze && is_lrss_weight(slot, weights)) continue;
            const unsigned group = support_group(slot, weights);
            if (group != 0u && (lrss_update_mask & group) == 0u) continue;
            const float slot_lr = lr *
                (is_lrss_weight(slot, weights) ? lrss_lr_scale :
                 is_pss_weight(slot, weights) ? pss_pred_lr_scale : 1.0f) *
                ((is_lrss_weight(slot, weights) || is_pss_weight(slot, weights))
                    ? support_transition_scale : 1.0f);
            const unsigned sr_seed = optimizer_sr_seed_mix
                ? (static_cast<unsigned>(optimizer_step) * 0x9e3779b9u) ^
                  (static_cast<unsigned>(seed_offset + static_cast<int>(slot_idx) + 1) * 0x85ebca6bu)
                : 0u;
            if (slot.fp8_amax) {
                IDA_CUDA_CHECK(cudaMemsetAsync(slot.fp8_amax, 0, sizeof(float), arena.stream));
            }
            const bool have_slot_name = slot_idx < names.size();
            if (have_slot_name) nvtxRangePushA(names[slot_idx].c_str());
            const std::size_t chunk_capacity = slot.g_bf16
                ? sb.grad_update_scratch_n : slot.n;
            if (chunk_capacity == 0) {
                throw std::runtime_error("empty optimizer staging capacity");
            }
            for (std::size_t offset = 0; offset < slot.n; offset += chunk_capacity) {
                const std::size_t chunk_n = std::min(chunk_capacity, slot.n - offset);
                float* optimizer_grad = slot.g ? slot.g + offset : sb.grad_update_scratch;
                if (slot.g_bf16) {
                    cast_bf16_to_f32(slot.g_bf16 + offset, sb.grad_update_scratch,
                                     chunk_n, arena.stream);
                    optimizer_grad = sb.grad_update_scratch;
                }
                if (use_lion) {
                    if (slot.m.bf16) {
                        lion_step(slot.w + offset, slot.m.bf16 + offset, optimizer_grad, chunk_n,
                                  slot_lr * lion_lr_mult, lion_b1, lion_b2,
                                  slot.wd * lion_wd_mult, arena.stream, slot.fp8_amax, sr_seed);
                    } else {
                        lion_step(slot.w + offset, slot.m.f32 + offset, optimizer_grad, chunk_n,
                                  slot_lr * lion_lr_mult, lion_b1, lion_b2,
                                  slot.wd * lion_wd_mult, arena.stream, slot.fp8_amax, sr_seed);
                    }
                }
#if IDA_NATIVE_ENABLE_ADAMW
                else if (slot.m.bf16) {
                    adamw_step(slot.w + offset, slot.m.bf16 + offset,
                               slot.v.bf16 + offset, optimizer_grad, chunk_n,
                               slot_lr, adam_b1, adam_b2, adam_eps, slot.wd,
                               optimizer_step - skipped_steps, arena.stream, slot.fp8_amax, sr_seed);
                } else {
                    adamw_step(slot.w + offset, slot.m.f32 + offset,
                               slot.v.f32 + offset, optimizer_grad, chunk_n,
                               slot_lr, adam_b1, adam_b2, adam_eps, slot.wd,
                               optimizer_step - skipped_steps, arena.stream, slot.fp8_amax, sr_seed);
                }
#else
                else {
                    throw std::runtime_error(
                        "Adam and AdamW are disabled in the native public runtime");
                }
#endif
            }
            if (have_slot_name) nvtxRangePop();
        }
        refresh_fp8_weights(f8, arena.stream, /*recompute_amax=*/false);
        refresh_ampere_packed_weights(ampere, arena.stream);
    };

    // 1F1B helpers: fill/steady-state/drain schedule for stage0's two
    // buffer slots. Stage1 (sb1/f81/fp41) stays single-buffered and is
    // always issued in strict, unchanged microbatch order -- only stage0's
    // forward for microbatch i+1 gets issued ahead of its own backward for
    // microbatch i, which is the entire overlap this buys.
    struct MicroWindow { std::size_t offset; std::size_t elements; int actual_mb; };
    auto compute_window = [&]() -> MicroWindow {
        int batch_start = seq_cursor % ds.num_sequences;
        int actual_mb = std::min(mb, ds.num_sequences - batch_start);
        if (actual_mb < mb && ds.num_sequences >= mb) {
            batch_start = 0;
            actual_mb = mb;
        }
        seq_cursor = (batch_start + actual_mb) % ds.num_sequences;
        return MicroWindow{static_cast<std::size_t>(batch_start) * S,
                           static_cast<std::size_t>(actual_mb) * S, actual_mb};
    };
    auto issue_stage0_forward = [&](int slot, const MicroWindow& win) {
        StepBuffers& sb0cur = (slot == 0) ? sb0 : sb0b;
        Fp8Ctx& f80cur = (slot == 0) ? f80 : f80b;
        PackedFp4AttentionCtx& fp40cur = (slot == 0) ? fp40 : fp40b;
        uint32_t* tokens0cur = (slot == 0) ? tokens0 : tokens0b;
        uint16_t* segs0cur = (slot == 0) ? segs0 : segs0b;
        sb0cur.B = win.actual_mb;
        set_device(first_stage);
        g_ampere_packed_weights = &ampere0;
        IDA_CUDA_CHECK(cudaMemcpyAsync(tokens0cur, ds.tokens.data() + win.offset,
                                        win.elements * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                        first_stage.stream));
        IDA_CUDA_CHECK(cudaMemcpyAsync(segs0cur, ds.segs.data() + win.offset,
                                        win.elements * sizeof(uint16_t), cudaMemcpyHostToDevice,
                                        first_stage.stream));
        forward(cublas0, attention_backend, f80cur, fp40cur, request, w0, tokens0cur, sb0cur,
                first_stage, /*run_embedding=*/true, /*run_output=*/false);
        IDA_CUDA_CHECK(cudaEventRecord(ev_fwd0_ready[slot], first_stage.stream));
    };
    auto issue_stage1_step = [&](int slot, const MicroWindow& win) -> std::size_t {
        StepBuffers& sb0cur = (slot == 0) ? sb0 : sb0b;
        set_device(second_stage);
        g_ampere_packed_weights = &ampere1;
        IDA_CUDA_CHECK(cudaStreamWaitEvent(second_stage.stream, ev_fwd0_ready[slot], 0));
        sb1.B = win.actual_mb;
        const std::size_t boundary_bytes = win.elements *
            static_cast<std::size_t>(w0.hidden_size) * sizeof(__nv_bfloat16);
        cuda_peer_copy_async(sb1.hidden, second_stage.device_id, sb0cur.hidden,
                             first_stage.device_id, boundary_bytes, second_stage.stream,
                             requested_peer_transport == "host_staged");
        peer_forward_bytes += boundary_bytes;
        IDA_CUDA_CHECK(cudaMemcpyAsync(tokens1, ds.tokens.data() + win.offset,
                                        win.elements * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                        second_stage.stream));
        IDA_CUDA_CHECK(cudaMemcpyAsync(labels1, ds.labels.data() + win.offset,
                                        win.elements * sizeof(int32_t), cudaMemcpyHostToDevice,
                                        second_stage.stream));
        IDA_CUDA_CHECK(cudaMemcpyAsync(segs1, ds.segs.data() + win.offset,
                                        win.elements * sizeof(uint16_t), cudaMemcpyHostToDevice,
                                        second_stage.stream));
        if (lrss1) {
            lrss_refresh_bank(*lrss1, w1.lrss_anchors, w1.hidden_size,
                              micro_done + 1, second_stage.stream);
        }
        forward(cublas1, attention_backend, f81, fp41, request, w1, tokens1, sb1,
                second_stage, /*run_embedding=*/false, /*run_output=*/true);
        backward_accumulate(cublas1, attention_backend, f81, fp41, request, w1, g1,
                            tokens1, labels1, sb1, second_stage, pss_pred_aux_weight(request),
                            /*run_output=*/true, /*run_embedding=*/false);
        if (lrss1) {
            lrss_record_anchor(*lrss1, w1.lrss_anchors, w1.hidden_size,
                               micro_done + 1, second_stage.stream);
        }
        IDA_CUDA_CHECK(cudaEventRecord(ev_bwd1_ready[slot], second_stage.stream));
        return boundary_bytes;
    };
    auto issue_stage0_backward = [&](int slot, std::size_t boundary_bytes) {
        StepBuffers& sb0cur = (slot == 0) ? sb0 : sb0b;
        Fp8Ctx& f80cur = (slot == 0) ? f80 : f80b;
        PackedFp4AttentionCtx& fp40cur = (slot == 0) ? fp40 : fp40b;
        uint32_t* tokens0cur = (slot == 0) ? tokens0 : tokens0b;
        set_device(first_stage);
        g_ampere_packed_weights = &ampere0;
        IDA_CUDA_CHECK(cudaStreamWaitEvent(first_stage.stream, ev_bwd1_ready[slot], 0));
        cuda_peer_copy_async(sb0cur.d_hidden, first_stage.device_id, sb1.d_hidden,
                             second_stage.device_id, boundary_bytes, first_stage.stream,
                             requested_peer_transport == "host_staged");
        peer_backward_bytes += boundary_bytes;
        backward_accumulate(cublas0, attention_backend, f80cur, fp40cur, request, w0, g0,
                            tokens0cur, nullptr, sb0cur, first_stage, pss_pred_aux_weight(request),
                            /*run_output=*/false, /*run_embedding=*/true);
    };

    while (micro_done < total_micro) {
        ++opt_step;
        if (w1.lrss_enabled && w1.lss_rank > 0) {
            lrss_params1.lss_aux_weight = lss_aux_weight_base * support_transition_scale;
        }
        sb1.support_transition_scale = support_transition_scale;
        const auto step_wall_start = std::chrono::steady_clock::now();
        const int accum_now = accum_at(micro_done);
        auto reset_pss_window = [&set_device](StepBuffers& pss_sb, NativeArena& pss_arena) {
            pss_sb.pss_scored_micros = 0;
            pss_sb.pss_last_blend_frac = 0.0f;
            pss_sb.pss_aux_weight_used = 0.0f;
            pss_sb.pss_aux_normalize_used = false;
            pss_sb.pss_conditioning_active = false;
            if (pss_sb.pss_confidence_minmax) {
                set_device(pss_arena);
                k_pss_reset_confidence_minmax<<<1, 1, 0, pss_arena.stream>>>(
                    pss_sb.pss_confidence_minmax);
            }
        };
        reset_pss_window(sb0, first_stage);
        if (peer_1f1b) reset_pss_window(sb0b, first_stage);
        reset_pss_window(sb1, second_stage);
        set_device(second_stage);
        float loss_sum = 0.0f;
        int micros_this_step = 0;
        bool step_embed_clip_fired = false;
        float step_embed_clip_max = 0.0f;
        int step_embed_clip_rows = 0;

        if (peer_1f1b) {
            MicroWindow win_cur = compute_window();
            int slot_cur = 0;
            issue_stage0_forward(slot_cur, win_cur);

            for (int micro = 0; micro < accum_now && micro_done < total_micro; ++micro) {
                sb1.lss_feedback_completed_optimizer_steps = opt_step - 1;
                const std::size_t boundary_bytes = issue_stage1_step(slot_cur, win_cur);

                const bool has_next = (micro + 1 < accum_now) && (micro_done + 1 < total_micro);
                const int slot_next = slot_cur ^ 1;
                MicroWindow win_next{};
                if (has_next) {
                    win_next = compute_window();
                    issue_stage0_forward(slot_next, win_next);
                }

                // Host catches up with dev1's completion for THIS microbatch
                // to read back loss/embed-clip/LRSS stats. dev0 keeps
                // computing the next microbatch's forward in the background
                // (already enqueued above) while the host waits here -- the
                // actual cross-GPU overlap this whole rewrite is for.
                IDA_CUDA_CHECK(cudaEventSynchronize(ev_bwd1_ready[slot_cur]));
                // Current device context may still be first_stage (left over
                // from issue_stage0_forward above) -- sb1.loss lives on
                // second_stage. UVA makes the memcpy itself work either way,
                // but match this file's own always-set_device-first style
                // rather than lean on that.
                set_device(second_stage);
                float micro_loss = 0.0f;
                IDA_CUDA_CHECK(cudaMemcpy(&micro_loss, sb1.loss, sizeof(float), cudaMemcpyDeviceToHost));

                issue_stage0_backward(slot_cur, boundary_bytes);

                StepBuffers& sb0cur = (slot_cur == 0) ? sb0 : sb0b;
                if (embed_clip_threshold > 0.0f) {
                    float clip_stats[2] = {0.0f, 0.0f};
                    IDA_CUDA_CHECK(cudaMemcpy(clip_stats, sb0cur.embed_clip_stats, sizeof(clip_stats),
                                               cudaMemcpyDeviceToHost));
                    const int rows = static_cast<int>(clip_stats[0]);
                    if (rows > 0) {
                        step_embed_clip_fired = true;
                        step_embed_clip_rows += rows;
                        step_embed_clip_max = std::max(step_embed_clip_max, clip_stats[1]);
                    }
                }
                if (lrss1 && lrss1->lss_aux_copy_pending) {
                    lrss1->lss_aux_copy_pending = 0;
                    lrss1->lss_aux_valid = 1;
                }
                if (lrss1 && lrss1->lss_aux_valid) last_lss_aux = lrss1->lss_last_aux;
                last_lss_skip_tail = sb1.lss_feedback_skip_tail;
                last_lss_residual_scale = last_lss_skip_tail > 0
                    ? lss_feedback_residual_scale() : 1.0f;
                loss_sum += micro_loss;
                total_tokens += win_cur.elements;
                examples_seen += static_cast<std::size_t>(win_cur.actual_mb);
                ++micro_done;
                ++micros_this_step;

                slot_cur = slot_next;
                win_cur = win_next;
            }
            // Drain: the last microbatch's dev0 backward (issued above) must
            // fully land in g0 before collect_norms/update_stage read it
            // below -- this is the once-per-optimizer-step sync the
            // pre-1F1B loop paid on every single microbatch instead.
            IDA_CUDA_CHECK(cudaStreamSynchronize(first_stage.stream));
        } else {
        for (int micro = 0; micro < accum_now && micro_done < total_micro; ++micro) {
            int batch_start = seq_cursor % ds.num_sequences;
            int actual_mb = std::min(mb, ds.num_sequences - batch_start);
            if (actual_mb < mb && ds.num_sequences >= mb) {
                batch_start = 0;
                actual_mb = mb;
            }
            seq_cursor = (batch_start + actual_mb) % ds.num_sequences;
            const std::size_t offset = static_cast<std::size_t>(batch_start) * S;
            const std::size_t elements = static_cast<std::size_t>(actual_mb) * S;
            sb0.B = actual_mb;
            sb1.B = actual_mb;
            sb1.lss_feedback_completed_optimizer_steps = opt_step - 1;

            set_device(first_stage);
            g_ampere_packed_weights = &ampere0;
            IDA_CUDA_CHECK(cudaMemcpyAsync(tokens0, ds.tokens.data() + offset,
                                            elements * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                            first_stage.stream));
            IDA_CUDA_CHECK(cudaMemcpyAsync(segs0, ds.segs.data() + offset,
                                            elements * sizeof(uint16_t), cudaMemcpyHostToDevice,
                                            first_stage.stream));
            forward(cublas0, attention_backend, f80, fp40, request, w0, tokens0, sb0,
                    first_stage, /*run_embedding=*/true, /*run_output=*/false);
            IDA_CUDA_CHECK(cudaStreamSynchronize(first_stage.stream));

            const std::size_t boundary_bytes = elements * static_cast<std::size_t>(w0.hidden_size) *
                                               sizeof(__nv_bfloat16);
            cuda_peer_copy_async(sb1.hidden, second_stage.device_id, sb0.hidden,
                                 first_stage.device_id, boundary_bytes, second_stage.stream,
                                 requested_peer_transport == "host_staged");
            peer_forward_bytes += boundary_bytes;

            set_device(second_stage);
            g_ampere_packed_weights = &ampere1;
            IDA_CUDA_CHECK(cudaMemcpyAsync(tokens1, ds.tokens.data() + offset,
                                            elements * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                            second_stage.stream));
            IDA_CUDA_CHECK(cudaMemcpyAsync(labels1, ds.labels.data() + offset,
                                            elements * sizeof(int32_t), cudaMemcpyHostToDevice,
                                            second_stage.stream));
            IDA_CUDA_CHECK(cudaMemcpyAsync(segs1, ds.segs.data() + offset,
                                            elements * sizeof(uint16_t), cudaMemcpyHostToDevice,
                                            second_stage.stream));
            IDA_CUDA_CHECK(cudaStreamSynchronize(second_stage.stream));
            if (lrss1) {
                lrss_refresh_bank(*lrss1, w1.lrss_anchors, w1.hidden_size,
                                  micro_done + 1, second_stage.stream);
            }
            forward(cublas1, attention_backend, f81, fp41, request, w1, tokens1, sb1,
                    second_stage, /*run_embedding=*/false, /*run_output=*/true);
            backward_accumulate(cublas1, attention_backend, f81, fp41, request, w1, g1,
                                tokens1, labels1, sb1, second_stage, pss_pred_aux_weight(request),
                                /*run_output=*/true, /*run_embedding=*/false);
            if (lrss1) {
                lrss_record_anchor(*lrss1, w1.lrss_anchors, w1.hidden_size,
                                   micro_done + 1, second_stage.stream);
            }
            IDA_CUDA_CHECK(cudaStreamSynchronize(second_stage.stream));
            float micro_loss = 0.0f;
            IDA_CUDA_CHECK(cudaMemcpy(&micro_loss, sb1.loss, sizeof(float), cudaMemcpyDeviceToHost));

            cuda_peer_copy_async(sb0.d_hidden, first_stage.device_id, sb1.d_hidden,
                                 second_stage.device_id, boundary_bytes, first_stage.stream,
                                 requested_peer_transport == "host_staged");
            peer_backward_bytes += boundary_bytes;
            set_device(first_stage);
            IDA_CUDA_CHECK(cudaStreamSynchronize(first_stage.stream));
            backward_accumulate(cublas0, attention_backend, f80, fp40, request, w0, g0,
                                tokens0, nullptr, sb0, first_stage, pss_pred_aux_weight(request),
                                /*run_output=*/false, /*run_embedding=*/true);
            IDA_CUDA_CHECK(cudaStreamSynchronize(first_stage.stream));

            if (embed_clip_threshold > 0.0f) {
                float clip_stats[2] = {0.0f, 0.0f};
                IDA_CUDA_CHECK(cudaMemcpy(clip_stats, sb0.embed_clip_stats, sizeof(clip_stats),
                                           cudaMemcpyDeviceToHost));
                const int rows = static_cast<int>(clip_stats[0]);
                if (rows > 0) {
                    step_embed_clip_fired = true;
                    step_embed_clip_rows += rows;
                    step_embed_clip_max = std::max(step_embed_clip_max, clip_stats[1]);
                }
            }
            if (lrss1 && lrss1->lss_aux_copy_pending) {
                lrss1->lss_aux_copy_pending = 0;
                lrss1->lss_aux_valid = 1;
            }
            if (lrss1 && lrss1->lss_aux_valid) last_lss_aux = lrss1->lss_last_aux;
            last_lss_skip_tail = sb1.lss_feedback_skip_tail;
            last_lss_residual_scale = last_lss_skip_tail > 0
                ? lss_feedback_residual_scale() : 1.0f;
            loss_sum += micro_loss;
            total_tokens += elements;
            examples_seen += static_cast<std::size_t>(actual_mb);
            ++micro_done;
            ++micros_this_step;
        }
        }

        last_loss = loss_sum / std::max(1, micros_this_step);
        loss_ema = loss_ema < 0.0f ? last_loss : 0.95f * loss_ema + 0.05f * last_loss;
        if (micros_this_step > 1) {
            const float inv = 1.0f / static_cast<float>(micros_this_step);
            scale_slots(first_stage, slots0, inv);
            scale_slots(second_stage, slots1, inv);
        }
        std::vector<float> host_norms0, host_norms1;
        const double sq = static_cast<double>(collect_norms(first_stage, sb0, slots0, normsq0, host_norms0)) +
                          static_cast<double>(collect_norms(second_stage, sb1, slots1, normsq1, host_norms1));
        last_grad_norm = static_cast<float>(std::sqrt(std::max(0.0, sq)));
        dominant_slot.clear();
        double dominant_sq = 0.0;
        auto find_dominant = [&](const std::vector<float>& norms, const std::vector<std::string>& names) {
            for (std::size_t i = 0; i < norms.size() && i < names.size(); ++i) {
                if (norms[i] > dominant_sq) {
                    dominant_sq = norms[i];
                    dominant_slot = names[i];
                }
            }
        };
        find_dominant(host_norms0, names0);
        find_dominant(host_norms1, names1);
        dominant_slot_norm = static_cast<float>(std::sqrt(dominant_sq));
        dominant_slot_frac = sq > 0.0 ? static_cast<float>(dominant_sq / sq) : 0.0f;

        {
            double pss_pred_sq = 0.0;
            auto add_pss = [&](const std::vector<float>& norms, const std::vector<std::string>& names) {
                for (std::size_t i = 0; i < norms.size() && i < names.size(); ++i) {
                    if (names[i] == "pss.pred_down" || names[i] == "pss.pred_up")
                        pss_pred_sq += norms[i];
                }
            };
            add_pss(host_norms0, names0);
            add_pss(host_norms1, names1);
            pss_predictor_grad_norm = static_cast<float>(std::sqrt(pss_pred_sq));
            pss_predictor_grad_frac = sq > 0.0 ? static_cast<float>(pss_pred_sq / sq) : 0.0f;
        }

        // collect_norms() has already fenced both stages; keep the PSS fence
        // on that same boundary and read its contiguous evidence surface once.
        if (sb1.pss_fence_metrics) {
            set_device(second_stage);
            PssFenceEvidence evidence{};
            IDA_CUDA_CHECK(cudaMemcpy(
                &evidence, sb1.pss_fence_metrics, sizeof(evidence), cudaMemcpyDeviceToHost));
            const PssFenceMetrics& metrics = evidence.metrics;
            const float int2_inv_rms = evidence.int2_inv_rms;
            const float confidence_minmax[2] = {
                evidence.confidence_min, evidence.confidence_max};
            const float err_sq = metrics.err_sq;
            const float target_sq = metrics.target_sq;
            const unsigned int covered = metrics.covered;
            const unsigned int matched = metrics.int2_matched;
            const unsigned int scored = metrics.int2_scored;
            const std::size_t n = static_cast<std::size_t>(sb1.B) * sb1.S * sb1.H;
            last_pss_covered = covered;
            last_pss_n = n;
            last_pss_int2_matched = matched;
            last_pss_int2_scored = scored;
            last_pss_scored_micros = sb1.pss_scored_micros;
            last_pss_int2_inv_rms = int2_inv_rms;
            last_pss_confidence_min =
                confidence_minmax[0] <= confidence_minmax[1] ? confidence_minmax[0] : 0.0f;
            last_pss_confidence_max =
                confidence_minmax[0] <= confidence_minmax[1] ? confidence_minmax[1] : 0.0f;
            last_pss_confidence = n ? static_cast<float>(static_cast<double>(covered) / n) : 0.0f;
            last_pss_err = target_sq > 1e-12f ? err_sq / target_sq : 0.0f;
            last_pss_int2 = scored ? static_cast<float>(static_cast<double>(matched) / scored) : 0.0f;
            last_pss_blend_delta_rms =
                (sb1.pss_last_blend_frac > 0.0f && n > 0 && std::isfinite(err_sq))
                    ? sb1.pss_last_blend_frac *
                          std::sqrt(std::max(0.0f, err_sq / static_cast<float>(n)))
                    : 0.0f;
            last_pss_aux_weight = sb1.pss_aux_weight_used;
            last_pss_aux_normalize = sb1.pss_aux_normalize_used;
            last_pss_aux_denom =
                (last_pss_aux_normalize && std::isfinite(target_sq) && target_sq > 1e-12f)
                    ? target_sq : static_cast<float>(n);
            last_pss_conditioning_active = sb1.pss_conditioning_active;
            last_pss_effective_confidence = last_pss_confidence;
            last_pss_override_active = false;
            last_pss_governor_event = "none";
            if (pss_governor_enabled() && opt_step >= pss_engage_min_optimizer_steps()) {
                float confidence = last_pss_confidence;
                float override_confidence = 0.0f;
                last_pss_override_active = pss_confidence_override(&override_confidence);
                if (last_pss_override_active) confidence = override_confidence;
                last_pss_effective_confidence = confidence;
                const float previous_frac = sb1.pss_engaged_frac;
                if (confidence >= pss_engage_in()) {
                    sb1.pss_engaged_frac = std::min(
                        sb1.pss_engaged_frac + pss_engage_step(), pss_engage_max());
                    if (sb1.pss_engaged_frac > previous_frac)
                        last_pss_governor_event = "engage";
                } else if (confidence < pss_engage_out()) {
                    sb1.pss_engaged_frac = 0.0f;
                    if (previous_frac > 0.0f)
                        last_pss_governor_event = "disengage";
                }
                if (last_pss_governor_event != "none") {
                    std::fprintf(stderr,
                        "[pss-governor] opt_step=%d event=%s confidence=%.6g "
                        "override=%s engaged_frac=%.6g stage=1\n",
                        opt_step, last_pss_governor_event.c_str(),
                        last_pss_effective_confidence,
                        last_pss_override_active ? "true" : "false",
                        sb1.pss_engaged_frac);
                }
            }
        }

        // Preserve the production support-transition policy across the stage
        // boundary. LSS/PSS are owned by stage 1, while dominance is measured
        // from both stages, so this controller must run after the merged norm
        // pass rather than treating either shard as the whole body.
        if (support_transition_enabled() && w1.lss_rank > 0) {
            const float prev_target = prev_lss_target_norm;
            const float target_growth =
                (prev_target > 1.0e-12f && last_lss_target_norm > 0.0f)
                    ? (last_lss_target_norm / prev_target) : 1.0f;
            const bool rmse_improving =
                prev_lss_relative_rmse > 0.0f &&
                last_lss_relative_rmse < prev_lss_relative_rmse;
            if (prev_lss_relative_rmse > 0.0f && last_lss_relative_rmse > 0.0f) {
                lss_rmse_non_improve_windows = rmse_improving
                    ? 0 : lss_rmse_non_improve_windows + 1;
            }
            const bool support_dominant =
                (dominant_slot.rfind("pss.", 0) == 0 ||
                 dominant_slot.rfind("lss.", 0) == 0 ||
                 dominant_slot.rfind("lrss.", 0) == 0) &&
                dominant_slot_frac > support_transition_dominant_frac_max();
            const bool growth_pressure =
                prev_target > 0.0f &&
                target_growth > support_transition_target_growth_max() &&
                lss_rmse_non_improve_windows >= 2;
            const bool recover_ready =
                prev_target > 0.0f &&
                target_growth < support_transition_target_growth_max() &&
                (last_lss_relative_rmse <= support_transition_recover_rmse() ||
                 rmse_improving) &&
                !support_dominant;
            if (growth_pressure || support_dominant) {
                support_transition_scale = std::max(
                    support_transition_floor(),
                    support_transition_scale * support_transition_step_down());
            } else if (recover_ready) {
                support_transition_scale = std::min(
                    1.0f,
                    support_transition_scale * support_transition_step_up());
            }
            if (last_lss_target_norm > 0.0f) {
                prev_lss_target_norm = last_lss_target_norm;
            }
            if (last_lss_relative_rmse > 0.0f) {
                prev_lss_relative_rmse = last_lss_relative_rmse;
            }
        }

        last_lr = static_cast<float>(lr_at(opt_step, total_opt, base_lr));
        if (lr_accum_coupling && accum_ceiling > 0) {
            last_lr *= static_cast<float>(micros_this_step) / std::max(1, accum_ceiling);
        }
        last_embed_clip_fired = step_embed_clip_fired;
        last_embed_clip_max = step_embed_clip_max;
        last_embed_clip_rows = step_embed_clip_rows;
        if (step_embed_clip_fired) {
            ++embed_clip_steps;
            embed_clip_rows_total += step_embed_clip_rows;
        }
        last_clip_fired = false;
        last_clip_scale = 1.0f;
        bool update_skipped = false;
        if (!std::isfinite(last_grad_norm) || last_grad_norm > grad_norm_abs_ceiling) {
            ++skipped;
            update_skipped = true;
        } else {
            if (last_grad_norm > clip) {
                last_clip_fired = true;
                last_clip_scale = clip / last_grad_norm;
                ++clip_steps;
                scale_slots(first_stage, slots0, last_clip_scale);
                scale_slots(second_stage, slots1, last_clip_scale);
            }
            update_stage(first_stage, w0, f80, ampere0, sb0, slots0, names0, 0, opt_step, skipped, last_lr);
            update_stage(second_stage, w1, f81, ampere1, sb1, slots1, names1, static_cast<int>(slots0.size()),
                         opt_step, skipped, last_lr);
            // f80b's weight-quantization cache is redundant with f80's (its
            // optimizer slots are never bound to it -- see the allocation
            // comment above) but must be refreshed in lockstep so slot 1's
            // forward reads the SAME post-update quantized weights slot 0's
            // does, not a stale pre-optimizer-step copy.
            if (peer_1f1b) {
                set_device(first_stage);
                g_ampere_packed_weights = &ampere0;
                refresh_fp8_weights(f80b, first_stage.stream, /*recompute_amax=*/false);
            }
        }
        zero_slots(first_stage, g0, w0);
        zero_slots(second_stage, g1, w1);

        {
            ::ida_native::ontology::StepOutcome outcome;
            outcome.opt_step = opt_step;
            outcome.loss = last_loss;
            outcome.wallclock_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            outcome.tokens = total_tokens;
            outcome.update_skipped = update_skipped;
            outcome.clip_fired = last_clip_fired;
            outcome.grad_norm = last_grad_norm;
            ::ida_native::ontology::observe_step(outcome);
        }
        const double elapsed = std::max(0.000001, std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count());
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
        if (shared_metrics.is_open() || burn_metrics.is_open()) {
            const auto metrics_now = std::chrono::steady_clock::now();
            const double monotonic_s = std::chrono::duration<double>(metrics_now - started).count();
            const double step_duration_ms = std::chrono::duration<double, std::milli>(
                metrics_now - step_wall_start).count();
            const double epoch = ds.num_sequences > 0
                ? static_cast<double>(examples_seen) / ds.num_sequences : 0.0;
            std::ostringstream line;
            line << "{\"timestamp\":\"" << now_utc_iso8601()
                 << "\",\"monotonic_s\":" << json_number(monotonic_s)
                 << ",\"seat\":\"" << json_escape(request.seat)
                 << "\",\"family\":\"" << json_escape(request.family)
                 << "\",\"version\":\"" << json_escape(request.version)
                 << "\",\"attention_backend\":\""
                 << attention_backend_name(attention_backend)
                 << "\",\"precision_profile\":\"" << json_escape(request.precision_profile)
                 << "\",\"global_step\":" << micro_done
                 << ",\"optimizer_step\":" << opt_step
                 << ",\"micro_step\":" << micro_done
                 << ",\"examples_seen\":" << examples_seen
                 << ",\"tokens_seen\":" << total_tokens
                 << ",\"epoch\":" << json_number(epoch)
                 << ",\"loss\":" << json_number(last_loss)
                 << ",\"loss_ema\":" << json_number(loss_ema)
                 << ",\"learning_rate\":" << json_number(last_lr)
                 << ",\"grad_norm\":" << json_number(last_grad_norm)
                 << ",\"global_grad_clip_fired\":"
                 << (last_clip_fired ? "true" : "false")
                 << ",\"global_grad_clip_scale\":" << json_number(last_clip_scale)
                 << ",\"embed_row_clip_enabled\":"
                 << (embed_clip_threshold > 0.0f ? "true" : "false")
                 << ",\"embed_row_clip_fired\":"
                 << (last_embed_clip_fired ? "true" : "false")
                 << ",\"embed_row_clip_threshold\":" << json_number(embed_clip_threshold)
                 << ",\"embed_row_clip_max_preclip_norm\":"
                 << json_number(last_embed_clip_max)
                 << ",\"embed_row_clipped_rows\":" << last_embed_clip_rows
                 << ",\"dominant_grad_slot\":\"" << json_escape(dominant_slot)
                 << "\",\"dominant_grad_slot_norm\":" << json_number(dominant_slot_norm)
                 << ",\"dominant_grad_slot_frac\":" << json_number(dominant_slot_frac)
                 << ",\"pss_predictor_grad_norm\":" << json_number(pss_predictor_grad_norm)
                 << ",\"pss_predictor_grad_frac\":" << json_number(pss_predictor_grad_frac)
                 << ",\"lss_feedback_skip_tail\":" << last_lss_skip_tail
                 << ",\"lss_feedback_residual_scale\":"
                 << json_number(last_lss_residual_scale)
                 << ",\"lss_aux\":" << json_number(last_lss_aux)
                 // The pipeline currently has no LSS recon-norm/target-norm
                 // counter. Emit null rather than presenting zero as a read.
                 << ",\"lss_recon_norm\":null,\"lss_target_norm\":null"
                 << ",\"lss_relative_rmse\":null"
                 << ",\"support_transition_scale\":"
                 << json_number(support_transition_scale)
                 << ",\"pss_slot_clip_steps\":0,\"pss_spike_ratio_max\":null"
                 << ",\"pss_spike_slot\":\"\""
                 << ",\"pss_confidence\":" << json_number(last_pss_confidence)
                 << ",\"pss_pred_err\":" << json_number(last_pss_err)
                 << ",\"pss_int2_agreement\":" << json_number(last_pss_int2)
                 << ",\"pss_engaged_frac\":" << json_number(sb1.pss_engaged_frac)
                 << ",\"pss_covered\":" << last_pss_covered
                 << ",\"pss_n\":" << last_pss_n
                 << ",\"pss_int2_matched\":" << last_pss_int2_matched
                 << ",\"pss_int2_scored\":" << last_pss_int2_scored
                 << ",\"pss_scored_micros\":" << last_pss_scored_micros
                 << ",\"pss_int2_inv_rms\":" << json_number(last_pss_int2_inv_rms)
                 << ",\"pss_confidence_min\":" << json_number(last_pss_confidence_min)
                 << ",\"pss_confidence_max\":" << json_number(last_pss_confidence_max)
                 << ",\"pss_blend_delta_rms\":" << json_number(last_pss_blend_delta_rms)
                 << ",\"pss_aux_weight\":" << json_number(last_pss_aux_weight)
                 << ",\"pss_aux_denom\":" << json_number(last_pss_aux_denom)
                 << ",\"pss_aux_normalize\":"
                 << (last_pss_aux_normalize ? "true" : "false")
                 << ",\"pss_conditioning_active\":"
                 << (last_pss_conditioning_active ? "true" : "false")
                 << ",\"pss_override_active\":"
                 << (last_pss_override_active ? "true" : "false")
                 << ",\"pss_effective_confidence\":"
                 << json_number(last_pss_effective_confidence)
                 << ",\"pss_governor_event\":\""
                 << json_escape(last_pss_governor_event) << "\""
                 << ",\"pss_conditioning_mode\":\""
                 << json_escape(pss_conditioning_mode) << "\""
                 << ",\"pss_aux_normalize_mode\":\""
                 << json_escape(pss_aux_normalize_mode) << "\""
                 << ",\"step_duration_ms\":" << json_number(step_duration_ms)
                 << ",\"active_grad_accum\":" << micros_this_step
                 << ",\"requested_grad_accum\":" << accum_ceiling
                 << ",\"effective_batch\":" << (mb * micros_this_step)
                 << ",\"model_parallel\":true,\"pipeline_stage_count\":2"
                 << ",\"pipeline_split_layer\":" << split_layer
                 << ",\"pipeline_first_device\":" << first_stage.device_id
                 << ",\"pipeline_second_device\":" << second_stage.device_id
                 << ",\"peer_transport\":\"" << json_escape(peer_transport_used) << "\""
                 << ",\"peer_forward_bytes\":" << peer_forward_bytes
                 << ",\"peer_backward_bytes\":" << peer_backward_bytes
                 << ",\"event\":\"native_step\"}";
            if (shared_metrics.is_open()) {
                shared_metrics << line.str() << "\n";
                shared_metrics.flush();
            }
            if (burn_metrics.is_open()) {
                burn_metrics << line.str() << "\n";
                burn_metrics.flush();
            }
        }
#endif
        if (on_step) {
            ProgressReport report{};
            report.micro_step = micro_done;
            report.optimizer_step = opt_step;
            report.active_grad_accum = micros_this_step;
            report.requested_grad_accum = accum_ceiling;
            report.tokens = total_tokens;
            report.elapsed_s = elapsed;
            report.loss = last_loss;
            report.loss_ema = loss_ema;
            report.grad_norm = last_grad_norm;
            report.lr = last_lr;
            report.effective_batch = mb * micros_this_step;
            report.skipped_steps = skipped;
            report.global_grad_clip_steps = clip_steps;
            report.embed_row_clip_steps = embed_clip_steps;
            report.embed_row_clipped_rows_total = embed_clip_rows_total;
            report.global_grad_clip_fired = last_clip_fired;
            report.global_grad_clip_scale = last_clip_scale;
            report.embed_row_clip_fired = last_embed_clip_fired;
            report.embed_row_clip_threshold = embed_clip_threshold;
            report.embed_row_clip_max_preclip_norm = last_embed_clip_max;
            report.embed_row_clipped_rows = last_embed_clip_rows;
            report.dominant_grad_slot = dominant_slot;
            report.dominant_grad_slot_norm = dominant_slot_norm;
            report.dominant_grad_slot_frac = dominant_slot_frac;
            report.lss_feedback_skip_tail = last_lss_skip_tail;
            report.lss_feedback_residual_scale = last_lss_residual_scale;
            report.lss_aux = last_lss_aux;
            report.support_transition_scale = support_transition_scale;
            report.pss_confidence = last_pss_confidence;
            report.pss_pred_err = last_pss_err;
            report.pss_int2_agreement = last_pss_int2;
            report.pss_engaged_frac = sb1.pss_engaged_frac;
            report.pss_covered = last_pss_covered;
            report.pss_n = last_pss_n;
            report.pss_int2_matched = last_pss_int2_matched;
            report.pss_int2_scored = last_pss_int2_scored;
            report.pss_scored_micros = last_pss_scored_micros;
            report.pss_int2_inv_rms = last_pss_int2_inv_rms;
            report.pss_confidence_min = last_pss_confidence_min;
            report.pss_confidence_max = last_pss_confidence_max;
            report.pss_blend_delta_rms = last_pss_blend_delta_rms;
            report.pss_aux_weight = last_pss_aux_weight;
            report.pss_aux_denom = last_pss_aux_denom;
            report.pss_aux_normalize = last_pss_aux_normalize;
            report.pss_conditioning_active = last_pss_conditioning_active;
            report.pss_override_active = last_pss_override_active;
            report.pss_effective_confidence = last_pss_effective_confidence;
            report.pss_governor_event = last_pss_governor_event;
            report.pss_conditioning_mode = pss_conditioning_mode;
            report.pss_aux_normalize_mode = pss_aux_normalize_mode;
            report.fp8_active = f80.on || f81.on;
            report.attention_backend = attention_backend_name(attention_backend);
            report.precision_profile = request.precision_profile;
            try { on_step(report); } catch (...) {}
        }
    }

    const double elapsed = std::max(0.000001, std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count());
    const bool fp8_active = f80.on || f81.on;
    const float final_pss_engaged_frac = sb1.pss_engaged_frac;
    double attn_p_zero_frac_stage0 = 0.0;
    double attn_p_zero_frac_stage1 = 0.0;
    float attn_lse_drift_max_stage0 = 0.0f;
    float attn_lse_drift_max_stage1 = 0.0f;
    set_device(first_stage);
    attn_bwd_health_read(
        &attn_p_zero_frac_stage0, &attn_lse_drift_max_stage0, first_stage.stream);
    set_device(second_stage);
    attn_bwd_health_read(
        &attn_p_zero_frac_stage1, &attn_lse_drift_max_stage1, second_stage.stream);

    {
        write_native_timeline_event(request, "native_checkpoint_save", "start");
        const auto save_started = std::chrono::steady_clock::now();
        std::string save_error;
        if (!save_model_parallel_lattice_weights_safetensors(
                request, w0, first_stage.device_id, first_stage.stream,
                w1, second_stage.device_id, second_stage.stream,
                save_error, opt_step, split_layer)) {
            throw std::runtime_error(
                "model-parallel merged weight save failed: " + save_error);
        }
        write_native_timeline_event(
            request, "native_checkpoint_save", "end",
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - save_started).count());
    }

    set_device(first_stage);
    IDA_CUDA_CHECK(cudaFreeAsync(normsq0, first_stage.stream));
    IDA_CUDA_CHECK(cudaFreeAsync(tokens0, first_stage.stream));
    IDA_CUDA_CHECK(cudaFreeAsync(segs0, first_stage.stream));
    CUBLAS_CHECK(cublasDestroy(cublas0));
    free_fp8_ctx(f80, first_stage);
    free_ampere_packed_weights(ampere0, first_stage);
    free_packed_fp4_attention_ctx(fp40, first_stage);
    free_step_buffers(sb0, first_stage);
    if (peer_1f1b) {
        IDA_CUDA_CHECK(cudaFreeAsync(tokens0b, first_stage.stream));
        IDA_CUDA_CHECK(cudaFreeAsync(segs0b, first_stage.stream));
        free_fp8_ctx(f80b, first_stage);
        free_packed_fp4_attention_ctx(fp40b, first_stage);
        free_step_buffers(sb0b, first_stage);
        IDA_CUDA_CHECK(cudaEventDestroy(ev_fwd0_ready[0]));
        IDA_CUDA_CHECK(cudaEventDestroy(ev_fwd0_ready[1]));
        IDA_CUDA_CHECK(cudaEventDestroy(ev_bwd1_ready[0]));
        IDA_CUDA_CHECK(cudaEventDestroy(ev_bwd1_ready[1]));
    }
    free_lattice_grads(g0, w0, first_stage);
    free_lattice_opt(opt0, w0, first_stage);
    free_lattice_weights(w0, first_stage);

    set_device(second_stage);
    if (lrss1) {
        sb1.lrss_p = nullptr; sb1.lrss_s = nullptr; sb1.lrss_g = nullptr;
        free_lrss_scratch(lrss1, second_stage);
    }
    IDA_CUDA_CHECK(cudaFreeAsync(normsq1, second_stage.stream));
    IDA_CUDA_CHECK(cudaFreeAsync(tokens1, second_stage.stream));
    IDA_CUDA_CHECK(cudaFreeAsync(labels1, second_stage.stream));
    IDA_CUDA_CHECK(cudaFreeAsync(segs1, second_stage.stream));
    CUBLAS_CHECK(cublasDestroy(cublas1));
    free_fp8_ctx(f81, second_stage);
    free_ampere_packed_weights(ampere1, second_stage);
    free_packed_fp4_attention_ctx(fp41, second_stage);
    free_step_buffers(sb1, second_stage);
    free_lattice_grads(g1, w1, second_stage);
    free_lattice_opt(opt1, w1, second_stage);
    free_lattice_weights(w1, second_stage);

    BurnResult result{};
    result.global_step = micro_done;
    result.optimizer_steps = opt_step;
    result.tokens_processed = total_tokens;
    result.elapsed_seconds = elapsed;
    result.tokens_per_second = static_cast<double>(total_tokens) / elapsed;
    result.device_bytes_touched = weight_bytes;
    // Mirrors run_lattice_training's dump call (~line 9641) -- was missing
    // here entirely, so the model-parallel/pipeline-split path (which is
    // what IDA_NATIVE_PEER_1F1B runs through) never emitted gemm_role/
    // gemm_total/pack_launch telemetry at all, only the objective trace.
    // gemm_trace's RoleStat table is a single process-wide static array (see
    // gemm_trace.hpp), not per-device, so this one dump reports the GEMM
    // work COMBINED across both pipeline stages -- correct for judging the
    // pipeline as a whole, but the "cuda_device" field in the emitted
    // gemm_trace_header just reflects whichever device is active at this
    // call site (second_stage here), not "the" device the work ran on.
    ::ida_native::gemm_trace::dump(
        ::ida_native::ontology::sink(), elapsed,
        ::ida_native::ontology::context_epoch());
    result.final_loss = last_loss;
    result.final_loss_ema = loss_ema;
    result.final_grad_norm = last_grad_norm;
    result.final_lr = last_lr;
    result.skipped_steps = skipped;
    result.global_grad_clip_steps = clip_steps;
    result.embed_row_clip_steps = embed_clip_steps;
    result.embed_row_clipped_rows_total = embed_clip_rows_total;
    result.final_global_grad_clip_fired = last_clip_fired;
    result.final_global_grad_clip_scale = last_clip_scale;
    result.final_embed_row_clip_fired = last_embed_clip_fired;
    result.final_embed_row_clip_threshold = embed_clip_threshold;
    result.final_embed_row_clip_max_preclip_norm = last_embed_clip_max;
    result.final_embed_row_clipped_rows = last_embed_clip_rows;
    result.final_dominant_grad_slot = dominant_slot;
    result.final_dominant_grad_slot_norm = dominant_slot_norm;
    result.final_dominant_grad_slot_frac = dominant_slot_frac;
    result.final_pss_predictor_grad_norm = pss_predictor_grad_norm;
    result.final_pss_predictor_grad_frac = pss_predictor_grad_frac;
    result.final_lss_feedback_skip_tail = last_lss_skip_tail;
    result.final_lss_feedback_residual_scale = last_lss_residual_scale;
    result.final_lss_aux = last_lss_aux;
    result.final_lss_recon_norm = last_lss_recon_norm;
    result.final_lss_target_norm = last_lss_target_norm;
    result.final_lss_relative_rmse = last_lss_relative_rmse;
    result.final_support_transition_scale = support_transition_scale;
    result.final_pss_confidence = last_pss_confidence;
    result.final_pss_pred_err = last_pss_err;
    result.final_pss_int2_agreement = last_pss_int2;
    result.final_pss_engaged_frac = final_pss_engaged_frac;
    result.final_pss_covered = last_pss_covered;
    result.final_pss_n = last_pss_n;
    result.final_pss_int2_matched = last_pss_int2_matched;
    result.final_pss_int2_scored = last_pss_int2_scored;
    result.final_pss_scored_micros = last_pss_scored_micros;
    result.final_pss_int2_inv_rms = last_pss_int2_inv_rms;
    result.final_pss_confidence_min = last_pss_confidence_min;
    result.final_pss_confidence_max = last_pss_confidence_max;
    result.final_pss_blend_delta_rms = last_pss_blend_delta_rms;
    result.final_pss_aux_weight = last_pss_aux_weight;
    result.final_pss_aux_denom = last_pss_aux_denom;
    result.final_pss_aux_normalize = last_pss_aux_normalize;
    result.final_pss_conditioning_active = last_pss_conditioning_active;
    result.final_pss_override_active = last_pss_override_active;
    result.final_pss_effective_confidence = last_pss_effective_confidence;
    result.final_pss_governor_event = last_pss_governor_event;
    result.final_pss_conditioning_mode = pss_conditioning_mode;
    result.final_pss_aux_normalize_mode = pss_aux_normalize_mode;
    result.fp8_active = fp8_active;
    result.attention_backend = attention_backend_name(attention_backend);
    result.precision_profile = request.precision_profile;
    result.attn_p_zero_frac_stage0 = attn_p_zero_frac_stage0;
    result.attn_p_zero_frac_stage1 = attn_p_zero_frac_stage1;
    result.attn_lse_drift_max_stage0 = attn_lse_drift_max_stage0;
    result.attn_lse_drift_max_stage1 = attn_lse_drift_max_stage1;
    result.attn_p_zero_frac = std::max(attn_p_zero_frac_stage0, attn_p_zero_frac_stage1);
    result.attn_lse_drift_max = std::max(
        attn_lse_drift_max_stage0, attn_lse_drift_max_stage1);
    result.model_parallel = true;
    result.pipeline_stage_count = 2;
    result.pipeline_split_layer = split_layer;
    result.pipeline_first_device = first_stage.device_id;
    result.pipeline_second_device = second_stage.device_id;
    result.peer_transport = peer_transport_used;
    result.peer_forward_bytes = peer_forward_bytes;
    result.peer_backward_bytes = peer_backward_bytes;
    return result;
}

}  // namespace ida_native
