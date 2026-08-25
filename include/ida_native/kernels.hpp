#pragma once

#include <string_view>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace ida_native {

enum class AttentionBackendKind : int {
    ScalarFlash = 0,
    HopperWgmmaPackedFp4 = 1,
    HopperWgmmaFp8 = 2,
    BlackwellMxf4Fp4 = 3,
};

struct PackedFp4AttentionOperands {
    const std::uint8_t* qk_packed{nullptr};
    const std::uint8_t* q_saved_e4m3{nullptr};
    const std::uint8_t* k_saved_e4m3{nullptr};
    const float* q_scale{nullptr};
    const float* q_descale{nullptr};
    const float* k_scale{nullptr};
    const float* k_descale{nullptr};
    // Reserved for a deployment-owned private backend.
    const float* q_mean{nullptr};
    const float* k_mean{nullptr};
    float* q_unpack_f32{nullptr};
    float* k_unpack_f32{nullptr};
    float* q_tile_stage_f32{nullptr};
    float* k_tile_stage_f32{nullptr};
    float* v_tile_stage_f32{nullptr};
    std::size_t tile_stage_elems{0};
    int tma_stage_depth{0};
    std::size_t packed_elems{0};
};

AttentionBackendKind parse_attention_backend(std::string_view backend);
const char* attention_backend_name(AttentionBackendKind backend);

// ── embedding ────────────────────────────────────────────────────────────────
void embedding_forward(
    const uint32_t*      d_tokens,
    const __nv_bfloat16* d_weight,
    __nv_bfloat16*       d_out,
    int B, int S, int H,
    cudaStream_t stream
);
// d_grad_weight is FP32 to allow atomicAdd accumulation across token collisions.
void embedding_backward(
    const __nv_bfloat16* d_grad_out,
    const uint32_t*      d_tokens,
    float*               d_grad_weight,
    int B, int S, int H,
    cudaStream_t stream
);
void embedding_backward(
    const __nv_bfloat16* d_grad_out,
    const uint32_t*      d_tokens,
    __nv_bfloat16*       d_grad_weight,
    int B, int S, int H,
    cudaStream_t stream
);
void position_embedding_forward(
    __nv_bfloat16* d_hidden,
    const __nv_bfloat16* d_position,
    const std::uint16_t* d_segs,
    int B, int S, int H, int max_positions,
    cudaStream_t stream
);
void position_embedding_backward(
    const __nv_bfloat16* d_grad_hidden,
    const std::uint16_t* d_segs,
    float* d_grad_position,
    int B, int S, int H, int max_positions,
    cudaStream_t stream
);
// k_embed_row_clip for why: high-frequency/near-universal tokens on
// JSONL-formatted corpora scatter-accumulate gradient via embedding_backward
// far faster than any content token, dominating the global grad-norm
// identically across every attention/quantization format tested). Call
// once per micro-step, right after embedding_backward, so no single
// micro-step's contribution to one row can exceed max_row_norm.
void embed_row_clip(
    float* d_grad_weight, int V, int H, float max_row_norm, cudaStream_t stream
);
// Same clip as above, but also reports step-local stats into d_stats:
// d_stats[0] = clipped row count, d_stats[1] = max pre-clip row norm.
void embed_row_clip_with_stats(
    float* d_grad_weight,
    int V,
    int H,
    float max_row_norm,
    float* d_stats,
    cudaStream_t stream
);
// LRSS-informed variant: rows flagged in common_mask (size V, 1 = "common"
// token per scripts/build_lrss_token_vocab.py's frequency+doc-frac detector,
// the same detector the PyTorch-era LRSS content pooling used to down-weight
// these tokens) get the tighter common_row_norm ceiling instead of the
// blanket max_row_norm. d_stats[0]/[1] as above, scoped to common rows only.
void embed_row_clip_masked_with_stats(
    float* d_grad_weight,
    int V,
    int H,
    float max_row_norm,
    float common_row_norm,
    const uint8_t* d_common_mask,
    float* d_stats,
    cudaStream_t stream
);

// ── RMSNorm ──────────────────────────────────────────────────────────────────
// eps = 1e-6 recommended.
void rmsnorm_forward(
    const __nv_bfloat16* d_x,
    const __nv_bfloat16* d_scale,
    __nv_bfloat16*       d_out,
    float*               d_rms,      // scratch: [rows] rms values for backward reuse
    int rows, int H, float eps,
    cudaStream_t stream
);
void rmsnorm_backward(
    const __nv_bfloat16* d_grad_out,
    const __nv_bfloat16* d_x,
    const __nv_bfloat16* d_scale,
    const float*         d_rms,
    __nv_bfloat16*       d_grad_x,
    float*               d_grad_scale,  // FP32 accumulator
    int rows, int H, float eps,
    cudaStream_t stream
);

void layernorm_forward(
    const __nv_bfloat16* d_x,
    const __nv_bfloat16* d_scale,
    const __nv_bfloat16* d_bias,
    __nv_bfloat16* d_out,
    float* d_inv_std,
    int rows, int H, float eps,
    cudaStream_t stream
);
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
);

// ── SiLU + gated multiply (SwiGLU gate step) ─────────────────────────────────
// out[i] = silu(gate[i]) * up[i]
void swiglu_forward(
    const __nv_bfloat16* d_gate,
    const __nv_bfloat16* d_up,
    __nv_bfloat16*       d_out,
    int N,
    cudaStream_t stream
);
// Backward: given d_swiglu_out, recover d_gate and d_up.
void swiglu_backward(
    const __nv_bfloat16* d_grad_out,
    const __nv_bfloat16* d_gate,
    const __nv_bfloat16* d_up,
    __nv_bfloat16*       d_grad_gate,
    __nv_bfloat16*       d_grad_up,
    int N,
    cudaStream_t stream
);

void gelu_new_forward(
    const __nv_bfloat16* d_x,
    __nv_bfloat16* d_out,
    std::size_t n,
    cudaStream_t stream
);
void gelu_new_backward(
    const __nv_bfloat16* d_grad_out,
    const __nv_bfloat16* d_x,
    __nv_bfloat16* d_grad_x,
    std::size_t n,
    cudaStream_t stream
);

// ── Softmax (row-wise, in-place) ─────────────────────────────────────────────
// d_x: [rows, cols] — overwrites with softmax probabilities (FP32).
void softmax_inplace_f32(float* d_x, int rows, int cols, cudaStream_t stream);

// ── Cognitive-architecture sparse MoE port (moe.cu, 2026-07-21/22) ──────────
// See moe.cu's header comment for the packing-aware pooling rationale.

// Pools hidden [B,S,H] over each packed sample's own span (segs[pos] =
// that sample's start position; segs may be null = whole row is one sample,
// matching the PyTorch reference's assumption). After this call,
// d_pool_scratch[b, start, :] holds the correct pooled mean ONLY at actual
// sample-start positions (start == segs[b,start]); every other position is
// left at zero (nothing ever scatters there). Reading the right pooled
// vector for an arbitrary position therefore always requires
// moe_gather_by_sample below (which looks up d_pool_scratch[b,
// segs[b,pos], :]) -- never read d_pool_scratch at a non-start position
// directly. d_pool_scratch/d_count_scratch: [B,S,H]/[B,S] scratch, zeroed
// internally by this call.
void moe_pool_by_sample(
    const __nv_bfloat16* d_hidden, const std::uint16_t* d_segs,
    float* d_pool_scratch, float* d_count_scratch,
    int B, int S, int H, cudaStream_t stream
);

// Broadcasts a [B,S,N] tensor that's only meaningful at each sample's START
// position (segs[pos]==pos) out to every position within that sample.
void moe_gather_by_sample(
    const float* d_start_values, const std::uint16_t* d_segs, float* d_out,
    int B, int S, int N, cudaStream_t stream
);

// Small per-row projection: out[row,n] = sum_h in[row,h] * w[n,h] (torch
// [out,in] weight convention, no bias). Direct kernel rather than the
// tensor-core-oriented bf16 GEMM helpers -- out_dim (num_routes/num_experts)
// is tiny (<=11 in every real config), not worth the dtype-conversion/shape
// overhead of the existing GEMM infra for something this small.
void moe_small_proj_f32(
    const float* d_in, const __nv_bfloat16* d_w, float* d_out,
    int rows, int in_dim, int out_dim, cudaStream_t stream
);

void tanh_inplace_f32(float* d_x, std::size_t n, cudaStream_t stream);
void sigmoid_inplace_f32(float* d_x, std::size_t n, cudaStream_t stream);

// ── QKV bias (2026-08-23; Qwen2-family portability -- attention q/k/v
// projections with an additive per-output-feature bias, unlike native's own
// bias-free convention). Both operate on a strided [rows, row_stride] BF16
// tensor's [base, base+cols) column slice -- matches sb.qkv's packed
// [BS, 3H]-with-per-section-base layout (Q at base=0, K at base=H,
// V at base=2H, each only cols=H or KV wide).
//
// out[row, base+c] += bias[c]  (forward: broadcast-add, in place)
void bias_add_strided_bf16(
    __nv_bfloat16* d_inout, const __nv_bfloat16* d_bias,
    int rows, int cols, int row_stride, int base,
    cudaStream_t stream
);
// out_accum[c] += sum_row d_in[row, base+c]  (backward: bias gradient is a
// plain column-sum of d(pre-bias output) -- bias-add doesn't change the
// gradient flowing to Q/K/V pre-activation, so this reduces the SAME
// d_qkv buffer the weight-gradient GEMMs already consume, no recompute).
// out_accum is an FP32 accumulator (zero it first; caller then casts/adds
// into the persistent BF16 grad slot, same pattern as rmsnorm_backward's
// d_grad_scale -> k_acc_bf16 two-step).
void col_sum_strided_bf16(
    const __nv_bfloat16* d_in, float* d_out_accum,
    int rows, int cols, int row_stride, int base,
    cudaStream_t stream
);

// ── Blackwell preparation contract ───────────────────────────────────────────
// These declarations are compatibility contracts only. Public builds provide
// fail-closed definitions; private deployment packages may supply the backend.
void nvfp4_pack_rows_bf16(
    const __nv_bfloat16* d_input,
    std::uint8_t* d_payload,
    std::uint8_t* d_scales,
    int rows,
    int cols,
    std::uint32_t* d_error,
    cudaStream_t stream
);
void nvfp4_pack_transpose_bf16(
    const __nv_bfloat16* d_input,
    std::uint8_t* d_payload,
    std::uint8_t* d_scales,
    int rows,
    int cols,
    std::uint32_t* d_error,
    cudaStream_t stream
);
void nvfp4_unpack_rows_bf16(
    const std::uint8_t* d_payload,
    const std::uint8_t* d_scales,
    __nv_bfloat16* d_output,
    int rows,
    int cols,
    cudaStream_t stream
);

// PressureField's gate: out = hidden * modulation (elementwise; modulation
// already broadcast per-sample by the caller via moe_gather_by_sample).
void moe_modulate_hidden_f32(
    const __nv_bfloat16* d_hidden, const float* d_modulation, __nv_bfloat16* d_out,
    std::size_t n, cudaStream_t stream
);

// d_a += d_b elementwise, in place. Used for ConstitutionalRouter's
// logits = router_score_out + routed_pressure (routed_pressure is either
// `pressure` itself, when pressure_to_routes is Identity, or its
// projection -- caller resolves which).
void add_inplace_f32(float* d_a, const float* d_b, std::size_t n, cudaStream_t stream);

// ConstitutionalRouter: softmax(d_scores) in place, then hard top-k select +
// scatter into zeroed + renormalize kept values to sum 1 -- mirrors
// constitutional_router.py's scatter_+renormalize exactly.
// d_row_sum_save (nullable): T = sum of kept (pre-renormalize) probs per
// row, needed by moe_norm_by_sum_backward_f32 for this step's backward.
// d_scores_save (nullable): pure softmax output, BEFORE top-k/scatter --
// needed by attn_softmax_backward_f32 for this step's own backward (the
// scores get overwritten in place by the top-k/scatter/renormalize that
// follows, so this is the only way to recover them for backward).
void moe_topk_route_f32(
    float* d_scores, int rows, int num_experts, int top_k, cudaStream_t stream,
    float* d_row_sum_save = nullptr, float* d_scores_save = nullptr,
    bool normalize_topk = true
);

// LateralInhibition: clamp any nonzero weight to [minority_floor,
// dominance_cap], renormalize. No learnable parameters (matches the
// reference). Reference defaults: dominance_cap=0.72, minority_floor=0.08.
// d_c1_save (nullable): clamp(p,max=dominance_cap) BEFORE the min-floor-raise
// -- needed by moe_inhib_clamp_backward_f32. d_row_sum_save (nullable): S =
// sum(bounded) pre-final-renormalize, needed by moe_norm_by_sum_backward_f32.
void moe_lateral_inhibition_f32(
    float* d_weights, int rows, int num_experts,
    float dominance_cap, float minority_floor, cudaStream_t stream,
    float* d_c1_save = nullptr, float* d_row_sum_save = nullptr
);

// CognitiveCircuitMLP's activation: exact-erf GELU (nn.GELU()'s default --
// NOT the tanh approximation, a real numerical divergence point if
// approximated; swiglu's SiLU gate is not a substitute).
void gelu_forward(const __nv_bfloat16* d_x, __nv_bfloat16* d_out, std::size_t n, cudaStream_t stream);

// Expert-bank mixing: mixed[row,:] += route_weights[row, expert_idx] * term[row,:].
// route_weights is [rows, num_experts] (stride num_experts between rows) --
// one column read per call, the per-row scalar gate for one expert's (or the
// trunk's, called with a weight of 1 via a separate path) contribution.
void moe_scale_accumulate_bf16(
    __nv_bfloat16* d_mixed, const __nv_bfloat16* d_term, const float* d_route_weights,
    int rows, int expert_idx, int num_experts, int H, cudaStream_t stream
);

// Copy one expert's contiguous column block out of a flattened
// [rows, num_experts * expert_width] expert-fc-in result into
// [rows, expert_width] scratch. Used by the guarded fused expert-fc-in path:
// one larger GEMM computes all experts' first projection, then the existing
// per-expert fcout/mix path consumes contiguous slices without changing
// quantization or accumulation semantics.
void moe_copy_expert_fcin_slice_bf16(
    const __nv_bfloat16* d_all_expert_fcin, __nv_bfloat16* d_expert_fcin,
    int rows, int num_experts, int expert_width, int expert_idx,
    cudaStream_t stream
);

// Experimental selected-row expert dispatch. Builds compact row lists for
// route_weights[row, expert] > 0, gathers selected rows, and scatters compact
// expert outputs back into the full mixed tensor. Counts are device-side int32
// and may be copied to host by ablation launchers to issue variable-M GEMMs.
void moe_build_expert_row_lists(
    const float* d_route_weights, int* d_counts, int* d_row_indices,
    int rows, int num_experts, cudaStream_t stream
);
void moe_gather_rows_bf16(
    const __nv_bfloat16* d_src, __nv_bfloat16* d_dst,
    const int* d_row_indices, int n_rows, int width, cudaStream_t stream
);
void moe_gather_rows_u8(
    const std::uint8_t* d_src, std::uint8_t* d_dst,
    const int* d_row_indices, int n_rows, int width, cudaStream_t stream
);
void moe_scatter_accumulate_rows_bf16(
    __nv_bfloat16* d_mixed, const __nv_bfloat16* d_compact,
    const float* d_route_weights, const int* d_row_indices,
    int n_rows, int expert_idx, int num_experts, int H, cudaStream_t stream
);
void moe_row_dot_selected_bf16(
    const __nv_bfloat16* d_a_full, const __nv_bfloat16* d_b_compact,
    float* d_out, const int* d_row_indices,
    int n_rows, int H, int out_stride, int out_col, cudaStream_t stream
);
void moe_gather_scale_rows_bf16(
    const __nv_bfloat16* d_src, __nv_bfloat16* d_dst,
    const float* d_route_weights, const int* d_row_indices,
    int n_rows, int expert_idx, int num_experts, int H, cudaStream_t stream
);
void moe_scatter_add_rows_bf16(
    __nv_bfloat16* d_dst_full, const __nv_bfloat16* d_src_compact,
    const int* d_row_indices, int n_rows, int H, cudaStream_t stream
);

// Grouped/device-side selected-expert scheduler. Computes device-side prefix
// offsets from device-side counts and launches a single kernel over all
// selected rows, removing the host count synchronization used by the ablation
// variable-M GEMM path. BF16 correctness path; FP8/WGMMA variant is future work.
void moe_expert_offsets_from_counts(
    const int* d_counts, int* d_offsets, int num_experts, cudaStream_t stream
);
void moe_grouped_expert_forward_bf16(
    const __nv_bfloat16* d_hidden, const float* d_route_weights,
    const int* d_counts, const int* d_offsets, const int* d_row_indices,
    const __nv_bfloat16* d_fc_in_w, const __nv_bfloat16* d_fc_out_w,
    __nv_bfloat16* d_mixed,
    int rows, int H, int Ie, int num_experts, int total_selected_rows,
    cudaStream_t stream
);

// FP8 grouped/device-side selected-expert forward -- same scheduling as the
// BF16 version above, real E4M3 weights (fc_in_w/fc_out_w, matching
// lt_gemm_fp8_nt's own weight slots) and E4M3-or-E5M2 activation input
// (fcin_is_e5m2 selects which). See moe.cu for the scoped precision note on
// why the GELU intermediate stays float rather than re-quantizing.
void moe_grouped_expert_forward_fp8(
    const void* d_hidden, const float* d_fcin_descale, int fcin_is_e5m2,
    const float* d_route_weights,
    const int* d_counts, const int* d_offsets, const int* d_row_indices,
    const void* d_fc_in_w, const float* d_fc_in_descale,
    const void* d_fc_out_w, const float* d_fc_out_descale,
    __nv_bfloat16* d_mixed,
    int rows, int H, int Ie, int num_experts, int total_selected_rows,
    cudaStream_t stream
);

void bf16_zero(__nv_bfloat16* d_x, std::size_t n, cudaStream_t stream);
void bf16_copy(const __nv_bfloat16* d_src, __nv_bfloat16* d_dst, std::size_t n, cudaStream_t stream);

// ── Cognitive-architecture sparse MoE port: backward (Step 5, 2026-07-22) ───

// GELU backward: needs the PRE-activation value (not recoverable from post
// alone) -- keep pre/post in separate buffers during forward recompute.
void gelu_backward(
    const __nv_bfloat16* d_pre, const __nv_bfloat16* d_grad_post, __nv_bfloat16* d_grad_pre,
    std::size_t n, cudaStream_t stream
);

// out[row*out_stride+out_col] = sum_h(a[row,h]*b[row,h]) -- writes one column
// of a [rows,num_experts]-shaped buffer directly (out_stride=num_experts,
// out_col=e). d(route_weight[:,e]) = dot(d_mixed, expert_out_e).
void moe_row_dot_bf16(
    const __nv_bfloat16* d_a, const __nv_bfloat16* d_b, float* d_out,
    int rows, int H, int out_stride, int out_col, cudaStream_t stream
);

// moe_small_proj_f32's backward: input-gradient and weight-gradient (FP32,
// += accumulate) halves, called separately since callers sometimes only
// need one (e.g. Identity branches skip the weight side entirely).
void moe_small_proj_backward_input(
    const float* d_out, const __nv_bfloat16* d_w, float* d_in,
    int rows, int in_dim, int out_dim, cudaStream_t stream
);
void moe_small_proj_backward_weight(
    const float* d_out, const float* d_x, float* d_w_grad,
    int rows, int in_dim, int out_dim, cudaStream_t stream
);
void moe_small_proj_backward_weight(
    const float* d_out, const float* d_x, __nv_bfloat16* d_w_grad,
    int rows, int in_dim, int out_dim, cudaStream_t stream
);

// tanh/sigmoid backward given the POST-activation value (both recoverable
// from post alone, unlike GELU).
void tanh_backward_f32(const float* d_post, const float* d_grad_post, float* d_grad_pre,
                        std::size_t n, cudaStream_t stream);
void sigmoid_backward_f32(const float* d_post, const float* d_grad_post, float* d_grad_pre,
                           std::size_t n, cudaStream_t stream);

// PressureField's hidden_gated = hidden*modulation backward: accumulates
// into d_hidden_accum (bf16, +=), overwrites d_modulation_grad_out (fresh).
void moe_modulate_hidden_backward(
    const __nv_bfloat16* d_hidden_gated, const __nv_bfloat16* d_hidden,
    const float* d_modulation, __nv_bfloat16* d_hidden_accum, float* d_modulation_grad_out,
    std::size_t n, cudaStream_t stream
);

// Generic "y = x/row_sum(x)" backward: d(x_i) = (d(y_i) - dot_row(d(y),y)) /
// row_sum. Reused for both the top-k renormalize and lateral inhibition's
// final renormalize.
void moe_norm_by_sum_backward_f32(
    const float* d_y, const float* d_dy, const float* d_row_sum, float* d_dx,
    int rows, int num_experts, cudaStream_t stream
);

// Zeros d_x wherever `reference` is exactly 0 -- masks out the topk scatter's
// unselected positions (a hard graph constant, not a real gradient path).
void moe_zero_where_zero_f32(const float* d_reference, float* d_x, std::size_t n, cudaStream_t stream);

// LateralInhibition's clamp backward: zeroes wherever the forward's max-cap
// or min-floor actually fired (needs c1 = clamp(p,max=dominance_cap) saved
// from forward, the value BEFORE the min-floor-raise).
void moe_inhib_clamp_backward_f32(
    const float* d_c1, float dominance_cap, float minority_floor, float* d_bounded_inout,
    std::size_t n, cudaStream_t stream
);

// Per-sample-start position counts from segs (B,S) -- shared by both
// backward pooling reductions in one layer (pooled and pooled2 use the same
// segs, hence identical counts); compute once, reuse for both.
void moe_compute_sample_counts(const std::uint16_t* d_segs, float* d_count, int B, int S, cudaStream_t stream);

// Backward of moe_pool_by_sample+moe_gather_by_sample together: scatter-add
// the per-position gathered-gradient back to its sample-start slot, divide
// by count, broadcast-add into d_hidden_grad_accum (bf16, +=). d_count must
// already hold this segs layout's counts (moe_compute_sample_counts).
void moe_pool_by_sample_backward(
    const float* d_grad_gathered, const std::uint16_t* d_segs, const float* d_count,
    float* d_start_grad_scratch, __nv_bfloat16* d_hidden_grad_accum,
    int B, int S, int H, cudaStream_t stream
);

// Compatibility contract for a bounded activation-quality probe. The public
// implementation does not expose private quantization math.
constexpr float kMoeFp4Max = 7.0f;
constexpr float kMoeInt2Max = 1.0f;
void moe_fake_quant_roundtrip(
    const __nv_bfloat16* d_x, __nv_bfloat16* d_out, float* d_amax_scratch,
    float max_level, std::size_t n, cudaStream_t stream
);

// ── Native packed-FP4 activation path (real storage reduction, not a probe) ─
// Pack: bf16 -> two signed 4-bit codes per byte (adjacent pairs from the
// SAME tensor -- half the bytes of FP8 storage). Delayed-scaling, same
// ActSlot fields (amax/scale/descale/scale_snapshot/descale_snapshot) as
// the direct-FP8 path reuse -- is_recompute=true reuses scale_snapshot
// instead of recomputing amax, matching fp8_quant_act's own contract.
void moe_pack_fp4_act(
    float* d_amax, float* d_scale, float* d_descale,
    float* d_scale_snapshot, float* d_descale_snapshot,
    const __nv_bfloat16* d_x, std::uint8_t* d_packed_out,
    std::size_t n, bool is_recompute, cudaStream_t stream
);

// Decode: packed FP4 -> FP8 e4m3, ready for the existing lt_gemm_fp8_nt
// call sites unchanged.
void moe_decode_packed_fp4_to_e4m3(
    const std::uint8_t* d_packed, __nv_fp8_e4m3* d_out, const float* d_descale,
    std::size_t n, cudaStream_t stream
);

// ── Causal attention softmax ─────────────────────────────────────────────────
// d_scores: [rows, S] where rows = B*nH*S and query position = row % S.
// Columns > position masked to 0 (causal LM attention).
void attn_causal_softmax_f32(float* d_scores, int rows, int S, cudaStream_t stream);

// Softmax backward: d_grad ← P ⊙ (d_grad − Σ d_grad·P) per row, in place.
// d_probs = forward probabilities (untouched), d_grad = dL/dP in, dL/dS out.
void attn_softmax_backward_f32(
    const float* d_probs, float* d_grad, int rows, int S, cudaStream_t stream
);

// ── Global gradient norm ─────────────────────────────────────────────────────
// Accumulates Σ g² into *d_acc (caller zeroes d_acc first, then sqrt on host).
void sq_sum_acc_f32(const float* d_g, std::size_t n, float* d_acc, cudaStream_t stream);

// ── Cross-entropy loss ───────────────────────────────────────────────────────
// logits:     [B*S, V]  (bf16)
// labels:     [B*S]     (int32, -100 = ignore)
// d_loss_out: [1]       (float, mean over non-ignored tokens)
// d_grad:     [B*S, V]  (bf16, d(loss)/d(logits), already divided by N_valid)
void cross_entropy_fwd_bwd(
    const __nv_bfloat16* d_logits,
    const int32_t*       d_labels,
    float*               d_loss_out,
    __nv_bfloat16*       d_grad,
    int BS, int V,
    cudaStream_t stream
);

// ── Chunked cross-entropy (fused-classifier path) ────────────────────────────
// Count non-ignored labels into *d_n_valid (caller zeroes it first).
void cross_entropy_count_valid(
    const int32_t* d_labels, int n, int* d_n_valid, cudaStream_t stream
);
// CE fwd+bwd for a row chunk; gradient and loss scaled by 1/(*d_n_valid) on
// device — chunks are independent, no rescale pass, no host sync.
// d_grad may alias d_logits (in-place).
void cross_entropy_fwd_bwd_chunk(
    const __nv_bfloat16* d_logits,
    const int32_t*       d_labels,
    float*               d_loss_out,
    __nv_bfloat16*       d_grad,
    int rows, int V,
    const int* d_n_valid,
    cudaStream_t stream
);

// ── Flash-style causal attention ─────────────────────────────────────────────
// q/k/v/o/do/dq/dk/dv: [BH, S, Hd] BF16 (all sources/sinks are BF16 — FP32
// operand buffers were pure bandwidth waste); accumulation stays FP32 in
// registers/SMEM.  lse/rowdot: [BH, S] FP32.
// Never materializes the [S, S] score matrix.

// Sliding-window width in tokens.  Unset env = Era-13 recipe default per
// head dim (128 at Hd>=256, 256 below); IDA_NATIVE_ATTN_WINDOW overrides;
// =0 forces full causal-within-segment.  One resolution for every attention
// kernel, scalar and WGMMA — the window is part of the forward/backward
// surface contract.  Defined in attention.cu.
int attn_window_tokens(int Hd);
// Per-burn override (>= 0) takes priority over both the env var and the
// head-dim heuristic above; call once per burn, before any attention
// kernel runs, with the request's local_attention_window (or -1 to clear
// / fall back to the pre-existing resolution). thread_local -- safe under
// the shared multi-tenant server's per-body_key burn threads. Defined in
// attention.cu.
void set_attn_window_request_override(int window);

// ── LRSS: temporal trace emitter + multiscale memory bank (lrss.cu) ─────────
// Contract: docs/lrss-native-port-contract.md.  Weights BF16 (safetensors
// contract), internal compute FP32, grads FP32 (param-slot contract).
struct LrssParams {
    const __nv_bfloat16* query_w{};   // [H,H]  torch orientation [out,in]
    const __nv_bfloat16* key_w{};     // [H,H]
    const __nv_bfloat16* gate_w{};    // [H,2H]
    const __nv_bfloat16* gate_b{};    // [H]
    const __nv_bfloat16* log_tau{};   // [J]
    const __nv_bfloat16* scale_w{};   // [J]
    int num_scales{8};
    float tau_min{1.0f}, tau_max{64.0f};
    // LSS — LowRankStateSupersampler head (IDA_NATIVE_LSS=1, requires the
    // bank).  Reconstructs the memory mix from the int2-quantized anchor
    // store + current pooled state: recon = anchor + up(relu(down(joint))).
    const __nv_bfloat16* lss_down{};  // [R, 2H (+ pss_spike_dim)]
    const __nv_bfloat16* lss_up{};    // [H, R]
    int lss_rank{0};                  // 0 = LSS off
    // The repaired inject path receives task and reconstruction gradients.
    // H100 isolation found 0.1 dominated the shared global clip; 0.01 kept
    // the auxiliary trainable without suppressing the task gradient.
    float lss_aux_weight{0.01f};
    // PSS Stage 1b: LRSS's per-slot spike-ratio bucket vector, appended to
    // the LSS joint input as [pooled; anchor; spike]. 0 = feature off
    // (checkpoint-shape-neutral); kPssSpikeBuckets when
    // IDA_NATIVE_PSS_SPIKE_JOINT=1. Value must match LatticeWeights::
    // pss_spike_joint_dim -- that is what actually sized lss_down/lss_joint.
    int pss_spike_dim{0};
};
// Bucket count for the PSS spike-feed conditioning vector: embed, lm_head,
// norms, attn-qk, attn-v/o, ffn, lrss-bank, lss-head.
inline constexpr int kPssSpikeBuckets = 8;
struct LrssGrads {
    float* query_w{};  float* key_w{};  float* gate_w{};
    float* gate_b{};   float* log_tau{}; float* scale_w{};
    float* lss_down{}; float* lss_up{};
};
struct LrssScratch {
    // ring (persistent across micro-steps)
    float* ring{};        // [max_anchors, H] device
    int    times[64]{};   // host causal times per slot
    int    cursor{0}, count{0};
    // per-forward view + saved-for-backward tensors
    float* bank{};        // [A,H] newest-first
    float* elapsed{};     // [A]
    int    bank_count{0};
    bool   applied{false};
    int    B_last{0};
    float *pooled{}, *n_content{}, *q{}, *k{}, *rel{}, *attn{}, *sw{},
          *wsum{}, *mix{}, *cat{}, *gpre{}, *gate{}, *delta{};
    // backward scratch
    float *d_delta{}, *dpre{}, *d_mix{}, *dcat{}, *d_pooled{}, *dwsum{},
          *drel{}, *dot_j{}, *dq{}, *dk{}, *dq_in{};
    // LSS head: int2 anchor store (the coarse operand) + reconstruction
    signed char* lss_anchor_q{};   // [H] int2 codes in {-3,-1,1,3} (÷3 × scale)
    float* lss_anchor_scale{};     // [1] per-anchor amax scale
    int    lss_anchor_valid{0};    // 0 until the first anchor is stored
    bool   lss_active_this_step{false};
    int    lss_aux_copy_pending{0}; // async evidence copy awaits an existing stream fence
    int    lss_aux_valid{0};       // host telemetry has a measured auxiliary loss
    int    lss_feedback_ready{0};  // measured aux may govern the feedback policy
    float  lss_last_aux{0.0f};     // host-side measured reconstruction loss
    float  lss_last_recon_norm{0.0f};
    float  lss_last_target_norm{0.0f};
    float  lss_last_relative_rmse{0.0f};
    float *lss_anchor{},           // [H] dequantized anchor
          *lss_joint{},            // [B, 2H] [pooled; anchor]
          *lss_hidden{},           // [B, R] relu(down·joint)
          *lss_recon{},            // [B, H] anchor + up·hidden
          *lss_aux{},              // [1] aux reconstruction loss
          *lss_recon_norm{},       // [1] mean squared reconstruction norm
          *lss_target_norm{},      // [1] mean squared target norm
          *d_lss_recon{},          // [B, H]
          *d_lss_hidden{},         // [B, R]
          *mix_bank_save{};        // [B, H] bank mix before injection (injection mode only)
    bool lss_injected{false};      // true if injection replaced s.mix this step
    // PSS Stage 1b: host-staged spike-ratio bucket vector, async-copied here
    // once per optimizer step (the existing heartbeat/window fence -- no new
    // sync) and read by every micro-step's LSS joint until the next update.
    // One-step-lag by construction, same as the anchor ring.
    float* pss_spike{};            // [kPssSpikeBuckets] device, null when off
    int    pss_spike_valid{0};     // 0 until the first window has published
};
void lrss_forward(const LrssParams& p, LrssScratch& s, __nv_bfloat16* hidden,
                  const std::uint32_t* tokens, const std::uint8_t* common_mask,
                  int B, int S, int H, cudaStream_t st, bool is_recompute = false);
void lrss_record_anchor(LrssScratch& s, int max_anchors, int H,
                        int causal_time, cudaStream_t st);
void lrss_backward(const LrssParams& p, LrssScratch& s, LrssGrads& g,
                   __nv_bfloat16* d_hidden, const std::uint32_t* tokens,
                   const std::uint8_t* common_mask,
                   int B, int S, int H, cudaStream_t st);
void lrss_refresh_bank(LrssScratch& s, int max_anchors, int H, int now,
                       cudaStream_t st);
void k_lrss_split_add(float* y, const float* x, std::size_t n, cudaStream_t st);
void flash_attn_forward(
    const __nv_bfloat16* d_q, const __nv_bfloat16* d_k, const __nv_bfloat16* d_v,
    __nv_bfloat16* d_o, float* d_lse,
    int BH, int S, int Hd, float scale,
    cudaStream_t stream
);
void flash_attn_backward(
    const __nv_bfloat16* d_q, const __nv_bfloat16* d_k, const __nv_bfloat16* d_v,
    const __nv_bfloat16* d_o, const __nv_bfloat16* d_do, const float* d_lse,
    __nv_bfloat16* d_dq, __nv_bfloat16* d_dk, __nv_bfloat16* d_dv,
    float* d_rowdot,   // scratch [BH * S]
    int BH, int S, int Hd, float scale,
    cudaStream_t stream
);
// d_segs: optional per-position sample-start offsets [B, S] (u16, B = BH/nH).
// Position i may attend only to j in [segs[i], i] — block-diagonal causal
// attention over packed samples.  nullptr = full causal (old behavior).
// nKVH (GQA, grouped-query attention): number of K/V heads.  0 (default)
// means MHA (nKVH = nH).  When 0 < nKVH < nH, K/V (and dK/dV) are laid out
// [B, nKVH, S, Hd] and each group of nH/nKVH query heads shares one KV head.
// Scalar-flash backend only; every other backend throws on a grouped request.
void attention_forward(
    AttentionBackendKind backend,
    const __nv_bfloat16* d_q, const __nv_bfloat16* d_k, const __nv_bfloat16* d_v,
    __nv_bfloat16* d_o, float* d_lse,
    const PackedFp4AttentionOperands* packed_fp4,
    const std::uint16_t* d_segs, int nH,
    int BH, int S, int Hd, float scale,
    cudaStream_t stream, int nKVH = 0
);
void attention_backward(
    AttentionBackendKind backend,
    const __nv_bfloat16* d_q, const __nv_bfloat16* d_k, const __nv_bfloat16* d_v,
    const __nv_bfloat16* d_o, const __nv_bfloat16* d_do, const float* d_lse,
    __nv_bfloat16* d_dq, __nv_bfloat16* d_dk, __nv_bfloat16* d_dv,
    float* d_rowdot,
    const PackedFp4AttentionOperands* packed_fp4,
    const std::uint16_t* d_segs, int nH,
    int BH, int S, int Hd, float scale,
    cudaStream_t stream, int nKVH = 0
);

// Private advanced attention contract. The public build supplies a fail-closed
// definition; implementation details remain deployment-owned.
void wgmma_flash_forward_hd64(
    const std::uint8_t* d_qk_packed,
    const float* d_q_descale,
    const float* d_k_descale,
    float* d_q_scratch,
    float* d_k_scratch,
    float* d_q_tile_stage,
    float* d_k_tile_stage,
    float* d_v_tile_stage,
    int tma_stage_depth,
    const __nv_bfloat16* d_v,
    __nv_bfloat16* d_o,
    float* d_lse,
    int BH, int S, int Hd, float sm_scale,
    cudaStream_t stream,
    const std::uint8_t* d_q_saved_e4m3 = nullptr,
    const std::uint8_t* d_k_saved_e4m3 = nullptr,
    // Reserved for a private deployment-owned decode contract.
    const float* d_q_mean = nullptr,
    const float* d_k_mean = nullptr,
    // Sample-boundary starts [B, S] for block-diagonal culling (nullptr = full causal).
    const std::uint16_t* d_segs = nullptr,
    int nH = 1
);
// Private advanced backward contract (stage A).
void wgmma_flash_bwd_dkv_hd64(
    const __nv_bfloat16* d_q,
    const __nv_bfloat16* d_k,
    const __nv_bfloat16* d_v,
    const __nv_bfloat16* d_do,
    const float* d_lse,
    const float* d_rowdot,
    const std::uint16_t* d_segs, int nH,
    __nv_bfloat16* d_dk,
    __nv_bfloat16* d_dv,
    int BH, int S, int Hd, float sm_scale,
    cudaStream_t stream
);

// Private advanced backward contract (stage B).
void wgmma_flash_bwd_dq(
    const __nv_bfloat16* d_q,
    const __nv_bfloat16* d_k,
    const __nv_bfloat16* d_v,
    const __nv_bfloat16* d_do,
    const float* d_lse,
    const float* d_rowdot,
    const std::uint16_t* d_segs, int nH,
    __nv_bfloat16* d_dq,
    int BH, int S, int Hd, float sm_scale,
    cudaStream_t stream
);

// Bounded backward-health telemetry. The public API exposes aggregate values;
// private counters and raw diagnostics are not part of this source surface.
void attn_bwd_health_reset(cudaStream_t stream);
void attn_bwd_health_read(double* p_zero_frac, float* drift_max,
                          cudaStream_t stream);
// Scalar-side bounded counters used by the public aggregate implementation.
void attn_bwd_health_reset_scalar(cudaStream_t stream);
void attn_bwd_health_read_raw_scalar(
    unsigned long long* p_zero, unsigned long long* p_total,
    unsigned int* drift_key, cudaStream_t stream
);

// Private precision-compatibility contracts; public definitions fail closed.
void roundtrip_f32_through_e4m3(float* d_x, std::size_t n, cudaStream_t stream);
void roundtrip_bf16_through_e4m3(__nv_bfloat16* d_x, std::size_t n, cudaStream_t stream);
void roundtrip_bf16_through_scaled_e4m3(__nv_bfloat16* d_x, std::size_t n, cudaStream_t stream);
void fp8_e4m3_pair_to_bf16(
    const std::uint8_t* d_in0,
    const std::uint8_t* d_in1,
    __nv_bfloat16* d_out0,
    __nv_bfloat16* d_out1,
    std::size_t n,
    cudaStream_t stream
);

// Private advanced FP8 attention contract. Public definitions fail closed.
void wgmma_flash_forward_bf16src(
    const __nv_bfloat16* d_q,
    const __nv_bfloat16* d_k,
    const __nv_bfloat16* d_v,
    float* d_q_scratch,
    float* d_k_scratch,
    __nv_bfloat16* d_o,
    float* d_lse,
    int BH, int S, int Hd, float sm_scale,
    const std::uint16_t* d_segs, int nH,
    cudaStream_t stream
);

// ── Private precision compatibility contracts ─────────────────────────────────
// These symbols are retained for ABI/source compatibility. Public builds do
// not expose the implementation or permit caller-selected activation math.
void fp4_scale_from_amax(
    const float* d_amax, float* d_scale, float* d_descale, cudaStream_t s);
void fp4_pack_pair_e2m1_record(
    const __nv_bfloat16* d_x0,
    const __nv_bfloat16* d_x1,
    std::uint8_t* d_out,
    const float* d_scale0,
    const float* d_scale1,
    float* d_amax0,
    float* d_amax1,
    std::size_t n,
    cudaStream_t s
);
// Private activation-format compatibility contract.
void fp4_scale_from_amax_e2m1(
    const float* d_amax, float* d_scale, float* d_descale, cudaStream_t s);
void fp4_pack_pair_e2m1_true_record(
    const __nv_bfloat16* d_x0,
    const __nv_bfloat16* d_x1,
    std::uint8_t* d_out,
    const float* d_scale0,
    const float* d_scale1,
    float* d_amax0,
    float* d_amax1,
    std::size_t n,
    cudaStream_t s
);
// Private diagnostic-format compatibility contract.
void fp4_scale_from_amax_int2(
    const float* d_amax, float* d_scale, float* d_descale, cudaStream_t s);
void fp4_pack_pair_int2_record(
    const __nv_bfloat16* d_x0,
    const __nv_bfloat16* d_x1,
    std::uint8_t* d_out,
    const float* d_scale0,
    const float* d_scale1,
    float* d_amax0,
    float* d_amax1,
    std::size_t n,
    cudaStream_t s
);
// Private calibration-format compatibility contract.
void fp4_scale_from_amax_gauss(
    const float* d_amax, float* d_scale, float* d_descale, cudaStream_t s);
void fp4_pack_pair_gauss_record(
    const __nv_bfloat16* d_x0,
    const __nv_bfloat16* d_x1,
    std::uint8_t* d_out,
    const float* d_scale0,
    const float* d_scale1,
    float* d_amax0,
    float* d_amax1,
    std::size_t n,
    cudaStream_t s
);
// Private calibration contract retained for compatibility.
void fp4_amax_pair(
    const __nv_bfloat16* d_x0,
    const __nv_bfloat16* d_x1,
    float* d_amax0,
    float* d_amax1,
    std::size_t n,
    cudaStream_t s
);
// Private centered-calibration contract retained for compatibility.
void fp4_scale_from_stats_centered(
    const float* d_sum, const float* d_max_abs_dev, float n,
    float* d_mean, float* d_scale, float* d_descale, cudaStream_t s);
// Private smoothed-calibration contract retained for compatibility.
void fp4_scale_from_stats_centered_smooth(
    const float* d_sum, const float* d_max_abs_dev, float n,
    float* d_mean, float* d_scale, float* d_descale, cudaStream_t s);
void fp4_pack_pair_centered_record(
    const __nv_bfloat16* d_x0,
    const __nv_bfloat16* d_x1,
    std::uint8_t* d_out,
    const float* d_mean0,
    const float* d_mean1,
    const float* d_scale0,
    const float* d_scale1,
    float* d_sum0,
    float* d_sum1,
    float* d_max_abs_dev0,
    float* d_max_abs_dev1,
    std::size_t n,
    cudaStream_t s
);
// Private centered-format compatibility contract.
void fp4_scale_from_stats_centered_int2(
    const float* d_sum, const float* d_max_abs_dev, float n,
    float* d_mean, float* d_scale, float* d_descale, cudaStream_t s);
void fp4_pack_pair_centered_int2_record(
    const __nv_bfloat16* d_x0,
    const __nv_bfloat16* d_x1,
    std::uint8_t* d_out,
    const float* d_mean0,
    const float* d_mean1,
    const float* d_scale0,
    const float* d_scale1,
    float* d_sum0,
    float* d_sum1,
    float* d_max_abs_dev0,
    float* d_max_abs_dev1,
    std::size_t n,
    cudaStream_t s
);
void fp4_unpack_pair_to_bf16(
    const std::uint8_t* d_in,
    __nv_bfloat16* d_out0,
    __nv_bfloat16* d_out1,
    const float* d_descale0,
    const float* d_descale1,
    std::size_t n,
    cudaStream_t s,
    const float* d_mean0 = nullptr,
    const float* d_mean1 = nullptr
);
void fp4_unpack_pair_to_bf16_via_e4m3(
    const std::uint8_t* d_in,
    __nv_bfloat16* d_out0,
    __nv_bfloat16* d_out1,
    const float* d_descale0,
    const float* d_descale1,
    std::size_t n,
    cudaStream_t s,
    const float* d_mean0 = nullptr,
    const float* d_mean1 = nullptr
);
void fp4_unpack_pair_to_f32(
    const std::uint8_t* d_in,
    float* d_out0,
    float* d_out1,
    const float* d_descale0,
    const float* d_descale1,
    std::size_t n,
    cudaStream_t s
);

void fp8_amax_bf16(const __nv_bfloat16* d_x, std::size_t n, float* d_amax, cudaStream_t s);
// Architecture-independent raw-byte E4M3 storage path for Ampere. The
// caller supplies BF16 master weights; pack computes one per-tensor scale,
// rejects nonfinite input through d_bad_count, and writes no CUDA FP8 tensor
// object. Unpack reconstructs BF16 staging values using d_descale.
void fp8_pack_bf16_e4m3_raw(
    const __nv_bfloat16* d_x, std::uint8_t* d_out,
    float* d_amax, float* d_scale, float* d_descale, int* d_bad_count,
    std::size_t n, cudaStream_t s);
// Fused refresh tail: encode the raw E4M3 byte and reconstruct the BF16
// staging value in the same pass. The durable master remains BF16; this only
// removes the packed-byte global-memory round trip and second kernel launch.
void fp8_pack_bf16_e4m3_raw_and_dequant(
    const __nv_bfloat16* d_x, std::uint8_t* d_out, __nv_bfloat16* d_dequant,
    float* d_amax, float* d_scale, float* d_descale, int* d_bad_count,
    std::size_t n, cudaStream_t s);
void fp8_unpack_e4m3_raw_bf16(
    const std::uint8_t* d_in, const float* d_descale,
    __nv_bfloat16* d_out, std::size_t n, cudaStream_t s);
void fp8_scale_from_amax(
    const float* d_amax, float fp8_max, float* d_scale, float* d_descale, cudaStream_t s);
void fp8_quantize_e4m3(
    const __nv_bfloat16* d_x, void* d_out, const float* d_scale, std::size_t n, cudaStream_t s);
void fp8_quantize_e5m2(
    const __nv_bfloat16* d_x, void* d_out, const float* d_scale, std::size_t n, cudaStream_t s);
// w[K,N] row-major → out[N,K] row-major (forward A@Bᵀ layout), quantized.
void fp8_quantize_transpose_e4m3(
    const __nv_bfloat16* d_w, void* d_out, const float* d_scale, int K, int N, cudaStream_t s);
// Fused: in[K,N] BF16 → act8_out[K,N] E4M3 (non-transposed) + snap_out[N,K] E4M3 (transposed).
// Replaces fp8_quant_act + transpose_u8 in the backward recompute snap path;
// reads BF16 once and writes both outputs through a shared-memory tile,
// saving the act8 HBM read-back that the two-step version requires.
void fp8_quant_and_transpose_e4m3(
    const __nv_bfloat16* d_in, void* d_act8_out, void* d_snap_out,
    const float* d_scale, int K, int N, cudaStream_t s);
// Delayed scaling: quantize with the previous scale, record current amax
// (single pass; caller zeroes d_amax first and derives the next scale after
// the consuming GEMM is enqueued).
void fp8_quantize_e4m3_record(
    const __nv_bfloat16* d_x, void* d_out, const float* d_scale, float* d_amax,
    std::size_t n, cudaStream_t s);
void fp8_quantize_e5m2_record(
    const __nv_bfloat16* d_x, void* d_out, const float* d_scale, float* d_amax,
    std::size_t n, cudaStream_t s);
// Update a delayed-scaling slot after a record pass: append the current amax to
// a short history window, derive the next scale/descale from the window max,
// and advance the ring cursor. Caller is expected to have just written d_amax
// on the same stream.
// Delayed-scaling saturation probe: records whether a call clipped, i.e.
// recorded_amax * scale_used > fp8_max. stats = [clipped_calls, max_ratio, total_calls].
void fp8_clip_probe(
    const float* d_amax, const float* d_scale, float fp8_max,
    float* d_stats, cudaStream_t s
);
// Element-level FP8 saturation counter (debug): out = [clipped_elems, total_elems].
void fp8_elem_clip_count(
    const __nv_bfloat16* d_x, const float* d_scale, float fp8_max,
    std::size_t n, float* d_out, cudaStream_t s
);
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
);

// ── AdamW weight update ───────────────────────────────────────────────────────
// Updates weight (BF16) in-place using either FP32 or BF16 resident m/v state.
// BF16 writes use stochastic rounding.  If d_weight_amax is non-null, records
// max|w_new| into it (caller zeroes it first) — free FP8 weight amax.
void adamw_step(
    __nv_bfloat16* d_w,
    float*         d_m,
    float*         d_v,
    const float*   d_grad,   // FP32 gradient
    std::size_t    n,
    float lr, float beta1, float beta2, float eps, float wd,
    int step,
    cudaStream_t stream,
    float*        d_weight_amax = nullptr,
    unsigned      sr_seed = 0
);
void adamw_step(
    __nv_bfloat16* d_w,
    __nv_bfloat16* d_m,
    __nv_bfloat16* d_v,
    const float*   d_grad,
    std::size_t    n,
    float lr, float beta1, float beta2, float eps, float wd,
    int step,
    cudaStream_t stream,
    float*        d_weight_amax = nullptr,
    unsigned      sr_seed = 0
);

// ── Lion weight update (IDA_NATIVE_OPTIMIZER=lion, default off) ───────────────
// EvoLved Sign Momentum (Chen et al. 2023): single momentum buffer, no
// second moment, no bias correction. beta1 blends the update DIRECTION,
// beta2 updates the STORED momentum -- deliberately distinct roles, not
// aliased to AdamW's beta pair. Same fused SR-BF16 weight write + free FP8
// amax pass as adamw_step.
void lion_step(
    __nv_bfloat16* d_w,
    float*         d_m,
    const float*   d_grad,
    std::size_t    n,
    float lr, float beta1, float beta2, float wd,
    cudaStream_t stream,
    float*        d_weight_amax = nullptr,
    unsigned      sr_seed = 0
);
void lion_step(
    __nv_bfloat16* d_w,
    __nv_bfloat16* d_m,
    const float*   d_grad,
    std::size_t    n,
    float lr, float beta1, float beta2, float wd,
    cudaStream_t stream,
    float*        d_weight_amax = nullptr,
    unsigned      sr_seed = 0
);

// Cast BF16 gradient → FP32 (for weight update path)
void cast_bf16_to_f32(
    const __nv_bfloat16* d_src,
    float*               d_dst,
    std::size_t          n,
    cudaStream_t         stream
);

// Cast FP32 → BF16 (for reading weights into compute)
void cast_f32_to_bf16(
    const float*   d_src,
    __nv_bfloat16* d_dst,
    std::size_t    n,
    cudaStream_t   stream
);

}  // namespace ida_native
