#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "ida_native/arena.hpp"
#include "ida_native/request.hpp"

namespace ida_native {

struct BurnResult {
    int global_step{0};
    int optimizer_steps{0};
    std::size_t tokens_processed{0};
    double elapsed_seconds{0.0};
    double tokens_per_second{0.0};
    std::size_t device_bytes_touched{0};
    float final_loss{0.0f};
    float final_loss_ema{0.0f};
    float final_grad_norm{0.0f};
    float final_lr{0.0f};
    int skipped_steps{0};
    int global_grad_clip_steps{0};
    int embed_row_clip_steps{0};
    int embed_row_clipped_rows_total{0};
    bool final_global_grad_clip_fired{false};
    float final_global_grad_clip_scale{1.0f};
    bool final_embed_row_clip_fired{false};
    float final_embed_row_clip_threshold{0.0f};
    float final_embed_row_clip_max_preclip_norm{0.0f};
    int final_embed_row_clipped_rows{0};
    std::string final_dominant_grad_slot{};
    float final_dominant_grad_slot_norm{0.0f};
    float final_dominant_grad_slot_frac{0.0f};
    // PSS predictor's own gradient norm (pred_down+pred_up, L2-combined),
    // reported unconditionally -- see trainer.cu's computation site.
    float final_pss_predictor_grad_norm{0.0f};
    float final_pss_predictor_grad_frac{0.0f};
    int final_lss_feedback_skip_tail{0};
    float final_lss_feedback_residual_scale{1.0f};
    float final_lss_aux{0.0f};
    float final_lss_recon_norm{0.0f};
    float final_lss_target_norm{0.0f};
    float final_lss_relative_rmse{0.0f};
    float final_support_transition_scale{1.0f};
    // PSS Stage 1a: per-slot spike detection + adaptive clip (shadow-safe,
    // default off via IDA_NATIVE_PSS_SLOT_CLIP).
    int final_pss_slot_clip_steps{0};
    float final_pss_spike_ratio_max{0.0f};
    std::string final_pss_spike_slot{};
    // PSS Stage 2: tail-layer FFN-output predictor, shadow-mode scoring
    // (default off via IDA_NATIVE_PSS_PRED). Snapshot of the last
    // micro-step's window, not window-averaged.
    float final_pss_confidence{0.0f};
    float final_pss_pred_err{0.0f};
    // Four-state shadow score. Observational only: the Stage-4 governor
    // continues to consume final_pss_confidence until separately promoted.
    float final_pss_int2_agreement{0.0f};
    // PSS Stage 4: governor engagement fraction (default off via
    // IDA_NATIVE_PSS_GOVERNOR).
    float final_pss_engaged_frac{0.0f};
    // PSS fence evidence. Ratios above are intentionally retained for
    // compatibility; these fields preserve the denominators, decision
    // provenance, and resolved auxiliary policy needed to interpret them.
    std::uint32_t final_pss_covered{0};
    std::size_t final_pss_n{0};
    std::uint32_t final_pss_int2_matched{0};
    std::uint32_t final_pss_int2_scored{0};
    int final_pss_scored_micros{0};
    float final_pss_int2_inv_rms{0.0f};
    float final_pss_confidence_min{0.0f};
    float final_pss_confidence_max{0.0f};
    float final_pss_blend_delta_rms{0.0f};
    float final_pss_aux_weight{0.0f};
    float final_pss_aux_denom{0.0f};
    bool final_pss_aux_normalize{false};
    bool final_pss_conditioning_active{false};
    bool final_pss_override_active{false};
    float final_pss_effective_confidence{0.0f};
    std::string final_pss_governor_event{"none"};
    std::string final_pss_conditioning_mode{"auto"};
    std::string final_pss_aux_normalize_mode{"auto"};
    bool fp8_active{false};
    std::string attention_backend{"scalar_flash"};
    std::string precision_profile{"legacy_fp8"};
    // Backward surface health (WGMMA backward only; 0/0 when scalar).
    double attn_p_zero_frac{0.0};
    float  attn_lse_drift_max{0.0f};
    // Two-stage runs retain each device's independently measured health.
    // The aggregate fields above report the conservative worst stage, not a
    // fabricated weighted average without the raw counter denominators.
    double attn_p_zero_frac_stage0{0.0};
    double attn_p_zero_frac_stage1{0.0};
    float  attn_lse_drift_max_stage0{0.0f};
    float  attn_lse_drift_max_stage1{0.0f};
    // Native two-stage peer pipeline. A completed result is intentionally not
    // checkpoint-resumable until sharded checkpoint serialization lands.
    bool model_parallel{false};
    int pipeline_stage_count{1};
    int pipeline_split_layer{0};
    int pipeline_first_device{-1};
    int pipeline_second_device{-1};
    std::string peer_transport{"none"};
    std::size_t peer_forward_bytes{0};
    std::size_t peer_backward_bytes{0};
};

// Backward-compat alias used by main.cpp.
using SmokeStepResult = BurnResult;

// Heartbeat payload delivered per optimizer step (at ≥5s intervals).
struct ProgressReport {
    int micro_step{0};          // micro-steps completed (data exposure)
    int optimizer_step{0};
    int active_grad_accum{0};
    int requested_grad_accum{0};
    std::size_t tokens{0};
    double elapsed_s{0.0};
    float loss{0.0f};
    float loss_ema{0.0f};
    float grad_norm{0.0f};
    float lr{0.0f};
    int effective_batch{0};     // microbatch × current accumulation
    int skipped_steps{0};
    int global_grad_clip_steps{0};
    int embed_row_clip_steps{0};
    int embed_row_clipped_rows_total{0};
    bool global_grad_clip_fired{false};
    float global_grad_clip_scale{1.0f};
    bool embed_row_clip_fired{false};
    float embed_row_clip_threshold{0.0f};
    float embed_row_clip_max_preclip_norm{0.0f};
    int embed_row_clipped_rows{0};
    std::string dominant_grad_slot{};
    float dominant_grad_slot_norm{0.0f};
    float dominant_grad_slot_frac{0.0f};
    int lss_feedback_skip_tail{0};
    float lss_feedback_residual_scale{1.0f};
    float lss_aux{0.0f};
    float lss_recon_norm{0.0f};
    float lss_target_norm{0.0f};
    float lss_relative_rmse{0.0f};
    float support_transition_scale{1.0f};
    // PSS Stage 1a/2 telemetry -- see final_pss_* on BurnResult above.
    int pss_slot_clip_steps{0};
    float pss_spike_ratio_max{0.0f};
    std::string pss_spike_slot{};
    float pss_confidence{0.0f};
    float pss_pred_err{0.0f};
    float pss_int2_agreement{0.0f};
    float pss_engaged_frac{0.0f};
    std::uint32_t pss_covered{0};
    std::size_t pss_n{0};
    std::uint32_t pss_int2_matched{0};
    std::uint32_t pss_int2_scored{0};
    int pss_scored_micros{0};
    float pss_int2_inv_rms{0.0f};
    float pss_confidence_min{0.0f};
    float pss_confidence_max{0.0f};
    float pss_blend_delta_rms{0.0f};
    float pss_aux_weight{0.0f};
    float pss_aux_denom{0.0f};
    bool pss_aux_normalize{false};
    bool pss_conditioning_active{false};
    bool pss_override_active{false};
    float pss_effective_confidence{0.0f};
    std::string pss_governor_event{"none"};
    std::string pss_conditioning_mode{"auto"};
    std::string pss_aux_normalize_mode{"auto"};
    bool fp8_active{false};
    std::string attention_backend{"scalar_flash"};
    std::string precision_profile{"legacy_fp8"};
};

// Must not throw.
using ProgressCallback = std::function<void(const ProgressReport&)>;

// ── Model weight tensors (all BF16, on device) ──────────────────────────────

struct LatticeLayerWeights {
    float expert_balancing_loss_coef{0.0f};
    __nv_bfloat16* attn_norm{nullptr};  // [H]
    __nv_bfloat16* q_proj{nullptr};     // [H, H]
    __nv_bfloat16* k_proj{nullptr};     // [H, kv_heads*Hd]
    __nv_bfloat16* v_proj{nullptr};     // [H, kv_heads*Hd]
    __nv_bfloat16* o_proj{nullptr};     // [H, H]
    // Qwen2-family QKV bias (2026-08-23; external-model portability). Never
    // allocated for native's own bodies (0/false = off, byte-identical to
    // before this field existed). No o_proj bias -- Qwen2Attention doesn't
    // have one either.
    __nv_bfloat16* q_bias{nullptr};  // [H]
    // GPT-2 projection/feed-forward biases; null for native/Qwen contracts.
    __nv_bfloat16* o_bias{nullptr};       // [H]
    __nv_bfloat16* attn_norm_bias{nullptr}; // [H]
    __nv_bfloat16* ffn_norm_bias{nullptr};  // [H]
    __nv_bfloat16* ffn_in_bias{nullptr};    // [I]
    __nv_bfloat16* ffn_out_bias{nullptr};   // [H]
    __nv_bfloat16* k_bias{nullptr};  // [kv_heads*Hd]
    __nv_bfloat16* v_bias{nullptr};  // [kv_heads*Hd]
    __nv_bfloat16* ffn_norm{nullptr};   // [H]
    __nv_bfloat16* gate_proj{nullptr};  // [H, I]
    __nv_bfloat16* up_proj{nullptr};    // [H, I]
    __nv_bfloat16* down_proj{nullptr};  // [I, H]
    // Cognitive-architecture sparse MoE port (2026-07-21). Per-layer, gated
    // by num_experts>0 (0 = off, byte-identical checkpoint, dense
    // gate/up/down above is used unchanged -- same convention as
    // lrss_enabled/pss_pred_rank on LatticeWeights below). No bias on any
    // of these (native's existing convention; see the port plan for why
    // PyTorch's bias=True isn't carried over).
    int num_experts{0};                 // num_personality_experts (Mode B) or num_cognitive_routes (Mode A)
    int expert_intermediate_size{0};    // per-expert width (personality_residual_expert_width in Mode B, else I)
    bool moe_shared_trunk{false};       // Mode B switch (use_personality_residual_experts)
    __nv_bfloat16* trunk_fc_in_w{nullptr};   // [I, H]   Mode B only, trunk width = I
    __nv_bfloat16* trunk_fc_out_w{nullptr};  // [H, I]
    __nv_bfloat16* expert_fc_in_w{nullptr};  // [num_experts, expert_intermediate_size, H] contiguous
    __nv_bfloat16* expert_fc_out_w{nullptr}; // [num_experts, H, expert_intermediate_size] contiguous
    // Routing (PressureField/ConstitutionalRouter, layer-local like the bank above)
    int num_routes{0};                  // num_cognitive_routes; pressure/router input+output width
    int top_k{0};                       // top_k_experts (Mode B) or top_k_routes (Mode A)
    __nv_bfloat16* pressure_proj_w{nullptr};      // [num_routes, H]
    __nv_bfloat16* pressure_mod_w{nullptr};       // [H, num_routes]
    __nv_bfloat16* router_score_w{nullptr};       // [num_experts, H]
    // nn.Identity() in the reference whenever num_routes==num_experts (the
    // common case) -- only allocated when they differ.
    __nv_bfloat16* pressure_to_routes_w{nullptr}; // [num_experts, num_routes], no bias

    // Generic per-token top-k SwiGLU MoE (2026-08-23; external-model
    // portability, see docs/ plan for "breadth beyond Llama-family"). A
    // SECOND, independent expert-bank mode living alongside the
    // cognitive-architecture one above -- gated by moe_kind, not by
    // num_experts alone, so every existing config (moe_kind stays
    // 0-initialized) is byte-identical to before this field existed.
    // Routing reuses num_experts/top_k/router_score_w AS-IS: standard
    // per-token softmax->top-k->renormalize (moe_topk_route_f32) needs
    // nothing pooling/pressure/lateral-inhibition-specific adds. Experts
    // use the SAME swiglu_forward/backward the dense FFN already uses
    // (silu(gate)*up -> down), matching HF Llama-MoE-family checkpoints
    // (Mixtral/Qwen-MoE/etc.) instead of IDA's own GELU 2-matrix experts.
    int moe_kind{0};  // 0 = legacy cognitive-architecture (pressure/pooling/lateral-inhibition), 1 = generic per-token top-k SwiGLU
    bool generic_moe_normalize_topk{true};
    __nv_bfloat16* generic_gate_proj_w{nullptr};  // [num_experts, expert_intermediate_size, H]
    __nv_bfloat16* generic_up_proj_w{nullptr};    // [num_experts, expert_intermediate_size, H]
    __nv_bfloat16* generic_down_proj_w{nullptr};  // [num_experts, H, expert_intermediate_size]

    // Shared expert (2026-08-23; Qwen2-MoE-family architecture feature):
    // an always-on expert applied to EVERY token alongside the top-k routed
    // ones above, additive to sb.ffn_out, scaled by a per-token sigmoid gate
    // (own [1,H] projection) -- NOT part of the softmax route-weight
    // distribution the routed experts share. 0 = off (byte-identical to
    // before this field existed); only meaningful when moe_kind==1.
    int shared_expert_width{0};  // shared expert's own SwiGLU intermediate width (may differ from expert_intermediate_size)
    __nv_bfloat16* generic_shared_gate_proj_w{nullptr};  // [shared_expert_width, H]
    __nv_bfloat16* generic_shared_up_proj_w{nullptr};    // [shared_expert_width, H]
    __nv_bfloat16* generic_shared_down_proj_w{nullptr};  // [H, shared_expert_width]
    __nv_bfloat16* generic_shared_gate_score_w{nullptr}; // [1, H]
};

struct LatticeWeights {
    __nv_bfloat16*      embed{nullptr};       // [V, H]
    __nv_bfloat16*      position_embeddings{nullptr}; // [max_position_embeddings, H]
    LatticeLayerWeights* layers{nullptr};      // host-side array of L structs, each field on device
    __nv_bfloat16*      final_norm{nullptr};  // [H]
    __nv_bfloat16*      final_norm_bias{nullptr}; // [H]
    __nv_bfloat16*      lm_head{nullptr};     // [V, H]
    // LRSS multiscale memory bank (IDA_NATIVE_LRSS=1; null when off).
    // docs/lrss-native-port-contract.md — model-level, not per-layer.
    __nv_bfloat16*      lrss_query{nullptr};  // [H, H]
    __nv_bfloat16*      lrss_key{nullptr};    // [H, H]
    __nv_bfloat16*      lrss_gate_w{nullptr}; // [H, 2H]
    __nv_bfloat16*      lrss_gate_b{nullptr}; // [H]
    __nv_bfloat16*      lrss_log_tau{nullptr};// [J]
    __nv_bfloat16*      lrss_scale_w{nullptr};// [J]
    // LSS supersampler head (IDA_NATIVE_LSS=1, requires lrss_enabled)
    __nv_bfloat16*      lss_down{nullptr};    // [R, 2H (+ pss_spike_joint_dim)]
    __nv_bfloat16*      lss_up{nullptr};      // [H, R]
    int  lss_rank{0};                          // 0 = off
    // PSS Stage 1b: 0 (default, off) or kPssSpikeBuckets when
    // IDA_NATIVE_PSS_SPIKE_JOINT=1. Set once at allocation time and never
    // changed for the life of the run -- every lss_down/lss_joint size
    // computation in this file reads it, so the checkpoint shape is a pure
    // function of this one field (0 = byte-identical to pre-PSS shape).
    int  pss_spike_joint_dim{0};
    // PSS Stage 2 (IDA_NATIVE_PSS_PRED, default off): shadow-mode low-rank
    // predictor of the tail layer's FFN output. down: [H,R] (in,out
    // convention, matches down_proj); up: [R,H]. pss_pred_rank=0 means the
    // whole head is absent -- byte-identical to pre-PSS shape.
    __nv_bfloat16* pss_pred_down{nullptr};  // [H, R]
    __nv_bfloat16* pss_pred_up{nullptr};    // [R, H]
    int  pss_pred_rank{0};
    // Standalone PSS predictor state (IDA_NATIVE_PSS_PRED_INIT_PATH): the
    // loaded state's cumulative optimizer-step count, carried forward so the
    // end-of-burn save reports total accumulated training, not just this
    // run's. 0 = fresh head (no state loaded, or the load RESET).
    int  pss_pred_prior_opt_steps{0};
    bool lrss_enabled{false};
    int  lrss_scales{8};
    int  lrss_anchors{32};
    int hidden_size{0};
    int intermediate_size{0};
    int num_layers{0};          // layers physically owned by this arena
    int layer_offset{0};        // global index of layers[0]
    int global_num_layers{0};   // full logical body depth
    int vocab_size{0};
    int max_position_embeddings{0};
    int heads{0};
    int kv_heads{0};
    std::string architecture_contract{"ida_lattice_native_v1"};
    bool owns_embedding{true};
    bool owns_output{true};     // final norm, LM head, LRSS/LSS/PSS
    std::size_t total_bytes{0};
};

// ── Current resident AdamW optimizer state (device-resident; storage precision
// follows request.optimizer_state_precision) ──────────────────────────────────

struct OptStateTensor {
    float* f32{nullptr};
    __nv_bfloat16* bf16{nullptr};
    // Non-null only when IDA_NATIVE_OPTIM_STATE_HOST_OFFLOAD is active: f32/
    // bf16 above then hold the device-mapped ALIAS of this host-pinned
    // allocation (via cudaHostGetDevicePointer), not a normal device
    // pointer. Freeing must use this original host pointer with
    // cudaFreeHost, never cudaFreeAsync on the device alias.
    void* host_backing{nullptr};
};

struct LatticeOptLayer {
    OptStateTensor m_q{};     OptStateTensor v_q{};
    OptStateTensor m_k{};     OptStateTensor v_k{};
    OptStateTensor m_v{};     OptStateTensor v_v{};
    OptStateTensor m_o{};     OptStateTensor v_o{};
    // Qwen2-family QKV bias (2026-08-23), mirrors LatticeLayerWeights'
    // q_bias/k_bias/v_bias (allocated only when qkv_bias is set).
    OptStateTensor m_q_bias{}; OptStateTensor v_q_bias{};
    OptStateTensor m_k_bias{}; OptStateTensor v_k_bias{};
    OptStateTensor m_v_bias{}; OptStateTensor v_v_bias{};
    OptStateTensor m_o_bias{}; OptStateTensor v_o_bias{};
    OptStateTensor m_attn_norm_bias{}; OptStateTensor v_attn_norm_bias{};
    OptStateTensor m_ffn_norm_bias{}; OptStateTensor v_ffn_norm_bias{};
    OptStateTensor m_ffn_in_bias{}; OptStateTensor v_ffn_in_bias{};
    OptStateTensor m_ffn_out_bias{}; OptStateTensor v_ffn_out_bias{};
    OptStateTensor m_gate{};  OptStateTensor v_gate{};
    OptStateTensor m_up{};    OptStateTensor v_up{};
    OptStateTensor m_down{};  OptStateTensor v_down{};
    OptStateTensor m_anorm{}; OptStateTensor v_anorm{};
    OptStateTensor m_fnorm{}; OptStateTensor v_fnorm{};
    // Cognitive-architecture sparse MoE port (2026-07-21), mirrors
    // LatticeLayerWeights above 1:1 (allocated only when num_experts>0).
    OptStateTensor m_moe_trunk_in{};   OptStateTensor v_moe_trunk_in{};
    OptStateTensor m_moe_trunk_out{};  OptStateTensor v_moe_trunk_out{};
    OptStateTensor m_moe_expert_in{};  OptStateTensor v_moe_expert_in{};
    OptStateTensor m_moe_expert_out{}; OptStateTensor v_moe_expert_out{};
    OptStateTensor m_pressure_proj{};  OptStateTensor v_pressure_proj{};
    OptStateTensor m_pressure_mod{};   OptStateTensor v_pressure_mod{};
    OptStateTensor m_router_score{};   OptStateTensor v_router_score{};
    OptStateTensor m_pressure_to_routes{}; OptStateTensor v_pressure_to_routes{};
    // Generic per-token top-k SwiGLU MoE (2026-08-23), mirrors
    // LatticeLayerWeights' generic_* fields 1:1 (allocated only when
    // moe_kind==1). router_score shares m_router_score/v_router_score
    // above -- both modes' router_score_w use the same optimizer slot.
    OptStateTensor m_generic_gate{};  OptStateTensor v_generic_gate{};
    OptStateTensor m_generic_up{};    OptStateTensor v_generic_up{};
    OptStateTensor m_generic_down{};  OptStateTensor v_generic_down{};
    // Shared expert (2026-08-23), mirrors LatticeLayerWeights' generic_shared_*
    // fields 1:1 (allocated only when shared_expert_width>0).
    OptStateTensor m_generic_shared_gate{};  OptStateTensor v_generic_shared_gate{};
    OptStateTensor m_generic_shared_up{};    OptStateTensor v_generic_shared_up{};
    OptStateTensor m_generic_shared_down{};  OptStateTensor v_generic_shared_down{};
    OptStateTensor m_generic_shared_gate_score{}; OptStateTensor v_generic_shared_gate_score{};
};

struct LatticeOptState {
    OptStateTensor m_embed{};
    OptStateTensor v_embed{};
    OptStateTensor m_position_embeddings{}; OptStateTensor v_position_embeddings{};
    LatticeOptLayer* layers{nullptr};
    OptStateTensor m_fnorm{};
    OptStateTensor v_fnorm{};
    OptStateTensor m_fnorm_bias{}; OptStateTensor v_fnorm_bias{};
    OptStateTensor m_lm_head{};
    OptStateTensor v_lm_head{};
    // LRSS (allocated only when weights.lrss_enabled)
    OptStateTensor m_lrss_q{};   OptStateTensor v_lrss_q{};
    OptStateTensor m_lrss_k{};   OptStateTensor v_lrss_k{};
    OptStateTensor m_lrss_gw{};  OptStateTensor v_lrss_gw{};
    OptStateTensor m_lrss_gb{};  OptStateTensor v_lrss_gb{};
    OptStateTensor m_lrss_lt{};  OptStateTensor v_lrss_lt{};
    OptStateTensor m_lrss_sw{};  OptStateTensor v_lrss_sw{};
    OptStateTensor m_lss_dn{};   OptStateTensor v_lss_dn{};
    OptStateTensor m_lss_up{};   OptStateTensor v_lss_up{};
    // PSS Stage 2 (allocated only when weights.pss_pred_rank > 0)
    OptStateTensor m_pss_pred_dn{}; OptStateTensor v_pss_pred_dn{};
    OptStateTensor m_pss_pred_up{}; OptStateTensor v_pss_pred_up{};
    bool optimizer_state_bf16{false};
    int adam_step{0};
};

// ── API ──────────────────────────────────────────────────────────────────────

LatticeWeights  allocate_lattice_weights(const NativeRequest& req, NativeArena& arena);
LatticeOptState allocate_lattice_opt(const NativeRequest& req, const LatticeWeights& w, NativeArena& arena);
void            free_lattice_weights(LatticeWeights& w, NativeArena& arena);
void            free_lattice_opt(LatticeOptState& opt, const LatticeWeights& w, NativeArena& arena);

BurnResult run_lattice_training(
    const NativeRequest& request,
    NativeArena& arena,
    ProgressCallback on_step = {}
);

// Two-stage, single-process model parallelism using CUDA peer copies only.
// This intentionally does not use NCCL or replicated data parallelism.
BurnResult run_lattice_training_model_parallel(
    const NativeRequest& request,
    NativeArena& first_stage,
    NativeArena& second_stage,
    ProgressCallback on_step = {}
);

// Alias: main.cpp still calls run_smoke_training — forward to the real impl.
inline BurnResult run_smoke_training(
    const NativeRequest& request,
    NativeArena& arena,
    ProgressCallback on_step = {}
) {
    return run_lattice_training(request, arena, std::move(on_step));
}

}  // namespace ida_native
