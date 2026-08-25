#pragma once

#include <filesystem>
#include <cstdint>
#include <string>
#include <vector>

#ifndef IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
#define IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY 0
#endif

namespace ida_native {

struct DeviceRequest {
    // "cuda" keeps the existing native trainer contract. "opencl" selects
    // the separate AMD portability smoke backend; it never falls through to
    // CUDA when the requested runtime is unavailable.
    std::string runtime{"cuda"};
    std::string required_arch{"sm_90"};
    std::string precision{"bf16"};
    // Boundary transport for a two-device native pipeline. "auto" prefers
    // CUDA peer access and falls back to host staging; "cuda_p2p" requires
    // direct peer access; "host_staged" is an explicit NVLink/P2P-free mode.
    std::string peer_transport{"auto"};
    // Empty keeps the established single-device execution contract. Exactly
    // two distinct device ordinals request one logical body split into two
    // native pipeline stages; this is CUDA peer transport, never NCCL or
    // replicated data parallelism.
    std::vector<int> model_parallel_devices{};
    // Global layer index where stage 0 hands the residual stream to stage 1.
    // Zero selects the balanced split during native admission.
    int pipeline_split_layer{0};
};

struct ModelRequest {
    std::string architecture_contract{"ida_lattice_native_v1"};
    int hidden_size{0};
    int intermediate_size{0};
    int layers{0};
    int heads{0};
    // Number of physical K/V heads. Defaults to heads for legacy MHA
    // requests; reduced K/V heads are expanded at scalar attention time.
    int kv_heads{0};
    int vocab_size{0};
    // Cognitive-architecture sparse MoE port (2026-07-21). 0 = off, byte-
    // identical to the pre-port dense gate/up/down FFN -- see
    // docs/lrss-native-port-contract.md-style precedent (LRSS/PSS-predictor)
    // for the "optional block gated by a size field" convention this follows.
    int num_cognitive_routes{0};
    int top_k_routes{0};
    int num_personality_experts{0};
    int personality_residual_expert_width{0};
    // Switch-form load-balancing aux loss strength. Present in every MoE
    // config since inception but never parsed or implemented until
    // 2026-07-30; nothing opposed routing concentration, and entropy
    // collapsed 0.44 -> 0.14 with 2 of 11 experts surviving.
    float expert_balancing_loss_coef{0.0f};
    int top_k_experts{0};
    bool use_personality_residual_experts{false};
    // Sliding-window attention size. 0 = unset -> fall back to the existing
    // head_dim-derived heuristic in attn_window_tokens(Hd) (attention.cu).
    // Added because that heuristic silently diverges from the real
    // atlas_moe config's local_attention_window=256 (Hd=512 -> heuristic
    // picks 128) -- this field lets the real config value actually reach
    // the engine instead of being silently dropped.
    int local_attention_window{0};
    // Native packed-FP4 activation path for the MoE expert bank (real
    // storage/bandwidth reduction layered on the existing FP8 GEMM path --
    // see moe_native_fp4_enabled()'s doc comment in trainer.cu). Requires
    // an FP8 precision profile active; a no-op otherwise. Default false --
    // not yet a validated production default, still under real-burn
    // comparison against plain FP8 as of 2026-07-22.
    bool moe_native_fp4{false};
    // Generic per-token top-k SwiGLU MoE for external model families such as
    // Mixtral and Qwen-MoE. This is separate from the native cognitive-route
    // expert bank; the two modes are mutually exclusive.
    int generic_moe_num_experts{0};
    int generic_moe_top_k{0};
    int generic_moe_expert_width{0};
    int generic_moe_shared_expert_width{0};
    bool generic_moe_normalize_topk{true};
    // Qwen-family attention projection bias for q/k/v. Native Lattice
    // requests remain bias-free when this is false.
    bool qkv_bias{false};
    // Rotary position embedding base (HF's "rope_theta", e.g. 10000 or
    // 130000). 0 = off -- IDA's own Lattice architecture has no positional
    // encoding at all (see the sideload evaluation: no absolute/learned
    // position embedding either), so this must default closed to leave
    // every existing production body byte-identical. Standard
    // Llama/Qwen/Phi-class architectures being sideloaded (2026-08-13
    // portability port) require this to be set to their published
    // rope_theta. Same "size/flag field gates an optional block" pattern
    // as num_cognitive_routes/moe_native_fp4 above.
    float rope_theta{0.0f};
    std::string normalization_type{"rmsnorm"};
    std::string activation_type{"swiglu"};
    std::string position_embedding_type{"none"};
    float norm_eps{1.0e-6f};
    int max_position_embeddings{0};
    bool projection_bias{false};
    bool tied_embeddings{false};
};

struct TrainingRequest {
    int microbatch{0};
    int grad_accumulation{0};
    double learning_rate{0.0};
    int max_steps{1};
};

struct InputRequest {
    std::filesystem::path token_blocks;
    std::filesystem::path label_blocks;
    std::filesystem::path seg_blocks;   // u16 per-position sample-start offsets (optional)
    std::filesystem::path native_input_dir;
    std::filesystem::path native_input_manifest;
    int batch_size{0};
    int sequence_length{0};
    int num_sequences{0};
};

struct ParentRequest {
    std::string backend;
    std::filesystem::path manifest;
    std::filesystem::path init_from_model;
    std::filesystem::path resume_from_checkpoint;
};

struct NativeRequest {
    std::filesystem::path request_json_path;
    std::filesystem::path config_path;
    std::filesystem::path dataset_path;
    std::filesystem::path output_dir;
    std::filesystem::path repo_root;
    std::filesystem::path status_file;
    std::filesystem::path tokenizer_path;
    std::string backend;
    std::string attention_backend{"scalar_flash"};
    std::string precision_profile{"legacy_fp8"};
    // Per-request hardware/precision identity. These are deliberately not
    // read from process environment by the native worker: a shared worker
    // must not inherit a prior body's 3050 memory bucket or FP8 policy.
    std::string hardware_profile{};
    std::string memory_profile{};
    std::string fp8_storage_format{};
    std::string fp8_compute_path{};
    std::string fp8_weight_scale_mode{};
    // NVFP4 is a Blackwell-only runtime cache.  Durable checkpoints remain
    // BF16 and the field is empty for every non-NVFP4 request.
    std::string nvfp4_storage_format{};
    std::string nvfp4_compute_path{};
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
    // Private observability is a per-burn admission contract. It is compiled
    // out of the public native request ABI.
    bool ontology_required{false};
    bool analytics_required{false};
    std::filesystem::path ontology_path{};
    std::filesystem::path analytics_path{};
    std::string analytics_contract{};
#endif
    // -1 = preserve the legacy env/default arena reservation; 0 = explicitly
    // disable pre-reservation for a small consumer-GPU body.
    std::int64_t arena_reserve_bytes{-1};
    // Per-request free-VRAM admission margin. -1 selects the conservative
    // native default; the launcher records the measured working-set margin.
    std::int64_t min_free_vram_bytes{-1};
    std::string optimizer_state_precision{"fp32"};
    std::string optimizer_type{"lion"};   // Adam/AdamW disabled; Lion is the only enabled optimizer
    std::string gradient_buffer_precision{"fp32"};
    std::string gemm_accumulator_precision{"fp32"};
    std::string family;
    std::string job_id;
    std::string seat;
    std::string version;
    std::string engine_revision;
    std::string architecture_compatibility;
    std::string architecture_contract;
    std::string expected_terminal_phase;
    DeviceRequest device;
    ModelRequest model;
    TrainingRequest training;
    InputRequest input;
    ParentRequest parent;
    int batch_size_override{0};
    int grad_accum_override{0};
    int native_max_steps{1};
    int max_samples{0};
    int seed{1337};
    bool promotion_enabled{false};
    // PSS Stage 2 predictor, per-burn (2026-07-15). The persistent worker
    // worker reads env vars from ITS OWN process environment, not the
    // launching wrapper's -- so env-based IDA_NATIVE_PSS_PRED activation
    // silently never reached worker-served burns (only direct launches).
    // Request fields carry the wrapper's family-gated decision per burn.
    // -1 = unset -> fall back to the env flags (old behavior); 0 = off;
    // >0 = predictor rank.
    int pss_pred_rank{-1};
    float pss_pred_lr_scale{-1.0f};
    // Per-burn PSS auxiliary-loss weight. This cannot rely on the worker
    // environment because one shared worker serves bodies with different PSS
    // profiles. Negative means use the direct-launch environment fallback.
    float pss_pred_aux_weight{-1.0f};
    // Per-burn PSS geometry policy. Empty = direct-launch env fallback.
    // "auto" keeps dense Edge/Swift on the old dense/raw-MSE path while
    // routing AI/MoE/wide bodies through conditioned + target-normalized PSS.
    std::string pss_conditioning_mode{};
    std::string pss_aux_normalize_mode{};
    // Standalone PSS predictor state (per-family×seat), same worker-caching
    // rationale as pss_pred_rank above: these are per-seat paths, and a
    // persistent worker shared across seats would cache whichever seat's
    // getenv() value was live at its own spawn time, silently cross-wiring
    // every other seat's predictor state to the wrong file. Empty = absent
    // -> fall back to the env vars (probe-script/direct-launch compat).
    std::string pss_pred_state_out{};
    std::string pss_pred_state_init_path{};
    // Global grad-clip ceiling, per-burn (2026-07-21, MPS-replacement
    // multi-tenancy fix). Same worker-caching problem as pss_pred_rank
    // above, but sharper: IDA_NATIVE_GLOBAL_CLIP is family-scaled
    // (edge/swift ~7, ai/moe ~22) and, unlike attn_window, has no
    // architecture-derived default to fall back on -- it's an empirically
    // tuned value with no equivalent computable from head_dim. A shared
    // multi-tenant process's fixed spawn-time env would apply whichever
    // family's clip was live at spawn to every OTHER body sharing that
    // process, silently wrong for one of them. -1 = unset -> fall back to
    // the env var / 1.0 default (direct-launch / pre-existing behavior).
    float global_clip_override{-1.0f};
    // Activation-row clip ceiling, per-burn (2026-08-09, same worker-caching
    // fix as global_clip_override above -- act_row_clip_mult() was a
    // function-local static getenv() cache in trainer.cu, so a shared
    // multi-tenant worker applied whichever family's IDA_NATIVE_ACT_ROW_CLIP
    // was live at its OWN first call to it, for the rest of that worker's
    // life, to every body sharing it regardless of family. Found while
    // wiring Swift's validated ACT_ROW_CLIP=0 win (never fires at Swift's
    // scale, +9.49% throughput with it off) to production. -1 = unset ->
    // fall back to the env var / 1000.0f default (direct-launch compat);
    // 0 = explicitly disabled; positive = explicit ceiling.
    float act_row_clip_override{-1.0f};
    // Canonical BurnSpec identity hash (Phase 1, 2026-07-22). Computed
    // Python-side only (src/ida_train/training/burn_spec.py) -- the engine
    // never recomputes it, only carries and string-compares it, since
    // reproducing Python's exact json.dumps(sort_keys=True) byte-for-byte
    // in C++ would itself be a two-implementations-drift risk. Empty =
    // pre-BurnSpec request (fail-open, not a validation error).
    std::string spec_hash{};
    // Lion optimizer policy, per-burn (Phase 2, 2026-07-22). Same worker-
    // caching problem as pss_pred_rank/global_clip_override above: these
    // seven knobs were previously getenv()-cached ONCE per worker-process
    // lifetime in trainer.cu (lion_lr_scale/wd_scale/beta1/beta2,
    // trust_ratio_enabled/lo/hi), so a persistent worker serving multiple
    // seats/families ran whichever Lion policy was live in its own
    // environment at spawn time regardless of what any later burn's
    // request actually asked for -- launch_family_queue.sh's own comment
    // (~1049-1058) already documented this exact gap. Sentinel semantics
    // match global_clip_override exactly: negative (or -1 for the bool-
    // shaped trust_ratio_enabled) means "unset, defer to the cached env
    // value"; any other value wins.
    float lion_lr_scale_override{-1.0f};
    float lion_wd_scale_override{-1.0f};
    float lion_beta1_override{-1.0f};
    float lion_beta2_override{-1.0f};
    int   lion_trust_ratio_enabled_override{-1};  // -1 unset, 0 off, 1 on
    float lion_trust_ratio_lo_override{-1.0f};
    float lion_trust_ratio_hi_override{-1.0f};
};

NativeRequest load_request(const std::filesystem::path& path);

}  // namespace ida_native
