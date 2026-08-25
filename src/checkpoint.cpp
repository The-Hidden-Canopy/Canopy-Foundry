#include "ida_native/checkpoint.hpp"

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "ida_native/kernels.hpp"   // attn_window_tokens (recipe-surface contract)

namespace ida_native {

namespace {

// -- Additive-architecture load policy ---------------------------------------
// Adding a parameter group to the architecture normally forces a re-genesis:
// enumerate_lattice_tensors() emits the new tensors, a parent trained before
// they existed has none of them, and load fails closed on "missing tensor".
// That is why every architecture change so far has cost a fresh lineage.
//
// A prefix declared here is instead allowed to be ABSENT from the parent: this
// burn keeps its own fresh initialization for those tensors and loads all the
// rest, which makes an additive change adoptable MID-LINEAGE.
//
// SOUNDNESS REQUIREMENT: the change must be identity-at-init -- with the new
// group at its initial values the model must compute the SAME function it did
// before the group existed (e.g. a chunk router whose uniform logits reproduce
// the previous dense sum). If it is not identity-at-init, the resume still
// "succeeds" but silently continues the body as a different function than the
// parent was trained as, which is worse than failing closed.
//
// Deliberately code-declared rather than env-driven: an env override could mask
// a genuinely corrupt or mismatched checkpoint, and this codebase has been
// bitten repeatedly by silent load-time fallbacks. Any prefix not listed here
// stays fatal.
constexpr const char* kOptionalTensorPrefixes[] = {
    "multiscale_memory.pss_predictor.",  // PSS Stage 2 (parent rank 0 -> N)
};

bool tensor_optional_at_load(const std::string& name) {
    for (const char* prefix : kOptionalTensorPrefixes) {
        if (name.rfind(prefix, 0) == 0) return true;
    }
    return false;
}

void write_text(const std::filesystem::path& path, const std::string& contents) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << contents;
}

std::string checkpoint_dir_name(int step) {
    std::ostringstream output;
    output << "checkpoint-" << std::setw(8) << std::setfill('0') << step;
    return output.str();
}

// ── safetensors tensor enumeration ───────────────────────────────────────────

struct TensorEntry {
    std::string name;
    const __nv_bfloat16* ptr;
    std::vector<std::int64_t> shape;
    std::size_t numel;
};

std::vector<TensorEntry> enumerate_lattice_tensors(const LatticeWeights& w) {
    const auto H = static_cast<std::int64_t>(w.hidden_size);
    const auto I = static_cast<std::int64_t>(w.intermediate_size);
    const auto V = static_cast<std::int64_t>(w.vocab_size);
    const auto KV = static_cast<std::int64_t>(w.kv_heads > 0 ? w.kv_heads : w.heads) *
        (H / static_cast<std::int64_t>(w.heads));

    std::vector<TensorEntry> out;
    out.reserve(3 + static_cast<std::size_t>(w.num_layers) * 9);
    auto add = [&](std::string name, const __nv_bfloat16* p,
                   std::vector<std::int64_t> shape) {
        std::size_t n = 1;
        for (const auto d : shape) n *= static_cast<std::size_t>(d);
        out.push_back({std::move(name), p, std::move(shape), n});
    };

    if (w.owns_embedding) {
        add("embed_tokens.weight", w.embed, {V, H});
        if (w.position_embeddings) add("position_embeddings.weight", w.position_embeddings, {w.max_position_embeddings, H});
    }
    const bool gpt2 = w.architecture_contract == "hf_gpt2_native_v1";
    for (int l = 0; l < w.num_layers; ++l) {
        const std::string base = "layers." + std::to_string(w.layer_offset + l) + ".";
        const auto& lw = w.layers[l];
        add(base + "attn_norm.weight", lw.attn_norm, {H});
        if (lw.attn_norm_bias) add(base + "attn_norm.bias", lw.attn_norm_bias, {H});
        add(base + "q_proj.weight",    lw.q_proj,    {H, H});
        add(base + "k_proj.weight",    lw.k_proj,    {H, KV});
        add(base + "v_proj.weight",    lw.v_proj,    {H, KV});
        add(base + "o_proj.weight",    lw.o_proj,    {H, H});
        // Qwen2-family QKV bias (2026-08-23). No o_proj bias (Qwen2Attention
        // doesn't have one either) -- absent from every checkpoint written
        // before this feature existed.
        if (lw.q_bias != nullptr) {
            add(base + "q_proj.bias", lw.q_bias, {H});
            add(base + "k_proj.bias", lw.k_bias, {KV});
            add(base + "v_proj.bias", lw.v_bias, {KV});
            if (lw.o_bias) add(base + "o_proj.bias", lw.o_bias, {H});
        }
        add(base + "ffn_norm.weight",  lw.ffn_norm,  {H});
        if (lw.ffn_norm_bias) add(base + "ffn_norm.bias", lw.ffn_norm_bias, {H});
        if (gpt2) {
            add(base + "ffn_in.weight",  lw.gate_proj, {H, I});
            add(base + "ffn_in.bias",    lw.ffn_in_bias, {I});
            add(base + "ffn_out.weight", lw.down_proj, {I, H});
            add(base + "ffn_out.bias",   lw.ffn_out_bias, {H});
        } else if (lw.moe_kind == 0) {
            add(base + "gate_proj.weight", lw.gate_proj, {H, I});
            add(base + "up_proj.weight",   lw.up_proj,   {H, I});
            add(base + "down_proj.weight", lw.down_proj, {I, H});
        }
        // Cognitive-architecture sparse MoE port (2026-07-21). Native-
        // internal tensor names -- NOT yet decided against the HF
        // state_dict convention the promotion/eval pipeline may need
        // (open question, see the port plan); expert_fc_in/out cover
        // every expert in one contiguous tensor rather than per-expert
        // names, since that's how they're allocated (one ParamSlot per
        // weight TYPE, not per expert).
        if (lw.num_experts > 0 && lw.moe_kind == 0) {
            const auto N = static_cast<std::int64_t>(lw.num_experts);
            const auto Ie = static_cast<std::int64_t>(lw.expert_intermediate_size);
            const auto R = static_cast<std::int64_t>(lw.num_routes);
            if (lw.moe_shared_trunk) {
                add(base + "moe.trunk_fc_in.weight",  lw.trunk_fc_in_w,  {I, H});
                add(base + "moe.trunk_fc_out.weight", lw.trunk_fc_out_w, {H, I});
            }
            add(base + "moe.expert_fc_in.weight",  lw.expert_fc_in_w,  {N, Ie, H});
            add(base + "moe.expert_fc_out.weight", lw.expert_fc_out_w, {N, H, Ie});
            add(base + "moe.pressure_proj.weight", lw.pressure_proj_w, {R, H});
            add(base + "moe.pressure_mod.weight",  lw.pressure_mod_w,  {H, R});
            add(base + "moe.router_score.weight",  lw.router_score_w,  {N, H});
            if (lw.pressure_to_routes_w != nullptr) {
                add(base + "moe.pressure_to_routes.weight", lw.pressure_to_routes_w, {N, R});
            }
        } else if (lw.num_experts > 0 && lw.moe_kind == 1) {
            // Generic per-token top-k SwiGLU MoE (2026-08-23). Same
            // "one ParamSlot per weight type, not per expert" convention
            // as the legacy block above.
            const auto N = static_cast<std::int64_t>(lw.num_experts);
            const auto Ie = static_cast<std::int64_t>(lw.expert_intermediate_size);
            add(base + "moe.generic_gate_proj.weight", lw.generic_gate_proj_w, {N, Ie, H});
            add(base + "moe.generic_up_proj.weight",   lw.generic_up_proj_w,   {N, Ie, H});
            add(base + "moe.generic_down_proj.weight", lw.generic_down_proj_w, {N, H, Ie});
            add(base + "moe.router_score.weight",      lw.router_score_w,      {N, H});
            // Shared expert (2026-08-23; Qwen2-MoE-family). Own tensor names,
            // gated on shared_expert_width>0 -- absent from every checkpoint
            // written before this feature existed.
            if (lw.shared_expert_width > 0) {
                const auto Ws = static_cast<std::int64_t>(lw.shared_expert_width);
                add(base + "moe.generic_shared_gate_proj.weight", lw.generic_shared_gate_proj_w, {Ws, H});
                add(base + "moe.generic_shared_up_proj.weight",   lw.generic_shared_up_proj_w,   {Ws, H});
                add(base + "moe.generic_shared_down_proj.weight", lw.generic_shared_down_proj_w, {H, Ws});
                add(base + "moe.generic_shared_gate_score.weight", lw.generic_shared_gate_score_w, {1, H});
            }
        }
    }
    if (w.owns_output) {
        add("final_norm.weight", w.final_norm, {H});
        if (w.final_norm_bias) add("final_norm.bias", w.final_norm_bias, {H});
        add("lm_head.weight",    w.lm_head,    {V, H});
    }
    if (w.owns_output && w.lrss_enabled) {
        const auto J = static_cast<std::int64_t>(w.lrss_scales);
        add("multiscale_memory.query_proj.weight",  w.lrss_query,  {H, H});
        add("multiscale_memory.key_proj.weight",    w.lrss_key,    {H, H});
        add("multiscale_memory.output_gate.weight", w.lrss_gate_w, {H, 2 * H});
        add("multiscale_memory.output_gate.bias",   w.lrss_gate_b, {H});
        add("multiscale_memory.log_tau",            w.lrss_log_tau, {J});
        add("multiscale_memory.scale_weights",      w.lrss_scale_w, {J});
        if (w.lss_rank > 0) {
            const auto R = static_cast<std::int64_t>(w.lss_rank);
            // PSS Stage 1b: joint_dim is 2*H unless pss_spike_joint_dim > 0
            // (IDA_NATIVE_PSS_SPIKE_JOINT=1), in which case down.weight is
            // wider by that many columns. 0 = byte-identical to the
            // pre-PSS shape.
            const auto joint_dim = 2 * H + static_cast<std::int64_t>(w.pss_spike_joint_dim);
            add("multiscale_memory.supersampler.down.weight", w.lss_down, {R, joint_dim});
            add("multiscale_memory.supersampler.up.weight",   w.lss_up,   {H, R});
        }
    }
    if (w.owns_output && w.pss_pred_rank > 0) {
        const auto R = static_cast<std::int64_t>(w.pss_pred_rank);
        add("multiscale_memory.pss_predictor.down.weight", w.pss_pred_down, {H, R});
        add("multiscale_memory.pss_predictor.up.weight",   w.pss_pred_up,   {R, H});
    }
    return out;
}

// ── Optimizer-state tensor enumeration (Phase 3, 2026-07-22) ────────────────
// Mirrors enumerate_lattice_tensors' shape/gating logic exactly (same H/I/V/
// per-layer/LRSS/LSS/PSS-predictor/MoE conditionals) but over LatticeOptState's
// m/v pairs instead of LatticeWeights' single tensor per slot. dtype is
// per-run (opt.optimizer_state_bf16), not hardcoded BF16 like the weight
// enumerator. A null ptr (v absent under Lion, or a gated-off feature) is
// silently skipped -- save and load both call this, so what's absent on save
// is exactly what's not expected on load.
struct OptTensorEntry {
    std::string name;
    const void* ptr;
    bool is_bf16;
    std::vector<std::int64_t> shape;
    std::size_t numel;
};

std::vector<OptTensorEntry> enumerate_lattice_opt_tensors(
    const LatticeWeights& w, const LatticeOptState& opt
) {
    const auto H = static_cast<std::int64_t>(w.hidden_size);
    const auto I = static_cast<std::int64_t>(w.intermediate_size);
    const auto V = static_cast<std::int64_t>(w.vocab_size);
    const auto KV = static_cast<std::int64_t>(w.kv_heads > 0 ? w.kv_heads : w.heads) *
        (H / static_cast<std::int64_t>(w.heads));
    const bool bf16 = opt.optimizer_state_bf16;
    const bool gpt2 = w.architecture_contract == "hf_gpt2_native_v1";

    std::vector<OptTensorEntry> out;
    auto add = [&](const std::string& name, const OptStateTensor& t,
                   std::vector<std::int64_t> shape) {
        const void* ptr = bf16 ? static_cast<const void*>(t.bf16)
                               : static_cast<const void*>(t.f32);
        if (ptr == nullptr) return;  // Lion's v, or a gated-off feature
        std::size_t n = 1;
        for (const auto d : shape) n *= static_cast<std::size_t>(d);
        out.push_back({name, ptr, bf16, std::move(shape), n});
    };

    if (w.owns_embedding) {
        add("embed.m", opt.m_embed, {V, H});
        add("embed.v", opt.v_embed, {V, H});
        if (w.position_embeddings) {
            add("position_embeddings.m", opt.m_position_embeddings, {w.max_position_embeddings, H});
            add("position_embeddings.v", opt.v_position_embeddings, {w.max_position_embeddings, H});
        }
    }
    for (int l = 0; l < w.num_layers; ++l) {
        const std::string base = "layers." + std::to_string(w.layer_offset + l) + ".";
        const auto& lo = opt.layers[l];
        const auto& lw = w.layers[l];
        add(base + "q.m", lo.m_q, {H, H});       add(base + "q.v", lo.v_q, {H, H});
        add(base + "k.m", lo.m_k, {H, KV});       add(base + "k.v", lo.v_k, {H, KV});
        add(base + "v.m", lo.m_v, {H, KV});       add(base + "v.v", lo.v_v, {H, KV});
        add(base + "o.m", lo.m_o, {H, H});       add(base + "o.v", lo.v_o, {H, H});
        if (lw.q_bias != nullptr) {
            add(base + "q_bias.m", lo.m_q_bias, {H});  add(base + "q_bias.v", lo.v_q_bias, {H});
            add(base + "k_bias.m", lo.m_k_bias, {KV}); add(base + "k_bias.v", lo.v_k_bias, {KV});
            add(base + "v_bias.m", lo.m_v_bias, {KV}); add(base + "v_bias.v", lo.v_v_bias, {KV});
            if (lw.o_bias) {
                add(base + "o_bias.m", lo.m_o_bias, {H}); add(base + "o_bias.v", lo.v_o_bias, {H});
        }
        }
        if (gpt2) {
            add(base + "ffn_in.m", lo.m_gate, {H, I}); add(base + "ffn_in.v", lo.v_gate, {H, I});
            add(base + "ffn_in_bias.m", lo.m_ffn_in_bias, {I}); add(base + "ffn_in_bias.v", lo.v_ffn_in_bias, {I});
            add(base + "ffn_out.m", lo.m_down, {I, H}); add(base + "ffn_out.v", lo.v_down, {I, H});
            add(base + "ffn_out_bias.m", lo.m_ffn_out_bias, {H}); add(base + "ffn_out_bias.v", lo.v_ffn_out_bias, {H});
        } else if (lw.moe_kind == 0) {
            add(base + "gate.m", lo.m_gate, {H, I}); add(base + "gate.v", lo.v_gate, {H, I});
            add(base + "up.m", lo.m_up, {H, I});     add(base + "up.v", lo.v_up, {H, I});
            add(base + "down.m", lo.m_down, {I, H}); add(base + "down.v", lo.v_down, {I, H});
        }
        add(base + "anorm.m", lo.m_anorm, {H});  add(base + "anorm.v", lo.v_anorm, {H});
        if (lw.attn_norm_bias) { add(base + "anorm_bias.m", lo.m_attn_norm_bias, {H}); add(base + "anorm_bias.v", lo.v_attn_norm_bias, {H}); }
        add(base + "fnorm.m", lo.m_fnorm, {H});  add(base + "fnorm.v", lo.v_fnorm, {H});
        if (lw.ffn_norm_bias) { add(base + "fnorm_bias.m", lo.m_ffn_norm_bias, {H}); add(base + "fnorm_bias.v", lo.v_ffn_norm_bias, {H}); }
        if (lw.num_experts > 0 && lw.moe_kind == 0) {
            const auto N  = static_cast<std::int64_t>(lw.num_experts);
            const auto Ie = static_cast<std::int64_t>(lw.expert_intermediate_size);
            const auto R  = static_cast<std::int64_t>(lw.num_routes);
            if (lw.moe_shared_trunk) {
                add(base + "moe_trunk_in.m",  lo.m_moe_trunk_in,  {I, H});
                add(base + "moe_trunk_in.v",  lo.v_moe_trunk_in,  {I, H});
                add(base + "moe_trunk_out.m", lo.m_moe_trunk_out, {H, I});
                add(base + "moe_trunk_out.v", lo.v_moe_trunk_out, {H, I});
            }
            add(base + "moe_expert_in.m",  lo.m_moe_expert_in,  {N, Ie, H});
            add(base + "moe_expert_in.v",  lo.v_moe_expert_in,  {N, Ie, H});
            add(base + "moe_expert_out.m", lo.m_moe_expert_out, {N, H, Ie});
            add(base + "moe_expert_out.v", lo.v_moe_expert_out, {N, H, Ie});
            add(base + "pressure_proj.m", lo.m_pressure_proj, {R, H});
            add(base + "pressure_proj.v", lo.v_pressure_proj, {R, H});
            add(base + "pressure_mod.m",  lo.m_pressure_mod,  {H, R});
            add(base + "pressure_mod.v",  lo.v_pressure_mod,  {H, R});
            add(base + "router_score.m",  lo.m_router_score,  {N, H});
            add(base + "router_score.v",  lo.v_router_score,  {N, H});
            if (lw.pressure_to_routes_w != nullptr) {
                add(base + "pressure_to_routes.m", lo.m_pressure_to_routes, {N, R});
                add(base + "pressure_to_routes.v", lo.v_pressure_to_routes, {N, R});
            }
        } else if (lw.num_experts > 0 && lw.moe_kind == 1) {
            const auto N  = static_cast<std::int64_t>(lw.num_experts);
            const auto Ie = static_cast<std::int64_t>(lw.expert_intermediate_size);
            add(base + "generic_gate.m", lo.m_generic_gate, {N, Ie, H});
            add(base + "generic_gate.v", lo.v_generic_gate, {N, Ie, H});
            add(base + "generic_up.m",   lo.m_generic_up,   {N, Ie, H});
            add(base + "generic_up.v",   lo.v_generic_up,   {N, Ie, H});
            add(base + "generic_down.m", lo.m_generic_down, {N, H, Ie});
            add(base + "generic_down.v", lo.v_generic_down, {N, H, Ie});
            add(base + "router_score.m", lo.m_router_score, {N, H});
            add(base + "router_score.v", lo.v_router_score, {N, H});
            if (lw.shared_expert_width > 0) {
                const auto Ws = static_cast<std::int64_t>(lw.shared_expert_width);
                add(base + "generic_shared_gate.m", lo.m_generic_shared_gate, {Ws, H});
                add(base + "generic_shared_gate.v", lo.v_generic_shared_gate, {Ws, H});
                add(base + "generic_shared_up.m",   lo.m_generic_shared_up,   {Ws, H});
                add(base + "generic_shared_up.v",   lo.v_generic_shared_up,   {Ws, H});
                add(base + "generic_shared_down.m", lo.m_generic_shared_down, {H, Ws});
                add(base + "generic_shared_down.v", lo.v_generic_shared_down, {H, Ws});
                add(base + "generic_shared_gate_score.m", lo.m_generic_shared_gate_score, {1, H});
                add(base + "generic_shared_gate_score.v", lo.v_generic_shared_gate_score, {1, H});
            }
        }
    }
    if (w.owns_output) {
        add("fnorm.m",   opt.m_fnorm,   {H});    add("fnorm.v",   opt.v_fnorm,   {H});
        if (w.final_norm_bias) { add("fnorm_bias.m", opt.m_fnorm_bias, {H}); add("fnorm_bias.v", opt.v_fnorm_bias, {H}); }
        add("lm_head.m", opt.m_lm_head, {V, H}); add("lm_head.v", opt.v_lm_head, {V, H});
    }
    if (w.owns_output && w.lrss_enabled) {
        const auto J = static_cast<std::int64_t>(w.lrss_scales);
        add("lrss_q.m",  opt.m_lrss_q,  {H, H});     add("lrss_q.v",  opt.v_lrss_q,  {H, H});
        add("lrss_k.m",  opt.m_lrss_k,  {H, H});     add("lrss_k.v",  opt.v_lrss_k,  {H, H});
        add("lrss_gw.m", opt.m_lrss_gw, {H, 2 * H}); add("lrss_gw.v", opt.v_lrss_gw, {H, 2 * H});
        add("lrss_gb.m", opt.m_lrss_gb, {H});        add("lrss_gb.v", opt.v_lrss_gb, {H});
        add("lrss_lt.m", opt.m_lrss_lt, {J});        add("lrss_lt.v", opt.v_lrss_lt, {J});
        add("lrss_sw.m", opt.m_lrss_sw, {J});        add("lrss_sw.v", opt.v_lrss_sw, {J});
        if (w.lss_rank > 0) {
            const auto R = static_cast<std::int64_t>(w.lss_rank);
            const auto joint_dim = 2 * H + static_cast<std::int64_t>(w.pss_spike_joint_dim);
            add("lss_dn.m", opt.m_lss_dn, {R, joint_dim}); add("lss_dn.v", opt.v_lss_dn, {R, joint_dim});
            add("lss_up.m", opt.m_lss_up, {H, R});         add("lss_up.v", opt.v_lss_up, {H, R});
        }
    }
    if (w.owns_output && w.pss_pred_rank > 0) {
        const auto R = static_cast<std::int64_t>(w.pss_pred_rank);
        add("pss_pred_dn.m", opt.m_pss_pred_dn, {H, R}); add("pss_pred_dn.v", opt.v_pss_pred_dn, {H, R});
        add("pss_pred_up.m", opt.m_pss_pred_up, {R, H}); add("pss_pred_up.v", opt.v_pss_pred_up, {R, H});
    }
    return out;
}

// Header: offsets tile [0, total) in enumeration order.  Returns the padded
// header string; total_bytes gets the summed tensor payload size.
std::string build_safetensors_header(
    const NativeRequest& request,
    const std::vector<TensorEntry>& tensors,
    std::size_t& total_bytes
) {
    nlohmann::json header;
    std::size_t offset = 0;
    for (const auto& t : tensors) {
        const std::size_t bytes = t.numel * sizeof(__nv_bfloat16);
        header[t.name] = {
            {"dtype", "BF16"},
            {"shape", t.shape},
            {"data_offsets", {offset, offset + bytes}},
        };
        offset += bytes;
    }
    header["__metadata__"] = {
        {"format", request.model.architecture_contract.empty()
            ? request.architecture_contract : request.model.architecture_contract},
        {"orientation", "native_row_major"},
        {"seat", request.seat},
        {"family", request.family},
        {"version", request.version},
    };
    std::string header_str = header.dump();
    // Pad to 8-byte alignment (matches the reference python writer).
    while (header_str.size() % 8 != 0) header_str.push_back(' ');
    total_bytes = offset;
    return header_str;
}

// Writes header + payload to {output_dir}/model.safetensors.tmp, renames into
// place, and writes the lineage manifest.  `payload` must hold every tensor at
// its enumeration-order offset.  Shared by the sync and async save paths.
bool write_safetensors_file(
    const NativeRequest& request,
    const LatticeWeights& w,
    const std::string& header_str,
    const char* payload,
    std::size_t payload_bytes,
    std::size_t tensor_count,
    std::string& error,
    // Generation stamp shared with save_lattice_opt_safetensors' own
    // cumulative_opt_steps metadata (2026-07-23 checkpoint-atomicity fix):
    // model.safetensors and optimizer_state.safetensors are written via two
    // SEPARATE atomic tmp+rename operations (weights via the async saver,
    // optimizer state synchronously right after), so a kill between them
    // leaves a torn pair -- fresh weights with stale optimizer state, or
    // vice versa. True cross-file atomicity isn't available at the
    // filesystem level, so instead both files record which opt_step they
    // were written at; the resume loader hard-fails if they disagree,
    // converting a silent torn-pair corruption into a loud, catchable one.
    // -1 means "not tracked at this save point" (e.g. a save path that
    // doesn't have a meaningful step counter yet) -- skips the check.
    int cumulative_opt_steps = -1,
    int pipeline_stage_count = 1,
    int pipeline_split_layer = 0
) {
    const auto final_path = request.output_dir / "model.safetensors";
    const auto tmp_path   = request.output_dir / "model.safetensors.tmp";
    std::error_code ec;
    std::filesystem::create_directories(request.output_dir, ec);
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            error = "cannot open " + tmp_path.string();
            return false;
        }
        const std::uint64_t header_len = header_str.size();
        out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
        out.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
        out.write(payload, static_cast<std::streamsize>(payload_bytes));
        out.flush();
        if (!out) {
            error = "write failed for " + tmp_path.string();
            return false;
        }
    }
    std::filesystem::rename(tmp_path, final_path, ec);
    if (ec) {
        error = "rename to " + final_path.string() + " failed: " + ec.message();
        return false;
    }

    // Lineage manifest — the Python request builder resolves
    // {parent_dir}/native_model_manifest.json for init_from_model parents.
    // The recipe fields make training-surface choices part of the lineage
    // contract: weights trained under a sliding window (or with LRSS
    // tensors) are adapted to that surface, and a resume that silently
    // changes it is a quiet corruption — the loader fail-closes on mismatch.
    nlohmann::json manifest = {
        {"backend", "native"},
        {"architecture_contract", request.model.architecture_contract.empty()
            ? request.architecture_contract : request.model.architecture_contract},
        {"weights_file", "model.safetensors"},
        {"weights_dtype", "BF16"},
        {"orientation", "native_row_major"},
        {"hidden_size", w.hidden_size},
        {"intermediate_size", w.intermediate_size},
        {"num_layers", w.global_num_layers > 0 ? w.global_num_layers : w.num_layers},
        {"vocab_size", w.vocab_size},
        {"heads", w.heads},
        {"kv_heads", w.kv_heads > 0 ? w.kv_heads : w.heads},
        {"seat", request.seat},
        {"family", request.family},
        {"version", request.version},
        {"tensor_count", tensor_count},
        {"total_weight_bytes", payload_bytes},
        {"attn_window", attn_window_tokens(w.hidden_size / (w.heads > 0 ? w.heads : 1))},
        {"lrss_enabled", w.lrss_enabled},
        {"lrss_scales", w.lrss_enabled ? w.lrss_scales : 0},
        {"lrss_anchors", w.lrss_enabled ? w.lrss_anchors : 0},
        {"lss_rank", w.lss_rank},
        {"pss_spike_joint_dim", w.pss_spike_joint_dim},
        {"pss_pred_rank", w.pss_pred_rank},
        {"precision_profile", request.precision_profile},
        {"fp8_storage", request.fp8_storage_format.empty() ? "none" : request.fp8_storage_format},
        {"fp8_compute", request.fp8_compute_path.empty() ? "none" : request.fp8_compute_path},
        {"native_fp8_tensorcore", request.precision_profile == "ampere_fp8_packed"
            ? nlohmann::json(false) : nlohmann::json(nullptr)},
        {"nvfp4_storage", request.nvfp4_storage_format.empty() ? "none" : request.nvfp4_storage_format},
        {"nvfp4_compute", request.nvfp4_compute_path.empty() ? "none" : request.nvfp4_compute_path},
        // The current NVFP4 port is a pack/cache surface only.  Keep this
        // null until the sm_120a arithmetic path and all backward directions
        // have a measured bring-up record.
        {"native_nvfp4_tensorcore", nlohmann::json(nullptr)},
        {"hardware_profile", request.hardware_profile},
        {"memory_profile", request.memory_profile},
        {"peer_transport", request.device.peer_transport.empty() ? "auto" : request.device.peer_transport},
        {"optimizer_type", request.optimizer_type},
        {"spec_hash", request.spec_hash},
        // Lion policy overrides as the request stated them (Phase 2,
        // 2026-07-22) -- see status.cpp's write_status for why these are
        // the request's own values, not the worker's resolved-with-env-
        // fallback ones.
        {"lion_policy", {
            {"lr_scale_override", request.lion_lr_scale_override},
            {"wd_scale_override", request.lion_wd_scale_override},
            {"beta1_override", request.lion_beta1_override},
            {"beta2_override", request.lion_beta2_override},
            {"trust_ratio_enabled_override", request.lion_trust_ratio_enabled_override},
            {"trust_ratio_lo_override", request.lion_trust_ratio_lo_override},
            {"trust_ratio_hi_override", request.lion_trust_ratio_hi_override},
        }},
        {"cumulative_opt_steps", cumulative_opt_steps},
    };
    if (pipeline_stage_count > 1) {
        manifest["model_parallel"] = true;
        manifest["pipeline_stage_count"] = pipeline_stage_count;
        manifest["pipeline_split_layer"] = pipeline_split_layer;
        manifest["checkpoint_resumable"] = false;
        manifest["checkpoint_contract"] = "merged_weights_only";
    }
    write_text(request.output_dir / "native_model_manifest.json",
               manifest.dump(2) + "\n");
    return true;
}

}  // namespace

bool save_lattice_weights_safetensors(
    const NativeRequest& request,
    const LatticeWeights& w,
    cudaStream_t stream,
    std::string& error,
    int cumulative_opt_steps
) {
    const auto tensors = enumerate_lattice_tensors(w);
    std::size_t total_bytes = 0;
    const std::string header_str = build_safetensors_header(request, tensors, total_bytes);

    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        error = "cudaStreamSynchronize failed before weight save";
        return false;
    }

    // Whole-body staging keeps the shared file writer simple; pageable is
    // fine here — the sync path only runs at end-of-burn.
    std::vector<char> payload(total_bytes);
    std::size_t off = 0;
    for (const auto& t : tensors) {
        const std::size_t bytes = t.numel * sizeof(__nv_bfloat16);
        if (cudaMemcpy(payload.data() + off, t.ptr, bytes,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            error = "D2H copy failed for " + t.name;
            return false;
        }
        off += bytes;
    }
    return write_safetensors_file(
        request, w, header_str, payload.data(), total_bytes,
        tensors.size(), error, cumulative_opt_steps);
}

bool save_model_parallel_lattice_weights_safetensors(
    const NativeRequest& request,
    const LatticeWeights& first_stage,
    int first_device,
    cudaStream_t first_stream,
    const LatticeWeights& second_stage,
    int second_device,
    cudaStream_t second_stream,
    std::string& error,
    int cumulative_opt_steps,
    int pipeline_split_layer
) {
    if (!first_stage.owns_embedding || first_stage.owns_output ||
        second_stage.owns_embedding || !second_stage.owns_output ||
        first_stage.global_num_layers != second_stage.global_num_layers ||
        first_stage.layer_offset != 0 ||
        second_stage.layer_offset != pipeline_split_layer) {
        error = "invalid two-stage ownership for merged model-parallel weight save";
        return false;
    }

    const auto first_tensors = enumerate_lattice_tensors(first_stage);
    const auto second_tensors = enumerate_lattice_tensors(second_stage);
    std::vector<TensorEntry> tensors;
    tensors.reserve(first_tensors.size() + second_tensors.size());
    tensors.insert(tensors.end(), first_tensors.begin(), first_tensors.end());
    tensors.insert(tensors.end(), second_tensors.begin(), second_tensors.end());

    std::size_t total_bytes = 0;
    const std::string header_str = build_safetensors_header(request, tensors, total_bytes);
    std::vector<char> payload(total_bytes);
    std::size_t offset = 0;

    const auto copy_stage = [&](const std::vector<TensorEntry>& stage_tensors,
                                int device, cudaStream_t stream) -> bool {
        if (cudaSetDevice(device) != cudaSuccess) {
            error = "cudaSetDevice failed before model-parallel weight save";
            return false;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            error = "cudaStreamSynchronize failed before model-parallel weight save";
            return false;
        }
        for (const auto& tensor : stage_tensors) {
            const std::size_t bytes = tensor.numel * sizeof(__nv_bfloat16);
            if (cudaMemcpy(payload.data() + offset, tensor.ptr, bytes,
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                error = "D2H copy failed for model-parallel tensor " + tensor.name;
                return false;
            }
            offset += bytes;
        }
        return true;
    };

    if (!copy_stage(first_tensors, first_device, first_stream) ||
        !copy_stage(second_tensors, second_device, second_stream) ||
        offset != total_bytes) {
        return false;
    }
    return write_safetensors_file(
        request, second_stage, header_str, payload.data(), total_bytes,
        tensors.size(), error, cumulative_opt_steps,
        /*pipeline_stage_count=*/2, pipeline_split_layer);
}

bool begin_async_weight_save(
    const NativeRequest& request,
    const LatticeWeights& w,
    cudaStream_t train_stream,
    AsyncWeightSaver& saver,
    std::string& error,
    int cumulative_opt_steps
) {
    if (saver.busy.load(std::memory_order_acquire)) {
        ++saver.saves_skipped;
        return true;  // best-effort periodic durability: skip, don't stall
    }
    // The previous writer finished (busy false) but its thread object still
    // needs collecting before reuse.
    if (saver.writer.joinable()) saver.writer.join();
    if (!saver.last_error.empty()) {
        // Surface (once) an error from the previous background write.
        error = "previous async save failed: " + saver.last_error;
        saver.last_error.clear();
        return false;
    }

    const auto tensors = enumerate_lattice_tensors(w);
    std::size_t total_bytes = 0;
    std::string header_str = build_safetensors_header(request, tensors, total_bytes);

    if (saver.pinned == nullptr || saver.capacity < total_bytes) {
        if (saver.pinned) cudaFreeHost(saver.pinned);
        saver.pinned = nullptr;
        saver.capacity = 0;
        if (cudaHostAlloc(&saver.pinned, total_bytes,
                          cudaHostAllocDefault) != cudaSuccess) {
            error = "cudaHostAlloc failed for async save staging";
            return false;
        }
        saver.capacity = total_bytes;
    }
    if (saver.copy_stream == nullptr &&
        cudaStreamCreateWithFlags(&saver.copy_stream,
                                  cudaStreamNonBlocking) != cudaSuccess) {
        error = "copy stream creation failed for async save";
        return false;
    }
    if (saver.ev_ready == nullptr &&
        cudaEventCreateWithFlags(&saver.ev_ready,
                                 cudaEventDisableTiming) != cudaSuccess) {
        error = "event creation failed for async save";
        return false;
    }
    if (saver.ev_copy_done == nullptr &&
        cudaEventCreateWithFlags(&saver.ev_copy_done,
                                 cudaEventDisableTiming) != cudaSuccess) {
        error = "event creation failed for async save";
        return false;
    }

    // Stage: copies ordered after everything queued on train_stream, and
    // train_stream's future work ordered after the copies (next optimizer
    // step must not mutate weights mid-snapshot).
    if (cudaEventRecord(saver.ev_ready, train_stream) != cudaSuccess) {
        error = "ev_ready record failed";
        return false;
    }
    if (cudaStreamWaitEvent(saver.copy_stream, saver.ev_ready, 0) != cudaSuccess) {
        error = "copy stream wait failed";
        return false;
    }
    std::size_t off = 0;
    for (const auto& t : tensors) {
        const std::size_t bytes = t.numel * sizeof(__nv_bfloat16);
        if (cudaMemcpyAsync(static_cast<char*>(saver.pinned) + off, t.ptr,
                            bytes, cudaMemcpyDeviceToHost,
                            saver.copy_stream) != cudaSuccess) {
            error = "async D2H failed for " + t.name;
            return false;
        }
        off += bytes;
    }
    if (cudaEventRecord(saver.ev_copy_done, saver.copy_stream) != cudaSuccess) {
        error = "ev_copy_done record failed";
        return false;
    }
    if (cudaStreamWaitEvent(train_stream, saver.ev_copy_done, 0) != cudaSuccess) {
        error = "train stream wait failed";
        return false;
    }

    saver.busy.store(true, std::memory_order_release);
    saver.writer = std::thread(
        [&saver, request, w, header = std::move(header_str), total_bytes,
         tensor_count = tensors.size(), cumulative_opt_steps]() {
            std::string err;
            if (cudaEventSynchronize(saver.ev_copy_done) != cudaSuccess) {
                err = "ev_copy_done synchronize failed";
            } else if (!write_safetensors_file(
                           request, w, header,
                           static_cast<const char*>(saver.pinned), total_bytes,
                           tensor_count, err, cumulative_opt_steps)) {
                // err already set
            } else {
                ++saver.saves_completed;
            }
            saver.last_error = err;
            saver.busy.store(false, std::memory_order_release);
        });
    return true;
}

void drain_async_weight_saver(AsyncWeightSaver& saver) {
    if (saver.writer.joinable()) saver.writer.join();
    if (saver.ev_copy_done) { cudaEventDestroy(saver.ev_copy_done); saver.ev_copy_done = nullptr; }
    if (saver.ev_ready)     { cudaEventDestroy(saver.ev_ready);     saver.ev_ready = nullptr; }
    if (saver.copy_stream)  { cudaStreamDestroy(saver.copy_stream); saver.copy_stream = nullptr; }
    if (saver.pinned)       { cudaFreeHost(saver.pinned);           saver.pinned = nullptr; }
    saver.capacity = 0;
}

bool load_lattice_weights_safetensors(
    const std::filesystem::path& dir_or_file,
    const LatticeWeights& w,
    cudaStream_t stream,
    std::string& error,
    const std::string& expected_spec_hash,
    int* out_cumulative_opt_steps
) {
    std::filesystem::path file = dir_or_file;
    if (std::filesystem::is_directory(file)) file /= "model.safetensors";
    if (!std::filesystem::is_regular_file(file)) {
        error = "no weights file at " + file.string();
        return false;
    }

    // Recipe-surface contract: if the sibling manifest records the training
    // surface, it must match the current run's.  Weights carry their
    // receptive field — resuming a W=128-trained body unwindowed (or vice
    // versa) is a silent scale-mismatch wedge, the exact class the lineage
    // gates exist to catch.  Manifests WITHOUT the field are pre-Era-13
    // full-causal bodies: with windows now default-on they fail shut too,
    // unless the migration is explicitly acknowledged.
    {
        const auto mpath = file.parent_path() / "native_model_manifest.json";
        if (std::filesystem::is_regular_file(mpath)) {
            std::ifstream min(mpath);
            nlohmann::json mj = nlohmann::json::parse(min, nullptr, false);
            if (!mj.is_discarded() && mj.contains("architecture_contract")) {
                const std::string parent_contract =
                    mj.value("architecture_contract", "");
                if (!w.architecture_contract.empty() &&
                    parent_contract != w.architecture_contract) {
                    error = "architecture_contract mismatch: parent=" +
                            parent_contract + ", current=" +
                            w.architecture_contract;
                    return false;
                }
            } else if (!mj.is_discarded() &&
                       w.architecture_contract != "ida_lattice_native_v1") {
                error = "parent manifest lacks architecture_contract for current " +
                        w.architecture_contract + " request";
                return false;
            }
            // Checkpoint-atomicity cross-check (2026-07-23): hand the
            // manifest's own recorded generation back to the caller so it
            // can compare against optimizer_state.safetensors' own
            // cumulative_opt_steps once that separate load also completes —
            // -1 (manifest predates this field, or parse failed) means
            // "not tracked," the caller skips the comparison rather than
            // treating absence as a mismatch.
            if (out_cumulative_opt_steps != nullptr) {
                *out_cumulative_opt_steps = (!mj.is_discarded() && mj.contains("cumulative_opt_steps"))
                    ? mj.value("cumulative_opt_steps", -1)
                    : -1;
            }
            // Identity check for a genuine same-lineage resume (Phase 3,
            // 2026-07-22) -- expected_spec_hash is only ever non-empty at
            // the resume_from_checkpoint call site, never init_from_model
            // (a child legitimately has a different spec than its parent).
            // Hard-fails, unlike every other check in this block: resuming
            // optimizer/cursor/curriculum state into the wrong BurnSpec
            // would silently corrupt training dynamics, not just fail a
            // shape check downstream.
            if (!expected_spec_hash.empty() && !mj.is_discarded() &&
                mj.contains("spec_hash")) {
                const std::string parent_hash = mj.value("spec_hash", "");
                if (!parent_hash.empty() && parent_hash != expected_spec_hash) {
                    error = "spec_hash mismatch on resume: checkpoint recorded " +
                            parent_hash + ", resuming request is " +
                            expected_spec_hash +
                            " (this is not the same burn's own checkpoint -- "
                            "use init_from_model for cross-lineage loads instead "
                            "of resume_from_checkpoint)";
                    return false;
                }
            }
            const int cur_w = attn_window_tokens(w.hidden_size / (w.heads > 0 ? w.heads : 1));
            if (!mj.is_discarded() && mj.contains("attn_window")) {
                const int parent_w = mj.value("attn_window", 0);
                if (parent_w != cur_w) {
                    error = "attn_window mismatch: parent trained at W=" +
                            std::to_string(parent_w) + ", current run W=" +
                            std::to_string(cur_w) +
                            " (set IDA_NATIVE_ATTN_WINDOW to match or "
                            "re-genesis)";
                    return false;
                }
            }
            // Precision-profile lineage: weights trained under one linear
            // format continue under the same one (re-genesis is the format
            // migration path, not resume).  The loader has no request, so
            // the current profile comes from the env the launch chain always
            // sets; unknown env skips the check (pre-field manifests pass —
            // the attn_window legacy gate below already fail-closes them).
            if (!mj.is_discarded() && mj.contains("precision_profile")) {
                const std::string parent_p = mj.value("precision_profile", "");
                const char* cur_p = std::getenv("IDA_NATIVE_PRECISION_PROFILE");
                const std::string cur_profile = cur_p ? cur_p : "";
                const char* legacy_p = std::getenv("IDA_NATIVE_ACCEPT_LEGACY_LINEAGE");
                if (!parent_p.empty() && !cur_profile.empty() &&
                    parent_p != cur_profile && !(legacy_p && legacy_p[0] == '1')) {
                    error = "precision_profile mismatch: parent trained under " +
                            parent_p + ", current run uses " + cur_profile +
                            " (IDA_NATIVE_ACCEPT_LEGACY_LINEAGE=1 to migrate "
                            "deliberately)";
                    return false;
                }
            }
            if (!mj.is_discarded() && !mj.contains("attn_window") && cur_w != 0) {
                const char* legacy = std::getenv("IDA_NATIVE_ACCEPT_LEGACY_LINEAGE");
                if (!(legacy && legacy[0] == '1')) {
                    error = "parent manifest predates the attn_window contract "
                            "(full-causal weights) but the current run uses W=" +
                            std::to_string(cur_w) +
                            "; set IDA_NATIVE_ATTN_WINDOW=0 to match the "
                            "parent, re-genesis, or set "
                            "IDA_NATIVE_ACCEPT_LEGACY_LINEAGE=1 to migrate "
                            "deliberately";
                    return false;
                }
            }
            // In a peer-pipeline load stage 0 owns only the embedding and
            // lower lattice layers. LRSS/LSS/PSS live exclusively on stage 1,
            // so validating their manifest contract on stage 0 would reject a
            // perfectly valid merged parent before that stage can load it.
            if (w.owns_output && !mj.is_discarded() && mj.contains("lrss_enabled")) {
                const bool parent_lrss = mj.value("lrss_enabled", false);
                if (parent_lrss != w.lrss_enabled) {
                    error = std::string("LRSS mismatch: parent ") +
                            (parent_lrss ? "has" : "lacks") +
                            " multiscale_memory tensors, current run " +
                            (w.lrss_enabled ? "expects" : "ignores") + " them";
                    return false;
                }
                if (w.lrss_enabled &&
                    (mj.value("lrss_scales", w.lrss_scales) != w.lrss_scales ||
                     mj.value("lrss_anchors", w.lrss_anchors) != w.lrss_anchors ||
                     mj.value("lss_rank", w.lss_rank) != w.lss_rank)) {
                    error = "LRSS/LSS config mismatch: parent J=" +
                            std::to_string(mj.value("lrss_scales", 0)) + "/A=" +
                            std::to_string(mj.value("lrss_anchors", 0)) + "/R=" +
                            std::to_string(mj.value("lss_rank", 0)) +
                            " vs current J=" + std::to_string(w.lrss_scales) +
                            "/A=" + std::to_string(w.lrss_anchors) +
                            "/R=" + std::to_string(w.lss_rank);
                    return false;
                }
                // PSS Stage 1b: down.weight's actual column count depends on
                // this field (2*H vs 2*H+K) -- a silent mismatch here is a
                // real tensor-shape corruption, not just a semantic drift,
                // so it fails closed exactly like lss_rank above.
                if (w.lrss_enabled && w.lss_rank > 0 &&
                    mj.value("pss_spike_joint_dim", w.pss_spike_joint_dim) !=
                        w.pss_spike_joint_dim) {
                    error = "PSS spike-joint mismatch: parent pss_spike_joint_dim=" +
                            std::to_string(mj.value("pss_spike_joint_dim", 0)) +
                            " vs current " + std::to_string(w.pss_spike_joint_dim) +
                            " (IDA_NATIVE_PSS_SPIKE_JOINT changed across a resume)";
                    return false;
                }
                // PSS Stage 2: the predictor is a shadow-mode AUXILIARY head
                // (never touches the trunk while the governor is off), so
                // 0<->N transitions across a lineage boundary are tolerated
                // rather than fail-closed (2026-07-15 relaxation -- the
                // worker/direct env split left production with mixed rank-0
                // and rank-64 parent manifests, and hard-failing either
                // direction wedges the queue):
                //   parent 0 / absent, current N  -> predictor fresh-inits
                //     (its tensors are skipped during load below);
                //   parent N, current 0           -> parent's extra
                //     pss_predictor.* tensors are simply not enumerated;
                //   parent N, current M (both >0, N != M) -> still a hard
                //     failure: a shape-changed predictor cannot be loaded
                //     and silently reshaping it would corrupt the aux head.
                {
                    const int parent_rank = mj.value("pss_pred_rank", 0);
                    if (parent_rank > 0 && w.pss_pred_rank > 0 &&
                        parent_rank != w.pss_pred_rank) {
                        error = "PSS predictor mismatch: parent pss_pred_rank=" +
                                std::to_string(parent_rank) +
                                " vs current " + std::to_string(w.pss_pred_rank) +
                                " (predictor rank changed across a resume)";
                        return false;
                    }
                    if (parent_rank != w.pss_pred_rank) {
                        std::fprintf(stderr,
                            "[ida_native_train] PSS predictor rank transition: "
                            "parent=%d current=%d (auxiliary head %s)\n",
                            parent_rank, w.pss_pred_rank,
                            w.pss_pred_rank > 0 ? "fresh-initialized"
                                                : "dropped");
                    }
                }
            }
        }
    }

    std::ifstream in(file, std::ios::binary);
    if (!in) {
        error = "cannot open " + file.string();
        return false;
    }
    std::uint64_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!in || header_len == 0 || header_len > (100ull << 20)) {
        error = "corrupt safetensors header length in " + file.string();
        return false;
    }
    std::string header_str(header_len, '\0');
    in.read(header_str.data(), static_cast<std::streamsize>(header_len));
    if (!in) {
        error = "truncated safetensors header in " + file.string();
        return false;
    }
    nlohmann::json header = nlohmann::json::parse(header_str, nullptr, false);
    if (header.is_discarded()) {
        error = "unparseable safetensors header in " + file.string();
        return false;
    }
    const std::string header_contract = header.contains("__metadata__")
        ? header["__metadata__"].value("format", "") : "";
    if (!w.architecture_contract.empty() &&
        header_contract != w.architecture_contract) {
        error = "architecture_contract mismatch in SafeTensors metadata: file=" +
                header_contract + ", current=" + w.architecture_contract;
        return false;
    }

    const std::size_t data_base = sizeof(header_len) + header_len;
    const auto tensors = enumerate_lattice_tensors(w);
    std::vector<char> staging;
    for (const auto& t : tensors) {
        if (!header.contains(t.name)) {
            // Declared-optional groups (see kOptionalTensorPrefixes): the
            // parent legitimately predates these tensors -- keep this burn's
            // fresh initialization for them and load everything else. Any
            // OTHER missing tensor is still fatal.
            if (tensor_optional_at_load(t.name)) {
                std::fprintf(stderr,
                    "[ida_native_train] parent lacks %s -- keeping fresh init\n",
                    t.name.c_str());
                continue;
            }
            error = "missing tensor " + t.name + " in " + file.string();
            return false;
        }
        const auto& entry = header[t.name];
        if (entry.value("dtype", "") != "BF16") {
            error = "tensor " + t.name + " dtype is not BF16";
            return false;
        }
        const auto shape = entry.value("shape", std::vector<std::int64_t>{});
        if (shape != t.shape) {
            std::ostringstream msg;
            msg << "shape mismatch for " << t.name << ": file [";
            for (std::size_t i = 0; i < shape.size(); ++i)
                msg << (i ? "," : "") << shape[i];
            msg << "] vs model [";
            for (std::size_t i = 0; i < t.shape.size(); ++i)
                msg << (i ? "," : "") << t.shape[i];
            msg << "]";
            error = msg.str();
            return false;
        }
        const auto offsets = entry.value("data_offsets", std::vector<std::uint64_t>{});
        const std::size_t bytes = t.numel * sizeof(__nv_bfloat16);
        if (offsets.size() != 2 || offsets[1] - offsets[0] != bytes) {
            error = "bad data_offsets for " + t.name;
            return false;
        }
        staging.resize(bytes);
        in.seekg(static_cast<std::streamoff>(data_base + offsets[0]));
        in.read(staging.data(), static_cast<std::streamsize>(bytes));
        if (!in) {
            error = "truncated tensor data for " + t.name;
            return false;
        }
        if (cudaMemcpyAsync(const_cast<__nv_bfloat16*>(t.ptr), staging.data(),
                            bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            error = "H2D copy failed for " + t.name;
            return false;
        }
        // staging is reused next iteration — the async copy must land first.
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            error = "stream sync failed while loading " + t.name;
            return false;
        }
    }
    return true;
}

// ── Optimizer state + resume cursor persistence (Phase 3, 2026-07-22) ───────
// See checkpoint.hpp's doc comment for the design rationale. Synchronous
// only (no periodic async variant yet, unlike the weight saver) -- called
// alongside save_lattice_weights_safetensors at the same end-of-burn/
// periodic-checkpoint boundaries.

bool save_lattice_opt_safetensors(
    const NativeRequest& request,
    const LatticeWeights& w,
    const LatticeOptState& opt,
    const ResumeState& resume_state,
    cudaStream_t stream,
    std::string& error
) {
    const auto tensors = enumerate_lattice_opt_tensors(w, opt);

    nlohmann::json header;
    std::size_t offset = 0;
    for (const auto& t : tensors) {
        const std::size_t elem_size = t.is_bf16 ? sizeof(__nv_bfloat16) : sizeof(float);
        const std::size_t bytes = t.numel * elem_size;
        header[t.name] = {
            {"dtype", t.is_bf16 ? "BF16" : "F32"},
            {"shape", t.shape},
            {"data_offsets", {offset, offset + bytes}},
        };
        offset += bytes;
    }
    header["__metadata__"] = {
        {"format", "ida_lattice_native_opt_v1"},
        {"seat", request.seat},
        {"family", request.family},
        {"version", request.version},
        {"spec_hash", request.spec_hash},
        {"optimizer_type", request.optimizer_type},
        {"optimizer_state_bf16", opt.optimizer_state_bf16},
        {"adam_step", opt.adam_step},
        {"cumulative_opt_steps", resume_state.cumulative_opt_steps},
        {"cumulative_micro_steps", resume_state.cumulative_micro_steps},
        {"dataset_cursor", static_cast<std::uint64_t>(resume_state.dataset_cursor)},
    };
    std::string header_str = header.dump();
    while (header_str.size() % 8 != 0) header_str.push_back(' ');
    const std::size_t total_bytes = offset;

    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        error = "cudaStreamSynchronize failed before optimizer-state save";
        return false;
    }
    std::vector<char> payload(total_bytes);
    std::size_t off = 0;
    for (const auto& t : tensors) {
        const std::size_t elem_size = t.is_bf16 ? sizeof(__nv_bfloat16) : sizeof(float);
        const std::size_t bytes = t.numel * elem_size;
        if (cudaMemcpy(payload.data() + off, t.ptr, bytes,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            error = "D2H copy failed for optimizer tensor " + t.name;
            return false;
        }
        off += bytes;
    }

    const auto final_path = request.output_dir / "optimizer_state.safetensors";
    const auto tmp_path   = request.output_dir / "optimizer_state.safetensors.tmp";
    std::error_code ec;
    std::filesystem::create_directories(request.output_dir, ec);
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            error = "cannot open " + tmp_path.string();
            return false;
        }
        const std::uint64_t header_len = header_str.size();
        out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
        out.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
        out.write(payload.data(), static_cast<std::streamsize>(total_bytes));
        out.flush();
        if (!out) {
            error = "write failed for " + tmp_path.string();
            return false;
        }
    }
    std::filesystem::rename(tmp_path, final_path, ec);
    if (ec) {
        error = "rename to " + final_path.string() + " failed: " + ec.message();
        return false;
    }
    return true;
}

bool load_lattice_opt_safetensors(
    const std::filesystem::path& dir_or_file,
    const LatticeWeights& w,
    LatticeOptState& opt,
    ResumeState& resume_state,
    cudaStream_t stream,
    std::string& error
) {
    std::filesystem::path file = dir_or_file;
    if (std::filesystem::is_directory(file)) file /= "optimizer_state.safetensors";
    if (!std::filesystem::is_regular_file(file)) {
        error = "no optimizer state file at " + file.string();
        return false;
    }
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        error = "cannot open " + file.string();
        return false;
    }
    std::uint64_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!in || header_len == 0 || header_len > (100ull << 20)) {
        error = "corrupt optimizer-state safetensors header length in " + file.string();
        return false;
    }
    std::string header_str(header_len, '\0');
    in.read(header_str.data(), static_cast<std::streamsize>(header_len));
    if (!in) {
        error = "truncated optimizer-state safetensors header in " + file.string();
        return false;
    }
    nlohmann::json header = nlohmann::json::parse(header_str, nullptr, false);
    if (header.is_discarded()) {
        error = "unparseable optimizer-state safetensors header in " + file.string();
        return false;
    }
    if (header.contains("__metadata__")) {
        const auto& meta = header["__metadata__"];
        const bool file_bf16 = meta.value("optimizer_state_bf16", false);
        if (file_bf16 != opt.optimizer_state_bf16) {
            error = "optimizer_state_bf16 mismatch: file has " +
                    std::string(file_bf16 ? "true" : "false") +
                    ", current run has " +
                    std::string(opt.optimizer_state_bf16 ? "true" : "false") +
                    " (a precision-profile change across a resume is a "
                    "re-genesis event, not a resumable transition)";
            return false;
        }
        opt.adam_step = meta.value("adam_step", 0);
        resume_state.cumulative_opt_steps = meta.value("cumulative_opt_steps", 0);
        resume_state.cumulative_micro_steps = meta.value("cumulative_micro_steps", 0);
        resume_state.dataset_cursor = static_cast<std::size_t>(
            meta.value("dataset_cursor", static_cast<std::uint64_t>(0)));
    }

    const std::size_t data_base = sizeof(header_len) + header_len;
    const auto tensors = enumerate_lattice_opt_tensors(w, opt);
    std::vector<char> staging;
    for (const auto& t : tensors) {
        if (!header.contains(t.name)) {
            if (tensor_optional_at_load(t.name)) {
                std::fprintf(stderr,
                    "[ida_native_train] parent lacks optimizer state for %s -- "
                    "keeping fresh init\n", t.name.c_str());
                continue;
            }
            error = "missing optimizer tensor " + t.name + " in " + file.string();
            return false;
        }
        const auto& entry = header[t.name];
        const std::string expected_dtype = t.is_bf16 ? "BF16" : "F32";
        if (entry.value("dtype", "") != expected_dtype) {
            error = "optimizer tensor " + t.name + " dtype is not " + expected_dtype;
            return false;
        }
        const auto shape = entry.value("shape", std::vector<std::int64_t>{});
        if (shape != t.shape) {
            error = "optimizer tensor shape mismatch for " + t.name;
            return false;
        }
        const std::size_t elem_size = t.is_bf16 ? sizeof(__nv_bfloat16) : sizeof(float);
        const std::size_t bytes = t.numel * elem_size;
        const auto offsets = entry.value("data_offsets", std::vector<std::uint64_t>{});
        if (offsets.size() != 2 || offsets[1] - offsets[0] != bytes) {
            error = "bad data_offsets for optimizer tensor " + t.name;
            return false;
        }
        staging.resize(bytes);
        in.seekg(static_cast<std::streamoff>(data_base + offsets[0]));
        in.read(staging.data(), static_cast<std::streamsize>(bytes));
        if (!in) {
            error = "truncated optimizer tensor data for " + t.name;
            return false;
        }
        if (cudaMemcpyAsync(const_cast<void*>(t.ptr), staging.data(), bytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            error = "H2D copy failed for optimizer tensor " + t.name;
            return false;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            error = "stream sync failed while loading optimizer tensor " + t.name;
            return false;
        }
    }
    return true;
}

// ── Standalone PSS predictor state (per-family×seat weights repo) ────────────

bool save_pss_pred_state_safetensors(
    const NativeRequest& request,
    const LatticeWeights& w,
    const std::filesystem::path& out_path,
    int cumulative_opt_steps,
    cudaStream_t stream,
    std::string& error
) {
    if (w.pss_pred_rank <= 0 || !w.pss_pred_down || !w.pss_pred_up) {
        error = "pss_pred head absent (rank=0) — nothing to save";
        return false;
    }
    const auto H = static_cast<std::int64_t>(w.hidden_size);
    const auto R = static_cast<std::int64_t>(w.pss_pred_rank);
    const std::size_t numel = static_cast<std::size_t>(H) * static_cast<std::size_t>(R);
    const std::size_t bytes_each = numel * sizeof(__nv_bfloat16);

    // safetensors __metadata__ values must be strings per the format spec.
    nlohmann::json meta = {
        {"format", "ida_pss_pred_state_v1"},
        {"predicts", "tail_ffn_output"},
        {"hidden_size", std::to_string(w.hidden_size)},
        {"vocab_size", std::to_string(w.vocab_size)},
        {"pss_pred_rank", std::to_string(w.pss_pred_rank)},
        {"family", request.family},
        {"seat", request.seat},
        {"version", request.version},
        {"cumulative_opt_steps", std::to_string(cumulative_opt_steps)},
    };
    nlohmann::json header;
    header["pss_predictor.down.weight"] = {
        {"dtype", "BF16"},
        {"shape", {H, R}},
        {"data_offsets", {0, bytes_each}},
    };
    header["pss_predictor.up.weight"] = {
        {"dtype", "BF16"},
        {"shape", {R, H}},
        {"data_offsets", {bytes_each, 2 * bytes_each}},
    };
    header["__metadata__"] = meta;
    std::string header_str = header.dump();
    while (header_str.size() % 8 != 0) header_str.push_back(' ');

    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        error = "cudaStreamSynchronize failed before pss-pred state save";
        return false;
    }
    std::vector<char> payload(2 * bytes_each);
    if (cudaMemcpy(payload.data(), w.pss_pred_down, bytes_each,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(payload.data() + bytes_each, w.pss_pred_up, bytes_each,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        error = "D2H copy failed for pss-pred state";
        return false;
    }

    std::error_code ec;
    if (out_path.has_parent_path())
        std::filesystem::create_directories(out_path.parent_path(), ec);
    const auto tmp_path = std::filesystem::path(out_path.string() + ".tmp");
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            error = "cannot open " + tmp_path.string();
            return false;
        }
        const std::uint64_t header_len = header_str.size();
        out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
        out.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
        out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        out.flush();
        if (!out) {
            error = "write failed for " + tmp_path.string();
            return false;
        }
    }
    std::filesystem::rename(tmp_path, out_path, ec);
    if (ec) {
        error = "rename to " + out_path.string() + " failed: " + ec.message();
        return false;
    }
    // Sidecar for the push/pull harness — identity without parsing safetensors.
    write_text(std::filesystem::path(out_path.string() + ".provenance.json"),
               meta.dump(2) + "\n");
    return true;
}

PssPredStateLoad load_pss_pred_state_safetensors(
    const NativeRequest& request,
    LatticeWeights& w,
    const std::filesystem::path& init_path,
    cudaStream_t stream,
    std::string& detail
) {
    w.pss_pred_prior_opt_steps = 0;
    if (w.pss_pred_rank <= 0 || !w.pss_pred_down || !w.pss_pred_up) {
        detail = "head absent (rank=0)";
        return PssPredStateLoad::kFresh;
    }
    // Per-burn env reads, no static cache — same convention as the PSS_PRED
    // env pair (worker-served burns re-read these each request).
    const char* force = std::getenv("IDA_NATIVE_PSS_PRED_STATE_RESET");
    if (force && force[0] == '1') {
        detail = "IDA_NATIVE_PSS_PRED_STATE_RESET=1 — forced fresh init";
        return PssPredStateLoad::kReset;
    }
    if (!std::filesystem::is_regular_file(init_path)) {
        detail = "no state file at " + init_path.string();
        return PssPredStateLoad::kFresh;
    }
    // Genesis rule: a body with no parent is a body this state never observed.
    // The head predicts token-position statistics of a specific body lineage,
    // so genesis resets it unless the cross-genesis probe explicitly opts in.
    if (request.parent.init_from_model.empty()) {
        const char* keep = std::getenv("IDA_NATIVE_PSS_PRED_STATE_KEEP_ON_GENESIS");
        if (!(keep && keep[0] == '1')) {
            detail = "fresh-genesis body (no parent) — head reset "
                     "(IDA_NATIVE_PSS_PRED_STATE_KEEP_ON_GENESIS=1 to resume anyway)";
            return PssPredStateLoad::kReset;
        }
    }

    std::ifstream in(init_path, std::ios::binary);
    if (!in) {
        detail = "cannot open " + init_path.string();
        return PssPredStateLoad::kReset;
    }
    std::uint64_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!in || header_len == 0 || header_len > (100ull << 20)) {
        detail = "corrupt safetensors header length in " + init_path.string();
        return PssPredStateLoad::kReset;
    }
    std::string header_str(header_len, '\0');
    in.read(header_str.data(), static_cast<std::streamsize>(header_len));
    if (!in) {
        detail = "truncated safetensors header in " + init_path.string();
        return PssPredStateLoad::kReset;
    }
    nlohmann::json header = nlohmann::json::parse(header_str, nullptr, false);
    if (header.is_discarded() || !header.contains("__metadata__")) {
        detail = "unparseable safetensors header in " + init_path.string();
        return PssPredStateLoad::kReset;
    }

    const auto& meta = header["__metadata__"];
    auto meta_int = [&](const char* key) {
        return std::atoi(meta.value(key, std::string("0")).c_str());
    };
    if (meta.value("format", "") != "ida_pss_pred_state_v1") {
        detail = "unknown format '" + meta.value("format", std::string("")) + "'";
        return PssPredStateLoad::kReset;
    }
    if (meta_int("hidden_size") != w.hidden_size ||
        meta_int("vocab_size") != w.vocab_size ||
        meta_int("pss_pred_rank") != w.pss_pred_rank) {
        detail = "identity mismatch: state H=" +
                 meta.value("hidden_size", std::string("?")) + "/V=" +
                 meta.value("vocab_size", std::string("?")) + "/R=" +
                 meta.value("pss_pred_rank", std::string("?")) +
                 " vs body H=" + std::to_string(w.hidden_size) +
                 "/V=" + std::to_string(w.vocab_size) +
                 "/R=" + std::to_string(w.pss_pred_rank);
        return PssPredStateLoad::kReset;
    }
    // The repo layout is per family×seat: cross-seat state is a different
    // body's map, reset rather than resumed.
    if (meta.value("family", "") != request.family ||
        meta.value("seat", "") != request.seat) {
        detail = "family/seat mismatch: state " +
                 meta.value("family", std::string("?")) + "/" +
                 meta.value("seat", std::string("?")) + " vs run " +
                 request.family + "/" + request.seat;
        return PssPredStateLoad::kReset;
    }

    const auto H = static_cast<std::int64_t>(w.hidden_size);
    const auto R = static_cast<std::int64_t>(w.pss_pred_rank);
    const std::size_t bytes_each =
        static_cast<std::size_t>(H) * static_cast<std::size_t>(R) * sizeof(__nv_bfloat16);
    const std::size_t data_base = sizeof(header_len) + header_len;

    struct Slot { const char* name; std::vector<std::int64_t> shape; __nv_bfloat16* dst; };
    const Slot slots[2] = {
        {"pss_predictor.down.weight", {H, R}, w.pss_pred_down},
        {"pss_predictor.up.weight",   {R, H}, w.pss_pred_up},
    };
    // Read and validate BOTH tensors into host staging before any device
    // write, so a bad file can never leave the head half-overwritten.
    std::vector<char> staging(2 * bytes_each);
    for (int i = 0; i < 2; ++i) {
        const auto& s = slots[i];
        if (!header.contains(s.name)) {
            detail = std::string("missing tensor ") + s.name;
            return PssPredStateLoad::kReset;
        }
        const auto& entry = header[s.name];
        if (entry.value("dtype", "") != "BF16" ||
            entry.value("shape", std::vector<std::int64_t>{}) != s.shape) {
            detail = std::string("dtype/shape mismatch for ") + s.name;
            return PssPredStateLoad::kReset;
        }
        const auto offsets = entry.value("data_offsets", std::vector<std::uint64_t>{});
        if (offsets.size() != 2 || offsets[1] - offsets[0] != bytes_each) {
            detail = std::string("bad data_offsets for ") + s.name;
            return PssPredStateLoad::kReset;
        }
        in.seekg(static_cast<std::streamoff>(data_base + offsets[0]));
        in.read(staging.data() + static_cast<std::size_t>(i) * bytes_each,
                static_cast<std::streamsize>(bytes_each));
        if (!in) {
            detail = std::string("truncated tensor data for ") + s.name;
            return PssPredStateLoad::kReset;
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (cudaMemcpyAsync(slots[i].dst,
                            staging.data() + static_cast<std::size_t>(i) * bytes_each,
                            bytes_each, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            detail = std::string("H2D copy failed for ") + slots[i].name;
            return PssPredStateLoad::kReset;
        }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        detail = "stream sync failed after pss-pred state load";
        return PssPredStateLoad::kReset;
    }

    w.pss_pred_prior_opt_steps = meta_int("cumulative_opt_steps");
    detail = init_path.string() + " (cumulative_opt_steps=" +
             std::to_string(w.pss_pred_prior_opt_steps) + ", trained_at_version=" +
             meta.value("version", std::string("?")) + ")";
    return PssPredStateLoad::kResumed;
}

void write_native_timeline_event(
    const NativeRequest& request,
    const std::string& phase,
    const std::string& event,
    double duration_s
) {
#if !IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
    (void)request;
    (void)phase;
    (void)event;
    (void)duration_s;
    return;
#else
    try {
        char buf[32];
        const std::time_t t = std::time(nullptr);
        std::tm tm_utc{};
#if defined(_WIN32)
        gmtime_s(&tm_utc, &t);
#else
        gmtime_r(&t, &tm_utc);
#endif
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);

        std::string student = request.seat;
        for (auto& c : student) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

        nlohmann::json row = {
            {"timestamp", buf},
            {"monotonic_s", std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count()},
            {"student", student.empty() ? nlohmann::json(nullptr) : nlohmann::json(student)},
            {"version", request.version},
            {"family", request.family},
            {"phase", phase},
            {"event", event},
            {"gpu_util_pct", nullptr},
            {"bytes", nullptr},
            {"duration_s", duration_s >= 0.0
                 ? nlohmann::json(std::round(duration_s * 1000.0) / 1000.0)
                 : nlohmann::json(nullptr)},
            {"reason", nullptr},
        };

        const auto path = request.repo_root / "artifacts" / "telemetry" / "burn_timeline.jsonl";
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        std::ofstream out(path, std::ios::app);
        if (!out) return;
        out << row.dump() << "\n";
    } catch (...) {
        // telemetry must never break a burn
    }
#endif
}

void write_checkpoint_artifacts(
    const NativeRequest& request,
    const SmokeStepResult& result
) {
    const auto checkpoint_dir = request.output_dir / checkpoint_dir_name(result.global_step);
    std::filesystem::create_directories(checkpoint_dir);

    const bool real_weights =
        std::filesystem::is_regular_file(request.output_dir / "model.safetensors");
    const char* artifact_format =
        real_weights ? "native_safetensors_v1" : "smoke_placeholder";
    const bool promotion_eligible = real_weights;

    if (real_weights) {
        write_text(
            checkpoint_dir / "native_weights_pointer.txt",
            "weights: ../model.safetensors (single durable body at student root)\n"
        );
    } else {
        write_text(
            checkpoint_dir / "native_weights_placeholder.txt",
            "native smoke placeholder weights\n"
        );
    }
    // Phase 3 (2026-07-22): optimizer_state.safetensors (adam/lion moments,
    // grad-accum ramp position, dataset cursor) now lives alongside
    // model.safetensors at the student root -- see save_lattice_opt_
    // safetensors. The two placeholder text files below are retired for any
    // burn produced under this engine revision; a pre-Phase-3 checkpoint
    // dir simply has neither file, which is itself the honest signal (the
    // absence of optimizer_state.safetensors already fails resume_from_
    // checkpoint closed via load_lattice_opt_safetensors' own file-not-found
    // check -- no separate placeholder text is needed to communicate that).
    const bool real_optimizer_state =
        std::filesystem::is_regular_file(request.output_dir / "optimizer_state.safetensors");
    if (real_optimizer_state) {
        write_text(
            checkpoint_dir / "native_optimizer_state_pointer.txt",
            "optimizer state: ../optimizer_state.safetensors (single durable "
            "body at student root, resumable via resume_from_checkpoint)\n"
        );
    } else {
        write_text(
            checkpoint_dir / "native_optimizer_placeholder.txt",
            "NOT PERSISTED: adam moments / lr position / grad-accum position.\n"
            "This checkpoint is a weight snapshot, not an exact resume point.\n"
        );
        write_text(
            checkpoint_dir / "native_rng_state_placeholder.txt",
            "NOT PERSISTED: dataset cursor / lrss anchor ring "
            "(no shuffle RNG exists in the native loader to persist).\n"
        );
    }

    std::ostringstream runtime;
    runtime
        << "{\n"
        << "  \"backend\": \"native\",\n"
        << "  \"attention_backend\": \"" << request.attention_backend << "\",\n"
        << "  \"precision_profile\": \"" << request.precision_profile << "\",\n"
        << "  \"fp8_storage\": \"" << (request.fp8_storage_format.empty() ? "none" : request.fp8_storage_format) << "\",\n"
        << "  \"fp8_compute\": \"" << (request.fp8_compute_path.empty() ? "none" : request.fp8_compute_path) << "\",\n"
        << "  \"native_fp8_tensorcore\": " << (request.precision_profile == "ampere_fp8_packed" ? "false" : "null") << ",\n"
        << "  \"nvfp4_storage\": \"" << (request.nvfp4_storage_format.empty() ? "none" : request.nvfp4_storage_format) << "\",\n"
        << "  \"nvfp4_compute\": \"" << (request.nvfp4_compute_path.empty() ? "none" : request.nvfp4_compute_path) << "\",\n"
        << "  \"native_nvfp4_tensorcore\": null,\n"
        << "  \"optimizer_state_precision\": \"" << request.optimizer_state_precision << "\",\n"
        << "  \"gradient_buffer_precision\": \"" << request.gradient_buffer_precision << "\",\n"
        << "  \"gemm_accumulator_precision\": \"" << request.gemm_accumulator_precision << "\",\n"
        << "  \"engine_revision\": \"" << request.engine_revision << "\",\n"
        << "  \"architecture_compatibility\": \"" << request.architecture_compatibility << "\",\n"
        << "  \"promotion_eligible\": " << (promotion_eligible ? "true" : "false") << ",\n"
        << "  \"resumable\": " << (real_optimizer_state ? "true" : "false") << ",\n"
        << "  \"tokens_processed\": " << result.tokens_processed << "\n"
        << "}\n";
    write_text(checkpoint_dir / "native_runtime.json", runtime.str());

    std::ostringstream manifest;
    manifest
        << "{\n"
        << "  \"backend\": \"native\",\n"
        << "  \"attention_backend\": \"" << request.attention_backend << "\",\n"
        << "  \"precision_profile\": \"" << request.precision_profile << "\",\n"
        << "  \"fp8_storage\": \"" << (request.fp8_storage_format.empty() ? "none" : request.fp8_storage_format) << "\",\n"
        << "  \"fp8_compute\": \"" << (request.fp8_compute_path.empty() ? "none" : request.fp8_compute_path) << "\",\n"
        << "  \"native_fp8_tensorcore\": " << (request.precision_profile == "ampere_fp8_packed" ? "false" : "null") << ",\n"
        << "  \"nvfp4_storage\": \"" << (request.nvfp4_storage_format.empty() ? "none" : request.nvfp4_storage_format) << "\",\n"
        << "  \"nvfp4_compute\": \"" << (request.nvfp4_compute_path.empty() ? "none" : request.nvfp4_compute_path) << "\",\n"
        << "  \"native_nvfp4_tensorcore\": null,\n"
        << "  \"optimizer_state_precision\": \"" << request.optimizer_state_precision << "\",\n"
        << "  \"gradient_buffer_precision\": \"" << request.gradient_buffer_precision << "\",\n"
        << "  \"gemm_accumulator_precision\": \"" << request.gemm_accumulator_precision << "\",\n"
        << "  \"engine_revision\": \"" << request.engine_revision << "\",\n"
        << "  \"cuda_arch\": \"" << request.device.required_arch << "\",\n"
        << "  \"precision\": \"" << request.device.precision << "\",\n"
        << "  \"seat\": \"" << request.seat << "\",\n"
        << "  \"family\": \"" << request.family << "\",\n"
        << "  \"version\": \"" << request.version << "\",\n"
        << "  \"global_step\": " << result.global_step << ",\n"
        << "  \"artifact_format\": \"" << artifact_format << "\",\n"
        << "  \"promotion_eligible\": " << (promotion_eligible ? "true" : "false") << "\n"
        << "}\n";
    write_text(checkpoint_dir / "native_checkpoint_manifest.json", manifest.str());

    std::ostringstream input_manifest;
    input_manifest
        << "{\n"
        << "  \"input_mode\": \"request_paths_only\",\n"
        << "  \"dataset_path\": \"" << request.dataset_path.string() << "\",\n"
        << "  \"tokenizer_path\": \"" << request.tokenizer_path.string() << "\"\n"
        << "}\n";
    write_text(checkpoint_dir / "native_input_manifest.json", input_manifest.str());
}

void write_final_artifacts(
    const NativeRequest& request,
    const SmokeStepResult& result
) {
    const auto final_dir = request.output_dir / "final";
    std::filesystem::create_directories(final_dir);

    const bool real_weights =
        std::filesystem::is_regular_file(request.output_dir / "model.safetensors");
    if (real_weights) {
        write_text(
            final_dir / "native_weights_pointer.txt",
            "weights: ../model.safetensors (single durable body at student root)\n"
        );
    } else {
        write_text(
            final_dir / "native_smoke_placeholder.txt",
            "native smoke placeholder final weights\n"
        );
    }

    std::ostringstream config;
    config
        << "{\n"
        << "  \"backend\": \"native\",\n"
        << "  \"attention_backend\": \"" << request.attention_backend << "\",\n"
        << "  \"precision_profile\": \"" << request.precision_profile << "\",\n"
        << "  \"optimizer_state_precision\": \"" << request.optimizer_state_precision << "\",\n"
        << "  \"gradient_buffer_precision\": \"" << request.gradient_buffer_precision << "\",\n"
        << "  \"gemm_accumulator_precision\": \"" << request.gemm_accumulator_precision << "\",\n"
        << "  \"engine_revision\": \"" << request.engine_revision << "\",\n"
        << "  \"architecture_compatibility\": \"" << request.architecture_compatibility << "\",\n"
        << "  \"promotion_eligible\": " << (real_weights ? "true" : "false") << ",\n"
        << "  \"global_step\": " << result.global_step << "\n"
        << "}\n";
    write_text(final_dir / "native_model_config.json", config.str());
}

}  // namespace ida_native
