#include "ida_native/request.hpp"

#include <array>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace ida_native {

#ifndef IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
#define IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY 0
#endif

namespace {

using json = nlohmann::json;

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("unable to open request json: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::filesystem::path as_path(const json& value) {
    if (value.is_null()) {
        return {};
    }
    return std::filesystem::path(value.get<std::string>());
}

// Like as_path but looks up by key, returning empty path for missing or null.
std::filesystem::path opt_path(const json& obj, const char* key) {
    const auto it = obj.find(key);
    if (it == obj.end() || it->is_null()) {
        return {};
    }
    return std::filesystem::path(it->get<std::string>());
}

template <typename T>
T get_or(const json& obj, const char* key, T fallback) {
    const auto it = obj.find(key);
    if (it == obj.end() || it->is_null()) {
        return fallback;
    }
    return it->get<T>();
}

void reject_governance_fields(const json& value) {
    static constexpr std::array<std::string_view, 22> forbidden{
        "org", "organization", "role", "role_tier", "justification",
        "transition", "audit", "audit_event", "domain_event",
        "execute_transition", "append_domain_event", "promotion",
        "promotion_enabled", "telemetry", "ontology", "evidence",
        "socket", "serve", "queue", "worker", "worker_manifest",
        "manifest",
    };

    if (value.is_object()) {
        for (const auto& [key, child] : value.items()) {
            std::string normalized = key;
            for (char& character : normalized) {
                character = static_cast<char>(std::tolower(
                    static_cast<unsigned char>(character)));
            }
            if (std::find(
                    forbidden.begin(), forbidden.end(), std::string_view(normalized)) !=
                forbidden.end()) {
                throw std::runtime_error("governance fields are not accepted by the native boundary");
            }
#if !IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
            static constexpr std::array<std::string_view, 5> private_fields{
                "ontology_required", "analytics_required", "ontology_path",
                "analytics_path", "analytics_contract",
            };
            if (std::find(
                    private_fields.begin(), private_fields.end(),
                    std::string_view(normalized)) != private_fields.end()) {
                throw std::runtime_error(
                    "private observability fields are not accepted by the public native boundary");
            }
#endif
            reject_governance_fields(child);
        }
    } else if (value.is_array()) {
        for (const auto& child : value) {
            reject_governance_fields(child);
        }
    }
}

}  // namespace

NativeRequest load_request(const std::filesystem::path& path) {
    const json payload = json::parse(read_text(path));
    reject_governance_fields(payload);

    NativeRequest request{};
    request.request_json_path = path;
    request.backend = get_or<std::string>(payload, "backend", "native");
    request.attention_backend = get_or<std::string>(
        payload,
        "attention_backend",
        "scalar_flash"
    );
    request.precision_profile = get_or<std::string>(
        payload,
        "precision_profile",
        "legacy_fp8"
    );
    request.hardware_profile = get_or<std::string>(payload, "hardware_profile", "");
    request.memory_profile = get_or<std::string>(payload, "memory_profile", "");
    request.fp8_storage_format = get_or<std::string>(payload, "fp8_storage_format", "");
    request.fp8_compute_path = get_or<std::string>(payload, "fp8_compute_path", "");
    request.fp8_weight_scale_mode = get_or<std::string>(
        payload, "fp8_weight_scale_mode", "");
    request.nvfp4_storage_format = get_or<std::string>(
        payload, "nvfp4_storage_format", "");
    request.nvfp4_compute_path = get_or<std::string>(
        payload, "nvfp4_compute_path", "");
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
    request.ontology_required = get_or<bool>(payload, "ontology_required", false);
    request.analytics_required = get_or<bool>(payload, "analytics_required", false);
    request.ontology_path = as_path(payload.value("ontology_path", ""));
    request.analytics_path = as_path(payload.value("analytics_path", ""));
    request.analytics_contract = get_or<std::string>(
        payload, "analytics_contract", "");
#endif
    request.arena_reserve_bytes = get_or<std::int64_t>(
        payload, "arena_reserve_bytes", -1);
    request.min_free_vram_bytes = get_or<std::int64_t>(
        payload, "min_free_vram_bytes", -1);
    request.optimizer_state_precision = get_or<std::string>(
        payload,
        "optimizer_state_precision",
        "fp32"
    );
    request.optimizer_type = get_or<std::string>(
        payload,
        "optimizer_type",
        "lion"
    );
    if (request.optimizer_type != "lion") {
        throw std::runtime_error("Adam and AdamW optimizers are disabled; only optimizer_type=lion is enabled");
    }
    request.gradient_buffer_precision = get_or<std::string>(
        payload,
        "gradient_buffer_precision",
        "fp32"
    );
    request.gemm_accumulator_precision = get_or<std::string>(
        payload,
        "gemm_accumulator_precision",
        "fp32"
    );
    request.config_path = as_path(payload.value("config_path", ""));
    request.dataset_path = as_path(payload.value("dataset_path", ""));
    request.output_dir = as_path(payload.value("output_dir", ""));
    request.repo_root = as_path(payload.value("repo_root", ""));
    request.status_file = as_path(payload.value("status_file", ""));
    request.tokenizer_path = as_path(payload.value("tokenizer_path", ""));
    request.family = get_or<std::string>(payload, "family", "edge");
    request.job_id = get_or<std::string>(payload, "job_id", "");
    request.seat = get_or<std::string>(payload, "seat", "UNKNOWN");
    request.version = get_or<std::string>(payload, "version", "unknown");
    request.engine_revision = get_or<std::string>(payload, "engine_revision", "0.2.0");
    request.architecture_compatibility = get_or<std::string>(
        payload,
        "architecture_compatibility",
        "native_smoke_only"
    );
    request.architecture_contract = get_or<std::string>(
        payload,
        "architecture_contract",
        "ida_lattice_native_v1"
    );
    request.expected_terminal_phase = get_or<std::string>(
        payload,
        "expected_terminal_phase",
        "native_smoke_complete"
    );
    request.batch_size_override = get_or<int>(payload, "batch_size_override", 0);
    request.grad_accum_override = get_or<int>(payload, "grad_accum_override", 0);
    request.native_max_steps = get_or<int>(payload, "native_max_steps", 1);
    request.max_samples = get_or<int>(payload, "max_samples", 0);
    request.seed = get_or<int>(payload, "seed", 1337);
    request.promotion_enabled = get_or<bool>(payload, "promotion_enabled", false);
    // PSS Stage 2 per-burn fields (see request.hpp): -1 = absent -> env
    // fallback. Negative ranks other than the sentinel are clamped to off.
    request.pss_pred_rank = get_or<int>(payload, "pss_pred_rank", -1);
    if (request.pss_pred_rank < -1) request.pss_pred_rank = 0;
    request.pss_pred_lr_scale = static_cast<float>(
        get_or<double>(payload, "pss_pred_lr_scale", -1.0));
    request.pss_pred_aux_weight = static_cast<float>(
        get_or<double>(payload, "pss_pred_aux_weight", -1.0));
    request.pss_conditioning_mode = get_or<std::string>(
        payload, "pss_conditioning_mode", "");
    request.pss_aux_normalize_mode = get_or<std::string>(
        payload, "pss_aux_normalize_mode", "");
    request.pss_pred_state_out = get_or<std::string>(payload, "pss_pred_state_out", "");
    request.pss_pred_state_init_path = get_or<std::string>(
        payload, "pss_pred_state_init_path", "");
    request.global_clip_override = static_cast<float>(
        get_or<double>(payload, "global_clip_override", -1.0));

    request.act_row_clip_override = static_cast<float>(
        get_or<double>(payload, "act_row_clip_override", -1.0));
    request.spec_hash = get_or<std::string>(payload, "spec_hash", "");
    request.lion_lr_scale_override = static_cast<float>(
        get_or<double>(payload, "lion_lr_scale_override", -1.0));
    request.lion_wd_scale_override = static_cast<float>(
        get_or<double>(payload, "lion_wd_scale_override", -1.0));
    request.lion_beta1_override = static_cast<float>(
        get_or<double>(payload, "lion_beta1_override", -1.0));
    request.lion_beta2_override = static_cast<float>(
        get_or<double>(payload, "lion_beta2_override", -1.0));
    request.lion_trust_ratio_enabled_override = get_or<int>(
        payload, "lion_trust_ratio_enabled_override", -1);
    request.lion_trust_ratio_lo_override = static_cast<float>(
        get_or<double>(payload, "lion_trust_ratio_lo_override", -1.0));
    request.lion_trust_ratio_hi_override = static_cast<float>(
        get_or<double>(payload, "lion_trust_ratio_hi_override", -1.0));

    const json device = payload.value("device", json::object());
    request.device.runtime = get_or<std::string>(
        device,
        "runtime",
        get_or<std::string>(payload, "runtime", "cuda")
    );
    request.device.required_arch = get_or<std::string>(
        device,
        "required_arch",
        get_or<std::string>(payload, "required_arch", "sm_90")
    );
    request.device.precision = get_or<std::string>(
        device,
        "precision",
        get_or<std::string>(payload, "precision", "bf16")
    );
    request.device.peer_transport = get_or<std::string>(
        device,
        "peer_transport",
        get_or<std::string>(payload, "peer_transport", "auto")
    );
    if (const auto it = device.find("model_parallel_devices");
        it != device.end() && !it->is_null()) {
        if (!it->is_array()) {
            throw std::runtime_error("device.model_parallel_devices must be an array");
        }
        request.device.model_parallel_devices = it->get<std::vector<int>>();
    }
    request.device.pipeline_split_layer = get_or<int>(device, "pipeline_split_layer", 0);

    const json model = payload.value("model", json::object());
    request.model.architecture_contract = get_or<std::string>(
        model,
        "architecture_contract",
        request.architecture_contract
    );
    request.model.hidden_size = get_or<int>(model, "hidden_size", 0);
    request.model.intermediate_size = get_or<int>(model, "intermediate_size", 0);
    request.model.layers = get_or<int>(model, "layers", 0);
    request.model.heads = get_or<int>(model, "heads", 0);
    request.model.kv_heads = get_or<int>(model, "kv_heads", request.model.heads);
    request.model.vocab_size = get_or<int>(model, "vocab_size", 0);
    // Cognitive-architecture sparse MoE port (2026-07-21). 0/false = off.
    request.model.num_cognitive_routes = get_or<int>(model, "num_cognitive_routes", 0);
    request.model.top_k_routes = get_or<int>(model, "top_k_routes", 0);
    request.model.num_personality_experts = get_or<int>(model, "num_personality_experts", 0);
    request.model.personality_residual_expert_width = get_or<int>(
        model, "personality_residual_expert_width", 0);
    request.model.top_k_experts = get_or<int>(model, "top_k_experts", 0);
    request.model.expert_balancing_loss_coef =
        get_or<float>(model, "expert_balancing_loss_coef", 0.0f);
    request.model.use_personality_residual_experts = get_or<bool>(
        model, "use_personality_residual_experts", false);
    request.model.local_attention_window = get_or<int>(model, "local_attention_window", 0);
    request.model.moe_native_fp4 = get_or<bool>(model, "moe_native_fp4", false);
    request.model.generic_moe_num_experts = get_or<int>(
        model, "generic_moe_num_experts", 0);
    request.model.generic_moe_top_k = get_or<int>(model, "generic_moe_top_k", 0);
    request.model.generic_moe_expert_width = get_or<int>(
        model, "generic_moe_expert_width", 0);
    request.model.generic_moe_shared_expert_width = get_or<int>(
        model, "generic_moe_shared_expert_width", 0);
    request.model.generic_moe_normalize_topk = get_or<bool>(
        model, "generic_moe_normalize_topk", true);
    request.model.qkv_bias = get_or<bool>(model, "qkv_bias", false);
    request.model.rope_theta = get_or<float>(model, "rope_theta", 0.0f);
    request.model.normalization_type = get_or<std::string>(
        model, "normalization_type", "rmsnorm");
    request.model.activation_type = get_or<std::string>(
        model, "activation_type", "swiglu");
    request.model.position_embedding_type = get_or<std::string>(
        model, "position_embedding_type", "none");
    request.model.norm_eps = get_or<float>(model, "norm_eps", 1.0e-6f);
    request.model.max_position_embeddings = get_or<int>(
        model, "max_position_embeddings", 0);
    request.model.projection_bias = get_or<bool>(model, "projection_bias", false);
    request.model.tied_embeddings = get_or<bool>(model, "tied_embeddings", false);

    const json training = payload.value("training", json::object());
    request.training.microbatch = get_or<int>(training, "microbatch",
        get_or<int>(training, "per_device_train_batch_size", 0));
    request.training.grad_accumulation = get_or<int>(training, "grad_accumulation",
        get_or<int>(training, "gradient_accumulation_steps", 0));
    request.training.learning_rate = get_or<double>(training, "learning_rate", 0.0);
    request.training.max_steps = get_or<int>(
        training,
        "max_steps",
        request.native_max_steps
    );

    const json input = payload.value("input", json::object());
    request.input.token_blocks = as_path(input.value("token_blocks", ""));
    request.input.label_blocks = as_path(input.value("label_blocks", ""));
    request.input.seg_blocks   = as_path(input.value("seg_blocks", ""));
    request.input.native_input_dir = as_path(payload.value("native_input_dir", ""));
    request.input.native_input_manifest = as_path(payload.value("native_input_manifest", ""));
    if (input.contains("shape") && input["shape"].is_array() && input["shape"].size() >= 2) {
        request.input.batch_size = input["shape"][0].get<int>();
        request.input.sequence_length = input["shape"][1].get<int>();
    } else {
        request.input.batch_size = get_or<int>(input, "batch_size", 0);
        request.input.sequence_length = get_or<int>(input, "sequence_length", 0);
    }
    request.input.num_sequences = get_or<int>(input, "num_sequences", 0);

    const json parent = payload.value("parent", json::object());
    request.parent.backend = get_or<std::string>(parent, "backend", "");
    request.parent.manifest = opt_path(parent, "manifest");
    request.parent.init_from_model = opt_path(payload, "init_from_model");
    request.parent.resume_from_checkpoint = opt_path(payload, "resume_from_checkpoint");

    if (request.status_file.empty()) {
        throw std::runtime_error("request.status_file is required");
    }
    if (request.output_dir.empty()) {
        throw std::runtime_error("request.output_dir is required");
    }
    return request;
}

}  // namespace ida_native
