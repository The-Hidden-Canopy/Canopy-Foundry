#include <cctype>
#include <algorithm>
#include <chrono>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "ida_native/arena.hpp"
#include "ida_native/cli_welcome.hpp"
#include "ida_native/checkpoint.hpp"
#include "ida_native/cuda_check.hpp"
#include "ida_native/device_info.hpp"
#include "ida_native/backend_contract.hpp"
#include "ida_native/request.hpp"
#include "ida_native/status.hpp"
#include "ida_native/telemetry.hpp"
#include "ida_native/trainer.hpp"

namespace {

using json = nlohmann::json;

struct Args {
    std::optional<std::filesystem::path> request_json;
    bool self_check{false};
    int device_id{0};
    int require_cuda_arch{90};
    bool require_cublaslt{false};
    bool require_private_pool{false};
};

void print_usage() {
    ida_native::print_cli_welcome(std::cout);
    std::cout
        << "Usage: ida_native_train --request-json REQUEST.json [--device GPU]\n"
        << "       ida_native_train --self-check [--device GPU]\n"
        << "\n"
        << "This public binary performs one-shot native training only.\n"
        << "Private Hub transports and orchestration are not part of this executable.\n";
}

Args parse_args(int argc, char** argv) {
    Args args{};
    for (int i = 1; i < argc; ++i) {
        const std::string_view token(argv[i]);
        if (token == "--help" || token == "-h") {
            print_usage();
            std::exit(0);
        } else if (token == "--request-json") {
            if (i + 1 >= argc) throw std::runtime_error("--request-json requires a path");
            args.request_json = std::filesystem::path(argv[++i]);
        } else if (token == "--self-check") {
            args.self_check = true;
        } else if (token == "--device") {
            if (i + 1 >= argc) throw std::runtime_error("--device requires an integer");
            args.device_id = std::stoi(argv[++i]);
        } else if (token == "--require-cuda-arch") {
            if (i + 1 >= argc) throw std::runtime_error("--require-cuda-arch requires an integer");
            args.require_cuda_arch = std::stoi(argv[++i]);
        } else if (token == "--require-cublaslt") {
            args.require_cublaslt = true;
        } else if (token == "--require-private-pool") {
            args.require_private_pool = true;
        } else {
            throw std::runtime_error("unknown or unsupported option: " + std::string(token));
        }
    }
    return args;
}

// Enables mapped (zero-copy) host memory for this device context. Must run
// before the context is actually initialized by any allocation, so this is
// called immediately after cudaSetDevice at every real-training entry point
// (not self_check, which never allocates optimizer state). Gated by the same
// env var trainer.cu checks before actually using host-mapped allocations
// for AdamW moments (see IDA_NATIVE_OPTIM_STATE_HOST_OFFLOAD) — calling this
// when the feature is off is a harmless no-op, so the check here is just to
// avoid the flag call's overhead/log noise on the common path.
static void maybe_enable_host_mapped_memory(int device_id) {
    // Era 13: the optimizer-state host offload defaults ON (trainer.cu), so
    // the mapped-host device flag must too — this gate mirrors
    // optim_state_host_offload_enabled() exactly (=0 disables both).
    const char* e = std::getenv("IDA_NATIVE_OPTIM_STATE_HOST_OFFLOAD");
    if (!e || e[0] == '1') {
        IDA_CUDA_CHECK(cudaSetDevice(device_id));
        IDA_CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    }
}

void run_self_check(const Args& args) {
    int device = args.device_id;
    IDA_CUDA_CHECK(cudaSetDevice(device));
    IDA_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    IDA_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const int actual_arch = prop.major * 10 + prop.minor;
    // Capability floor, not exact match (2026-08-12, external port audit):
    // an exact-equality check rejects every arch newer than the one this
    // binary was built to require, including ones that are a strict
    // superset of its capabilities (e.g. sm_120 rejected by a require=90
    // build, even though sm_120 satisfies everything sm_90 does). This is
    // a conscious loosening of admission -- it no longer catches "wrong,
    // unvalidated arch" the way an exact match did, only "arch too old."
    // A no-op on this box (two H100s = sm_90 either way); it matters once
    // a newer/different card (e.g. Blackwell) runs this same binary.
    if (actual_arch < args.require_cuda_arch) {
        throw std::runtime_error(
            "native self-check: required sm" + std::to_string(args.require_cuda_arch) +
            " but found sm" + std::to_string(actual_arch) +
            " (" + std::string(prop.name) + ")"
        );
    }

    if (args.require_cublaslt) {
        cublasLtHandle_t handle{};
        const auto status = cublasLtCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasLtCreate failed during native self-check");
        }
        cublasLtDestroy(handle);
    }

    if (args.require_private_pool) {
        auto arena = ida_native::create_arena(device);
        if (arena.pool == nullptr) {
            throw std::runtime_error("native self-check could not create a private cudaMemPool");
        }
        ida_native::destroy_arena(arena);
    }
}

std::string utc_timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    const auto t = std::chrono::system_clock::to_time_t(now);
    char buf[32]{};
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}

std::optional<std::string> validate_attention_contract(
    const ida_native::NativeRequest& request
) {
    if (request.attention_backend.empty() || request.attention_backend == "scalar_flash") {
        if (request.precision_profile == ida_native::kBlackwellNvfp4PrecisionProfile) {
            return "blackwell_nvfp4 requires a private advanced runtime package";
        }
        // legacy_bf16 added 2026-07-22 -- see the matching validate_attention_
        // request in trainer.cu for why this was never a hardware requirement.
        // This is a THIRD, independent copy of that same check (found via a
        // real deploy failure -- keep all three in sync, this codebase
        // duplicates this validation rather than sharing it across the CLI
        // entry point, the worker path, and the Python wrapper).
        if (request.precision_profile != "legacy_fp8" &&
            request.precision_profile != "legacy_bf16" &&
            request.precision_profile != "ampere_fp8_packed") {
            return "scalar_flash attention backend requires precision_profile=legacy_fp8, legacy_bf16, or ampere_fp8_packed";
        }
        if (request.model.heads > 0) {
            const int kv_heads = request.model.kv_heads > 0
                ? request.model.kv_heads : request.model.heads;
            if (kv_heads <= 0 || kv_heads > request.model.heads ||
                (request.model.heads % kv_heads) != 0) {
                return "scalar_flash attention requires kv_heads <= heads and heads divisible by kv_heads";
            }
            if (kv_heads != request.model.heads &&
                request.precision_profile != "legacy_bf16") {
                return "native GQA currently requires scalar_flash with precision_profile=legacy_bf16";
            }
        }
        return std::nullopt;
    }
    if (request.attention_backend == ida_native::kBlackwellMxf4Fp4Backend) {
        return "blackwell_mxf4_fp4 requires a private advanced runtime package";
    }
    const bool is_packed_fp4 = request.attention_backend == "hopper_wgmma_packed_fp4";
    const bool is_wgmma_fp8  = request.attention_backend == "hopper_wgmma_fp8";
    if (!is_packed_fp4 && !is_wgmma_fp8) {
        return "unknown native attention backend: " + request.attention_backend;
    }
    return request.attention_backend +
           " requires a private advanced runtime package";
    // Advanced backend selection is deployment-owned. The public target never
    // exposes its launch shape, precision, or device contract.
    return request.attention_backend +
           " requires a private advanced runtime package";
}

static std::optional<std::string> validate_ampere_fp8_contract(
    const ida_native::NativeRequest& request
) {
    const bool ampere_profile = request.precision_profile == "ampere_fp8_packed";
    const bool ampere_3050 = request.hardware_profile == "rtx3050_swift";
    const bool ampere_3090 = request.hardware_profile == "rtx3090_swift";
    const bool ampere_3090_generic = request.hardware_profile == "rtx3090_ampere_packed";
    const bool ampere_3090_moe_1f1b = request.hardware_profile == "rtx3090_moe_1f1b_bf16grad";
    // rtx3050_edge extends the same weight-only E4M3/BF16-dequant contract to
    // Edge Full's head_dim=64 shape. fp8_pack_bf16_e4m3_raw_and_dequant packs
    // each weight tensor by flat element count with no head_dim dependency of
    // its own -- head_dim=32 below was only ever "the shape Swift happens to
    // use", not a computational requirement (confirmed 2026-08-23 reading
    // build_ampere_packed_weights). microbatch=48 is the empirically measured
    // real working-set ceiling on a 4096 MiB card with arena_reserve_bytes=0
    // (~3075 MiB / 75%, leaving genuine headroom for a laptop's own desktop
    // compositor sharing the GPU); see resolve_rtx3050_edge_profile in
    // train_student_native.py for the full measurement.
    const bool ampere_edge = request.hardware_profile == "rtx3050_edge";
    const bool ampere_hardware = ampere_3050 || ampere_3090 || ampere_3090_generic ||
                                  ampere_3090_moe_1f1b || ampere_edge;
    if (!ampere_profile && !ampere_hardware) return std::nullopt;
    if (ampere_3090_moe_1f1b) {
        if (!ampere_profile) {
            return "rtx3090_moe_1f1b_bf16grad requires precision_profile=ampere_fp8_packed";
        }
        if (request.family != "moe") {
            return "rtx3090_moe_1f1b_bf16grad requires family=moe";
        }
        if (request.device.required_arch != "sm_86") {
            return "rtx3090_moe_1f1b_bf16grad requires device.required_arch=sm_86";
        }
        if (request.attention_backend != "scalar_flash") {
            return "rtx3090_moe_1f1b_bf16grad requires attention_backend=scalar_flash";
        }
        if (request.fp8_storage_format != "weights_e4m3" ||
            request.fp8_compute_path != "bf16_dequant" ||
            request.fp8_weight_scale_mode != "per_tensor") {
            return "rtx3090_moe_1f1b_bf16grad requires weights_e4m3/bf16_dequant/per_tensor metadata";
        }
        if (request.gradient_buffer_precision != "bf16") {
            return "rtx3090_moe_1f1b_bf16grad requires gradient_buffer_precision=bf16";
        }
        if (request.memory_profile != "24gb") {
            return "rtx3090_moe_1f1b_bf16grad requires memory_profile=24gb";
        }
        if (request.input.batch_size != 1 || request.input.sequence_length != 2048 ||
            request.training.microbatch != 1 || request.training.grad_accumulation != 1) {
            return "rtx3090_moe_1f1b_bf16grad requires microbatch=1, grad_accumulation=1, sequence_length=2048";
        }
        if (request.device.model_parallel_devices.size() != 2 ||
            request.device.model_parallel_devices[0] != 0 ||
            request.device.model_parallel_devices[1] != 1) {
            return "rtx3090_moe_1f1b_bf16grad requires model_parallel_devices=[0,1]";
        }
        if (request.device.pipeline_split_layer != 4) {
            return "rtx3090_moe_1f1b_bf16grad requires pipeline_split_layer=4";
        }
        if (request.device.peer_transport != "host_staged") {
            return "rtx3090_moe_1f1b_bf16grad requires device.peer_transport=host_staged";
        }
        if (request.arena_reserve_bytes != 0) {
            return "rtx3090_moe_1f1b_bf16grad requires arena_reserve_bytes=0";
        }
        return std::nullopt;
    }
    if (ampere_3090_generic) {
        if (!ampere_profile) {
            return "rtx3090_ampere_packed requires precision_profile=ampere_fp8_packed";
        }
        if (request.family != "edge" && request.family != "ai" &&
            request.family != "swift" && request.family != "moe") {
            return "rtx3090_ampere_packed requires family=edge|ai|swift|moe";
        }
        if (request.device.required_arch != "sm_86") {
            return "rtx3090_ampere_packed requires device.required_arch=sm_86";
        }
        if (request.attention_backend != "scalar_flash") {
            return "rtx3090_ampere_packed requires attention_backend=scalar_flash";
        }
        if (request.fp8_storage_format != "weights_e4m3" ||
            request.fp8_compute_path != "bf16_dequant" ||
            request.fp8_weight_scale_mode != "per_tensor") {
            return "rtx3090_ampere_packed requires weights_e4m3/bf16_dequant/per_tensor metadata";
        }
        if (request.memory_profile != "24gb") {
            return "rtx3090_ampere_packed requires memory_profile=24gb";
        }
        if (request.input.batch_size <= 0 || request.input.sequence_length != 2048 ||
            request.training.microbatch <= 0 || request.training.grad_accumulation <= 0) {
            return "rtx3090_ampere_packed requires positive batch policy and sequence_length=2048";
        }
        if (!request.device.model_parallel_devices.empty()) {
            return "rtx3090_ampere_packed is single-device only";
        }
        if (request.arena_reserve_bytes != 0) {
            return "rtx3090_ampere_packed requires arena_reserve_bytes=0";
        }
        return std::nullopt;
    }
    if (!ampere_profile || !ampere_hardware) {
        return "Ampere requests require hardware_profile=rtx3050_swift, rtx3090_swift, or rtx3050_edge and precision_profile=ampere_fp8_packed";
    }
    const std::string expected_family = ampere_edge ? "edge" : "swift";
    if (request.family != expected_family) {
        return request.hardware_profile + " hardware_profile is restricted to family=" + expected_family;
    }
    if (request.device.required_arch != "sm_86") {
        return "ampere_fp8_packed requires device.required_arch=sm_86";
    }
    if (request.attention_backend != "scalar_flash") {
        return "ampere_fp8_packed requires attention_backend=scalar_flash";
    }
    if (request.fp8_storage_format != "weights_e4m3" ||
        request.fp8_compute_path != "bf16_dequant" ||
        request.fp8_weight_scale_mode != "per_tensor") {
        return "ampere_fp8_packed requires weights_e4m3/bf16_dequant/per_tensor metadata";
    }
    // "4gb_max" is opt-in only (never auto-selected by the nvidia-smi probe in
    // resolve_rtx3050_swift_profile): the nominal "4gb" bucket's mb=4 was a
    // conservative guess that measured at only 515 MiB of a 4096 MiB card
    // (~12.6%) in the validated production run. Real measurement with
    // arena_reserve_bytes=0 (2026-08-23, same card) found Swift's actual
    // per-microbatch VRAM cost fits `419 + 20*mb MiB` across mb=4/64/128/160
    // (515/1699/2979/3651 MiB measured). mb=128 keeps the same effective-batch
    // invariant as the other three buckets (128*32=4096) while landing at
    // ~2979 MiB (73%, ~1.1 GiB of real headroom).
    const bool ampere_3050_max = ampere_3050 && request.memory_profile == "4gb_max";
    const bool bucket_ok = ampere_edge
        ? request.memory_profile == "4gb"
        : ampere_3050
            ? (request.memory_profile == "4gb" || request.memory_profile == "6gb" ||
               request.memory_profile == "8gb" || request.memory_profile == "4gb_max")
            : request.memory_profile == "24gb";
    if (!bucket_ok) {
        return ampere_edge
            ? "rtx3050_edge requires memory_profile=4gb"
            : ampere_3050
                ? "rtx3050_swift requires memory_profile=4gb, 6gb, 8gb, or 4gb_max"
                : "rtx3090_swift requires memory_profile=24gb";
    }
    const int expected_mb = ampere_edge ? 48
        : ampere_3090 ? 64
        : ampere_3050_max ? 128
        : request.memory_profile == "4gb" ? 4
        : request.memory_profile == "6gb" ? 8 : 16;
    const int expected_accum = ampere_edge ? 170
        : ampere_3090 ? 64
        : ampere_3050_max ? 32
        : request.memory_profile == "4gb" ? 1024
        : request.memory_profile == "6gb" ? 512 : 256;
    if (request.input.batch_size != expected_mb || request.training.microbatch != expected_mb) {
        return request.hardware_profile + " microbatch does not match the selected nominal VRAM bucket";
    }
    if (request.training.grad_accumulation != expected_accum) {
        return request.hardware_profile + " gradient_accumulation does not match the selected nominal VRAM bucket";
    }
    const int expected_head_dim = ampere_edge ? 64 : 32;
    if (request.input.sequence_length != 2048 || request.model.hidden_size <= 0 ||
        request.model.heads <= 0 || request.model.hidden_size / request.model.heads != expected_head_dim) {
        return request.hardware_profile + " requires sequence_length=2048 and head_dim=" +
            std::to_string(expected_head_dim);
    }
    if (request.model.num_cognitive_routes > 0 || request.model.num_personality_experts > 0 ||
        !request.device.model_parallel_devices.empty()) {
        return request.hardware_profile + " v1 supports dense single-device Swift/Edge only";
    }
    if (request.arena_reserve_bytes != 0) {
        return request.hardware_profile + " requires arena_reserve_bytes=0";
    }
    return std::nullopt;
}

static std::optional<std::string> validate_ampere_runtime_contract(
    const ida_native::NativeRequest& request,
    const ida_native::NativeArena& arena
) {
    if (request.precision_profile != "ampere_fp8_packed") return std::nullopt;
    cudaDeviceProp prop{};
    IDA_CUDA_CHECK(cudaGetDeviceProperties(&prop, arena.device_id));
    const int actual_arch = prop.major * 10 + prop.minor;
    if (actual_arch != 86) {
        return "ampere_fp8_packed requires exact CUDA compute capability sm_86";
    }
    std::size_t free_bytes = 0, total_bytes = 0;
    IDA_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const double total_gib = static_cast<double>(total_bytes) / (1ull << 30);
    const char* bucket = total_gib < 5.0 ? "4gb" : total_gib < 7.0 ? "6gb"
        : total_gib < 10.0 ? "8gb"
        : (total_gib >= 20.0 && total_gib < 28.0) ? "24gb" : nullptr;
    // "4gb_max" is the opt-in higher-microbatch bucket for the same measured
    // ~4GiB size class as "4gb" -- see the bucket table comment in
    // validate_ampere_fp8_contract above.
    const bool memory_profile_matches = bucket != nullptr && (
        request.memory_profile == bucket ||
        (std::string(bucket) == "4gb" && request.memory_profile == "4gb_max")
    );
    if (!memory_profile_matches) {
        return request.hardware_profile + " memory_profile does not match the measured nominal VRAM bucket";
    }
    const std::size_t default_margin = std::max<std::size_t>(512ull << 20, total_bytes / 8);
    const std::size_t margin = request.min_free_vram_bytes >= 0
        ? static_cast<std::size_t>(request.min_free_vram_bytes) : default_margin;
    if (free_bytes < margin) {
        return request.hardware_profile + " rejected: free VRAM is below the measured working-set margin";
    }
    return std::nullopt;
}

static std::optional<std::string> validate_observability_contract(
    const ida_native::NativeRequest& request
) {
#if !IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
    (void)request;
    return std::nullopt;
#else
    if (!request.ontology_required && !request.analytics_required) {
        return std::nullopt;
    }
    if (request.ontology_required) {
        const char* enabled = std::getenv("IDA_NATIVE_ONTOLOGY");
        const char* configured_path = std::getenv("IDA_NATIVE_ONTOLOGY_PATH");
        if (!enabled || std::string(enabled) != "1") {
            return "ontology_required but IDA_NATIVE_ONTOLOGY is not 1";
        }
        if (!configured_path || !*configured_path || request.ontology_path.empty()) {
            return "ontology_required needs both IDA_NATIVE_ONTOLOGY_PATH and ontology_path";
        }
        const auto configured = std::filesystem::absolute(
            std::filesystem::path(configured_path)).lexically_normal();
        const auto requested = std::filesystem::absolute(request.ontology_path).lexically_normal();
        if (configured != requested) {
            return "ontology_path does not match IDA_NATIVE_ONTOLOGY_PATH";
        }
    }
    if (request.analytics_required) {
        if (request.analytics_path.empty()) {
            return "analytics_required needs analytics_path";
        }
        if (request.analytics_contract != "native_training_metrics_and_evidence_writer") {
            return "analytics_required needs the native analytics contract metadata";
        }
    }
    return std::nullopt;
#endif
}

std::optional<std::string> validate_batch_size_override_contract(
    const ida_native::NativeRequest& request
) {
    // batch_size_override is parsed (request.cpp) but was never read again
    // anywhere in the engine -- the real microbatch is request.input.batch_size,
    // baked into the native_input files at data-prep time (train_student_
    // native.py's own --batch-size-override CLI flag repacks the tensors
    // there; this JSON field is a separate, disconnected copy). A request
    // hand-edited to request one shape while the input files were packed for
    // another silently trained the WRONG shape with no error (found 2026-07-23
    // debugging an mb sweep that produced identical results at every
    // microbatch -- the override was a no-op the whole time). Fail loudly
    // instead of silently training whatever shape the input happens to be.
    if (request.batch_size_override > 0
        && request.input.batch_size > 0
        && request.batch_size_override != request.input.batch_size) {
        return "batch_size_override=" + std::to_string(request.batch_size_override) +
               " does not match the microbatch the input files were packed for "
               "(input.shape[0]=" + std::to_string(request.input.batch_size) +
               ") -- batch_size_override cannot reshape already-packed native_input "
               "files, it must match what actually produced them";
    }
    return std::nullopt;
}

std::optional<std::string> validate_precision_state_contract(
    const ida_native::NativeRequest& request
) {
    const auto require_supported = [](
        const std::string& value,
        const char* field,
        std::initializer_list<const char*> supported
    )
        -> std::optional<std::string> {
        for (const char* candidate : supported) {
            if (value == candidate) {
                return std::nullopt;
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
        return "native precision policy " + std::string(field) + "=" + value +
               " is unsupported; supported values: " + supported_values;
    };
    if (const auto err = require_supported(
            request.optimizer_state_precision, "optimizer_state_precision", {"fp32", "bf16"})) {
        return err;
    }
    if (const auto err = require_supported(
            request.optimizer_type, "optimizer_type", {"lion"})) {
        return err;
    }
    if (const auto err = require_supported(
            request.gradient_buffer_precision, "gradient_buffer_precision", {"fp32", "bf16"})) {
        return err;
    }
    if (const auto err = require_supported(
            request.gemm_accumulator_precision, "gemm_accumulator_precision", {"fp32"})) {
        return err;
    }
    return std::nullopt;
}

bool preserve_failure_phase(const std::filesystem::path& status_file) {
    try {
        std::ifstream input(status_file, std::ios::binary);
        if (!input) {
            return false;
        }
        json payload = json::parse(input);
        const auto phase = payload.value("phase", "");
        return phase == "blocked_attention_backend" ||
               phase == "blocked_native_precision_policy" ||
               phase == "blocked_native_hardware_contract";
    } catch (...) {
        return false;
    }
}

void execute_burn(
    const ida_native::NativeRequest& request,
    ida_native::NativeArena& arena,
    ida_native::NativeArena* second_stage = nullptr
) {
    const bool model_parallel = second_stage != nullptr;
    const auto& mp_devices = request.device.model_parallel_devices;
    if (!mp_devices.empty() && !model_parallel) {
        const std::string error =
            "model-parallel request requires direct two-device execution; "
            "resident single-device workers cannot accept it";
        ida_native::write_status(
            request.status_file, request, "blocked_model_parallel_contract",
            {"\"error\": \"" + ida_native::json_escape(error) + "\"",
             "\"promotion_eligible\": false"}
        );
        throw std::runtime_error(error);
    }
    if (model_parallel && (mp_devices.size() != 2 ||
                           mp_devices[0] != arena.device_id ||
                           mp_devices[1] != second_stage->device_id)) {
        const std::string error =
            "direct two-device arenas do not match model_parallel_devices";
        ida_native::write_status(
            request.status_file, request, "blocked_model_parallel_contract",
            {"\"error\": \"" + ida_native::json_escape(error) + "\"",
             "\"promotion_eligible\": false"}
        );
        throw std::runtime_error(error);
    }
    const std::string device_fields = model_parallel
        ? "\"cuda_devices\": [" + std::to_string(arena.device_id) + ", " +
              std::to_string(second_stage->device_id) + "], \"model_parallel\": true, \"peer_transport\": \"" +
              ida_native::json_escape(request.device.peer_transport) + "\""
        : "\"cuda_device\": " + std::to_string(arena.device_id) +
              ", \"model_parallel\": false, \"peer_transport\": \"none\"";
    ida_native::write_status(request.status_file, request, "setup");
    if (const auto attention_error = validate_attention_contract(request)) {
        ida_native::write_status(
            request.status_file,
            request,
            "blocked_attention_backend",
            {
                "\"error\": \"" + ida_native::json_escape(*attention_error) + "\"",
                "\"promotion_eligible\": false"
            }
        );
        throw std::runtime_error(*attention_error);
    }
    if (const auto ampere_error = validate_ampere_fp8_contract(request)) {
        ida_native::write_status(
            request.status_file, request, "blocked_native_hardware_contract",
            {"\"error\": \"" + ida_native::json_escape(*ampere_error) + "\"",
             "\"promotion_eligible\": false"}
        );
        throw std::runtime_error(*ampere_error);
    }
    if (const auto ampere_runtime_error = validate_ampere_runtime_contract(request, arena)) {
        ida_native::write_status(
            request.status_file, request, "blocked_native_hardware_contract",
            {"\"error\": \"" + ida_native::json_escape(*ampere_runtime_error) + "\"",
             "\"promotion_eligible\": false"}
        );
        throw std::runtime_error(*ampere_runtime_error);
    }
    if (const auto observability_error = validate_observability_contract(request)) {
        ida_native::write_status(
            request.status_file, request, "blocked_native_observability_contract",
            {"\"error\": \"" + ida_native::json_escape(*observability_error) + "\"",
             "\"promotion_eligible\": false"}
        );
        throw std::runtime_error(*observability_error);
    }
    if (const auto batch_size_error = validate_batch_size_override_contract(request)) {
        ida_native::write_status(
            request.status_file,
            request,
            "blocked_batch_size_override_mismatch",
            {
                "\"error\": \"" + ida_native::json_escape(*batch_size_error) + "\"",
                "\"promotion_eligible\": false"
            }
        );
        throw std::runtime_error(*batch_size_error);
    }
    if (const auto precision_error = validate_precision_state_contract(request)) {
        ida_native::write_status(
            request.status_file,
            request,
            "blocked_native_precision_policy",
            {
                "\"error\": \"" + ida_native::json_escape(*precision_error) + "\"",
                "\"promotion_eligible\": false"
            }
        );
        throw std::runtime_error(*precision_error);
    }
    ida_native::write_status(request.status_file, request, "validation_complete");
    ida_native::write_status(
        request.status_file,
        request,
        "native_input_ready",
        {
            "\"native_input_mode\": \"request_paths_only\"",
            "\"promotion_eligible\": false"
        }
    );
    ida_native::write_status(request.status_file, request, "model_building");
    ida_native::write_status(
        request.status_file,
        request,
        "native_allocator_ready",
        {
            device_fields,
            "\"promotion_eligible\": false"
        }
    );

    const auto report_progress = [&](const ida_native::ProgressReport& r) {
        const double tps = static_cast<double>(r.tokens) / std::max(r.elapsed_s, 0.000001);
        ida_native::write_status(
            request.status_file,
            request,
            "training_in_progress",
            {
                "\"job_id\": \"" + request.job_id + "\"",
                "\"global_step\": " + std::to_string(r.micro_step),
                "\"optimizer_step\": " + std::to_string(r.optimizer_step),
                "\"active_grad_accum\": " + std::to_string(r.active_grad_accum),
                "\"requested_grad_accum\": " + std::to_string(r.requested_grad_accum),
                "\"tokens_processed\": " + std::to_string(r.tokens),
                "\"tokens_per_second\": " + ida_native::json_number(tps),
                "\"elapsed_seconds\": " + ida_native::json_number(r.elapsed_s),
                "\"loss\": " + ida_native::json_number(r.loss),
                "\"loss_ema\": " + ida_native::json_number(r.loss_ema),
                "\"grad_norm\": " + ida_native::json_number(r.grad_norm),
                "\"learning_rate\": " + ida_native::json_number(r.lr),
                "\"effective_batch\": " + std::to_string(r.effective_batch),
                "\"skipped_steps\": " + std::to_string(r.skipped_steps),
                "\"global_grad_clip_steps\": " + std::to_string(r.global_grad_clip_steps),
                "\"embed_row_clip_steps\": " + std::to_string(r.embed_row_clip_steps),
                "\"embed_row_clipped_rows_total\": " + std::to_string(r.embed_row_clipped_rows_total),
                std::string("\"global_grad_clip_fired\": ") + (r.global_grad_clip_fired ? "true" : "false"),
                "\"global_grad_clip_scale\": " + ida_native::json_number(r.global_grad_clip_scale),
                std::string("\"embed_row_clip_fired\": ") + (r.embed_row_clip_fired ? "true" : "false"),
                "\"embed_row_clip_threshold\": " + ida_native::json_number(r.embed_row_clip_threshold),
                "\"embed_row_clip_max_preclip_norm\": " + ida_native::json_number(r.embed_row_clip_max_preclip_norm),
                "\"embed_row_clipped_rows\": " + std::to_string(r.embed_row_clipped_rows),
                "\"dominant_grad_slot\": \"" + ida_native::json_escape(r.dominant_grad_slot) + "\"",
                "\"dominant_grad_slot_norm\": " + ida_native::json_number(r.dominant_grad_slot_norm),
                "\"dominant_grad_slot_frac\": " + ida_native::json_number(r.dominant_grad_slot_frac),
                "\"lss_feedback_skip_tail\": " + std::to_string(r.lss_feedback_skip_tail),
                "\"lss_feedback_residual_scale\": " + ida_native::json_number(r.lss_feedback_residual_scale),
                "\"lss_aux\": " + ida_native::json_number(r.lss_aux),
                "\"pss_slot_clip_steps\": " + std::to_string(r.pss_slot_clip_steps),
                "\"pss_spike_ratio_max\": " + ida_native::json_number(r.pss_spike_ratio_max),
                "\"pss_spike_slot\": \"" + ida_native::json_escape(r.pss_spike_slot) + "\"",
                "\"pss_confidence\": " + ida_native::json_number(r.pss_confidence),
                "\"pss_pred_err\": " + ida_native::json_number(r.pss_pred_err),
                "\"pss_int2_agreement\": " + ida_native::json_number(r.pss_int2_agreement),
                "\"pss_engaged_frac\": " + ida_native::json_number(r.pss_engaged_frac),
                "\"pss_covered\": " + std::to_string(r.pss_covered),
                "\"pss_n\": " + std::to_string(r.pss_n),
                "\"pss_int2_matched\": " + std::to_string(r.pss_int2_matched),
                "\"pss_int2_scored\": " + std::to_string(r.pss_int2_scored),
                "\"pss_scored_micros\": " + std::to_string(r.pss_scored_micros),
                "\"pss_int2_inv_rms\": " + ida_native::json_number(r.pss_int2_inv_rms),
                "\"pss_confidence_min\": " + ida_native::json_number(r.pss_confidence_min),
                "\"pss_confidence_max\": " + ida_native::json_number(r.pss_confidence_max),
                "\"pss_blend_delta_rms\": " + ida_native::json_number(r.pss_blend_delta_rms),
                "\"pss_aux_weight\": " + ida_native::json_number(r.pss_aux_weight),
                "\"pss_aux_denom\": " + ida_native::json_number(r.pss_aux_denom),
                std::string("\"pss_aux_normalize\": ") + (r.pss_aux_normalize ? "true" : "false"),
                std::string("\"pss_conditioning_active\": ") + (r.pss_conditioning_active ? "true" : "false"),
                std::string("\"pss_override_active\": ") + (r.pss_override_active ? "true" : "false"),
                "\"pss_effective_confidence\": " + ida_native::json_number(r.pss_effective_confidence),
                "\"pss_governor_event\": \"" + ida_native::json_escape(r.pss_governor_event) + "\"",
                "\"pss_conditioning_mode\": \"" + ida_native::json_escape(r.pss_conditioning_mode) + "\"",
                "\"pss_aux_normalize_mode\": \"" + ida_native::json_escape(r.pss_aux_normalize_mode) + "\"",
                std::string("\"fp8_active\": ") + (r.fp8_active ? "true" : "false"),
                device_fields,
                "\"current_phase\": \"training\"",
                "\"last_heartbeat_at\": \"" + utc_timestamp_now() + "\"",
                "\"promotion_eligible\": false"
            }
        );
    };
    const auto result = model_parallel
        ? ida_native::run_lattice_training_model_parallel(request, arena, *second_stage, report_progress)
        : ida_native::run_smoke_training(request, arena, report_progress);

    ida_native::write_status(
        request.status_file,
        request,
        "training",
        {
            "\"job_id\": \"" + request.job_id + "\"",
            "\"global_step\": " + std::to_string(result.global_step),
            "\"tokens_processed\": " + std::to_string(result.tokens_processed),
            "\"tokens_per_second\": " + ida_native::json_number(result.tokens_per_second),
            device_fields,
            "\"promotion_eligible\": false"
        }
    );

    const std::string result_device_fields = result.model_parallel
        ? "\"cuda_devices\": [" + std::to_string(result.pipeline_first_device) + ", " +
              std::to_string(result.pipeline_second_device) + "], \"model_parallel\": true, \"peer_transport\": \"" +
              ida_native::json_escape(result.peer_transport) + "\""
        : device_fields;
    if (result.model_parallel) {
        const bool merged_weights =
            std::filesystem::is_regular_file(request.output_dir / "model.safetensors");
        ida_native::write_status(
            request.status_file,
            request,
            "checkpoint_write",
            {
                "\"job_id\": \"" + request.job_id + "\"",
                "\"global_step\": " + std::to_string(result.global_step),
                "\"checkpoint_resumable\": false",
                "\"checkpoint_contract\": \"merged_weights_only\"",
                "\"pipeline_split_layer\": " + std::to_string(result.pipeline_split_layer),
                "\"peer_forward_bytes\": " + std::to_string(result.peer_forward_bytes),
                "\"peer_backward_bytes\": " + std::to_string(result.peer_backward_bytes),
                result_device_fields,
                std::string("\"promotion_eligible\": ") +
                    (merged_weights ? "true" : "false")
            }
        );
        if (!merged_weights) {
            throw std::runtime_error(
                "model-parallel run completed without merged model.safetensors");
        }
        ida_native::write_checkpoint_artifacts(request, result);
        ida_native::write_final_artifacts(request, result);
        ida_native::write_training_contract_files(request, result);
    } else {
        ida_native::write_status(
            request.status_file,
            request,
            "checkpoint_write",
            {
                "\"job_id\": \"" + request.job_id + "\"",
                "\"global_step\": " + std::to_string(result.global_step),
                "\"promotion_eligible\": false"
            }
        );
        ida_native::write_checkpoint_artifacts(request, result);
        ida_native::write_final_artifacts(request, result);
        ida_native::write_training_contract_files(request, result);
    }
    // The ontology-derived 3090 1F1B/BF16-gradient profile is an experiment,
    // not a promotion path.  A model artifact alone is insufficient evidence
    // for this profile because it is not exact-resume capable and still needs
    // the BF16-versus-packed numerical canary decision.
    const bool experimental_moe_1f1b =
        request.hardware_profile == "rtx3090_moe_1f1b_bf16grad";
    const bool promotion_eligible = !experimental_moe_1f1b &&
        std::filesystem::is_regular_file(request.output_dir / "model.safetensors");
    ida_native::write_status(
        request.status_file,
        request,
        request.expected_terminal_phase,
        {
            "\"job_id\": \"" + request.job_id + "\"",
            "\"global_step\": " + std::to_string(result.global_step),
            "\"tokens_processed\": " + std::to_string(result.tokens_processed),
            "\"tokens_per_second\": " + ida_native::json_number(result.tokens_per_second),
            "\"elapsed_seconds\": " + ida_native::json_number(result.elapsed_seconds),
            "\"final_loss\": " + ida_native::json_number(result.final_loss),
            "\"final_loss_ema\": " + ida_native::json_number(result.final_loss_ema),
            "\"final_grad_norm\": " + ida_native::json_number(result.final_grad_norm),
            "\"final_lr\": " + ida_native::json_number(result.final_lr),
            "\"optimizer_steps\": " + std::to_string(result.optimizer_steps),
            "\"skipped_steps\": " + std::to_string(result.skipped_steps),
            "\"global_grad_clip_steps\": " + std::to_string(result.global_grad_clip_steps),
            "\"embed_row_clip_steps\": " + std::to_string(result.embed_row_clip_steps),
            "\"embed_row_clipped_rows_total\": " + std::to_string(result.embed_row_clipped_rows_total),
            std::string("\"final_global_grad_clip_fired\": ") + (result.final_global_grad_clip_fired ? "true" : "false"),
            "\"final_global_grad_clip_scale\": " + ida_native::json_number(result.final_global_grad_clip_scale),
            std::string("\"final_embed_row_clip_fired\": ") + (result.final_embed_row_clip_fired ? "true" : "false"),
            "\"final_embed_row_clip_threshold\": " + ida_native::json_number(result.final_embed_row_clip_threshold),
            "\"final_embed_row_clip_max_preclip_norm\": " + ida_native::json_number(result.final_embed_row_clip_max_preclip_norm),
            "\"final_embed_row_clipped_rows\": " + std::to_string(result.final_embed_row_clipped_rows),
            "\"final_dominant_grad_slot\": \"" + ida_native::json_escape(result.final_dominant_grad_slot) + "\"",
            "\"final_dominant_grad_slot_norm\": " + ida_native::json_number(result.final_dominant_grad_slot_norm),
            "\"final_dominant_grad_slot_frac\": " + ida_native::json_number(result.final_dominant_grad_slot_frac),
            "\"final_pss_predictor_grad_norm\": " + ida_native::json_number(result.final_pss_predictor_grad_norm),
            "\"final_pss_predictor_grad_frac\": " + ida_native::json_number(result.final_pss_predictor_grad_frac),
            "\"final_lss_feedback_skip_tail\": " + std::to_string(result.final_lss_feedback_skip_tail),
            "\"final_lss_feedback_residual_scale\": " + ida_native::json_number(result.final_lss_feedback_residual_scale),
            "\"final_lss_aux\": " + ida_native::json_number(result.final_lss_aux),
            "\"final_pss_slot_clip_steps\": " + std::to_string(result.final_pss_slot_clip_steps),
            "\"final_pss_spike_ratio_max\": " + ida_native::json_number(result.final_pss_spike_ratio_max),
            "\"final_pss_spike_slot\": \"" + ida_native::json_escape(result.final_pss_spike_slot) + "\"",
            "\"final_pss_confidence\": " + ida_native::json_number(result.final_pss_confidence),
            "\"final_pss_pred_err\": " + ida_native::json_number(result.final_pss_pred_err),
            "\"final_pss_int2_agreement\": " + ida_native::json_number(result.final_pss_int2_agreement),
            "\"final_pss_engaged_frac\": " + ida_native::json_number(result.final_pss_engaged_frac),
            "\"final_pss_covered\": " + std::to_string(result.final_pss_covered),
            "\"final_pss_n\": " + std::to_string(result.final_pss_n),
            "\"final_pss_int2_matched\": " + std::to_string(result.final_pss_int2_matched),
            "\"final_pss_int2_scored\": " + std::to_string(result.final_pss_int2_scored),
            "\"final_pss_scored_micros\": " + std::to_string(result.final_pss_scored_micros),
            "\"final_pss_int2_inv_rms\": " + ida_native::json_number(result.final_pss_int2_inv_rms),
            "\"final_pss_confidence_min\": " + ida_native::json_number(result.final_pss_confidence_min),
            "\"final_pss_confidence_max\": " + ida_native::json_number(result.final_pss_confidence_max),
            "\"final_pss_blend_delta_rms\": " + ida_native::json_number(result.final_pss_blend_delta_rms),
            "\"final_pss_aux_weight\": " + ida_native::json_number(result.final_pss_aux_weight),
            "\"final_pss_aux_denom\": " + ida_native::json_number(result.final_pss_aux_denom),
            std::string("\"final_pss_aux_normalize\": ") + (result.final_pss_aux_normalize ? "true" : "false"),
            std::string("\"final_pss_conditioning_active\": ") + (result.final_pss_conditioning_active ? "true" : "false"),
            std::string("\"final_pss_override_active\": ") + (result.final_pss_override_active ? "true" : "false"),
            "\"final_pss_effective_confidence\": " + ida_native::json_number(result.final_pss_effective_confidence),
            "\"final_pss_governor_event\": \"" + ida_native::json_escape(result.final_pss_governor_event) + "\"",
            "\"final_pss_conditioning_mode\": \"" + ida_native::json_escape(result.final_pss_conditioning_mode) + "\"",
            "\"final_pss_aux_normalize_mode\": \"" + ida_native::json_escape(result.final_pss_aux_normalize_mode) + "\"",
            std::string("\"fp8_active\": ") + (result.fp8_active ? "true" : "false"),
            "\"device_bytes_touched\": " + std::to_string(result.device_bytes_touched),
            "\"attn_p_zero_frac\": " + ida_native::json_number(result.attn_p_zero_frac),
            "\"attn_lse_drift_max\": " + ida_native::json_number(result.attn_lse_drift_max),
            "\"attn_p_zero_frac_stage0\": " + ida_native::json_number(result.attn_p_zero_frac_stage0),
            "\"attn_p_zero_frac_stage1\": " + ida_native::json_number(result.attn_p_zero_frac_stage1),
            "\"attn_lse_drift_max_stage0\": " + ida_native::json_number(result.attn_lse_drift_max_stage0),
            "\"attn_lse_drift_max_stage1\": " + ida_native::json_number(result.attn_lse_drift_max_stage1),
            std::string("\"attn_health_aggregation\": \"") +
                (result.model_parallel ? "max_stage" : "single_stage") + "\"",
            std::string("\"model_parallel\": ") + (result.model_parallel ? "true" : "false"),
            "\"pipeline_stage_count\": " + std::to_string(result.pipeline_stage_count),
            "\"pipeline_split_layer\": " + std::to_string(result.pipeline_split_layer),
            "\"peer_transport\": \"" + ida_native::json_escape(result.peer_transport) + "\"",
            "\"peer_forward_bytes\": " + std::to_string(result.peer_forward_bytes),
            "\"peer_backward_bytes\": " + std::to_string(result.peer_backward_bytes),
            std::string("\"checkpoint_resumable\": ") + (result.model_parallel ? "false" : "true"),
            result_device_fields,
            std::string("\"promotion_eligible\": ") + (promotion_eligible ? "true" : "false")
        }
    );
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        ida_native::print_cli_welcome(std::cerr);
        if (args.self_check) {
            run_self_check(args);
            return 0;
        }
        if (!args.request_json.has_value()) {
            throw std::runtime_error("usage: ida_native_train --request-json <path>");
        }

        const auto request = ida_native::load_request(*args.request_json);
        const bool model_parallel = !request.device.model_parallel_devices.empty();
        const int first_device = model_parallel
            ? request.device.model_parallel_devices.at(0) : args.device_id;
        maybe_enable_host_mapped_memory(first_device);
        auto arena = ida_native::create_arena(first_device, request.arena_reserve_bytes);
        (void)ida_native::runtime_device_info(first_device);
        std::optional<ida_native::NativeArena> second_stage;
        if (model_parallel) {
            if (request.device.model_parallel_devices.size() != 2) {
                throw std::runtime_error(
                    "model_parallel_devices must contain exactly two device ordinals");
            }
            const int second_device = request.device.model_parallel_devices.at(1);
            maybe_enable_host_mapped_memory(second_device);
            second_stage = ida_native::create_arena(
                second_device, request.arena_reserve_bytes);
            (void)ida_native::runtime_device_info(second_device);
        }
        try {
            execute_burn(request, arena, second_stage ? &*second_stage : nullptr);
        } catch (const std::exception& exc) {
            // One-shot mode has no worker loop to report through: without a
            // terminal failure phase in the status file, the Python client
            // waits on it forever while this process exits.
            try {
                if (!preserve_failure_phase(request.status_file)) {
                    ida_native::write_status(
                        request.status_file, request,
                        "native_worker_command_failed",
                        {
                            "\"error\": \"" +
                                ida_native::json_escape(exc.what()) + "\"",
                            "\"promotion_eligible\": false"
                        }
                    );
                }
            } catch (...) {}
            throw;
        }
        if (second_stage) {
            ida_native::destroy_arena(*second_stage);
        }
        ida_native::destroy_arena(arena);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "[ida_native_train] " << exc.what() << '\n';
        return 1;
    }
}
