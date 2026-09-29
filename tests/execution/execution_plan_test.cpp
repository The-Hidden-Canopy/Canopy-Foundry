#include "canopy_foundry/execution/plan.hpp"

#include <iostream>
#include <string>

namespace {

ida_native::NativeRequest baseline_request() {
    ida_native::NativeRequest request{};
    request.backend = "native";
    request.device.runtime = "cuda";
    request.device.required_arch = "sm_90";
    request.device.peer_transport = "auto";
    request.attention_backend = "scalar_flash";
    request.precision_profile = "legacy_bf16";
    request.optimizer_type = "lion";
    request.optimizer_state_precision = "fp32";
    request.gradient_buffer_precision = "fp32";
    request.gemm_accumulator_precision = "fp32";
    request.family = "edge";
    request.model.architecture_contract = "ida_lattice_native_v1";
    request.model.hidden_size = 128;
    request.model.intermediate_size = 512;
    request.model.layers = 2;
    request.model.heads = 4;
    request.model.kv_heads = 4;
    request.model.vocab_size = 1024;
    return request;
}

bool require(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

}  // namespace

int main() {
    using namespace canopy_foundry;
    const auto capabilities = RuntimeCapabilities::from_build(true, false, true);
    const auto resolution = resolve_execution_plan(baseline_request(), capabilities);
    const auto& plan = resolution.plan;

    if (!require(plan.accepted, "baseline CUDA request was rejected")) return 1;
    if (!require(plan.runtime_verification_required,
                 "baseline plan did not retain runtime verification")) return 1;
    if (!require(plan.precision.master_storage == "bf16",
                 "baseline master storage was not resolved")) return 1;
    if (!require(plan.device.parallel_mode == "single_device",
                 "baseline device plan changed parallel mode")) return 1;

    auto fp8 = baseline_request();
    fp8.precision_profile = "legacy_fp8";
    const auto fp8_plan = resolve_execution_plan(fp8, capabilities).plan;
    if (!require(fp8_plan.accepted,
                 "existing legacy FP8 request was rejected")) return 1;
    if (!require(fp8_plan.precision.forward_execution == "legacy_fp8",
                 "legacy FP8 execution profile was not preserved")) return 1;

    auto ampere = baseline_request();
    ampere.precision_profile = "ampere_fp8_packed";
    ampere.hardware_profile = "rtx3090_swift";
    const auto ampere_plan = resolve_execution_plan(ampere, capabilities).plan;
    if (!require(ampere_plan.accepted,
                 "existing Ampere packed request was rejected")) return 1;
    if (!require(ampere_plan.runtime_verification_required,
                 "Ampere plan did not retain runtime verification")) return 1;

    auto pipeline = baseline_request();
    pipeline.device.model_parallel_devices = {0, 1};
    pipeline.device.peer_transport = "host_staged";
    const auto pipeline_plan = resolve_execution_plan(pipeline, capabilities).plan;
    if (!require(pipeline_plan.accepted,
                 "existing two-device host-staged request was rejected")) return 1;
    if (!require(pipeline_plan.device.parallel_mode == "layer_pipeline" &&
                     pipeline_plan.device.transport == "host_staged",
                 "pipeline transport was not preserved")) return 1;

    auto empty_attention = baseline_request();
    empty_attention.attention_backend.clear();
    const auto empty_attention_plan =
        resolve_execution_plan(empty_attention, capabilities).plan;
    if (!require(empty_attention_plan.accepted,
                 "legacy empty attention alias was rejected")) return 1;

    const auto hash = execution_plan_sha256(plan);
    if (!require(hash.size() == 64, "execution plan hash is not SHA-256 sized")) return 1;
    if (!require(hash == execution_plan_sha256(plan),
                 "execution plan hash is not deterministic")) return 1;
    if (!require(sha256_hex("abc") ==
                     "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                 "SHA-256 implementation failed known vector")) return 1;

    auto invalid = baseline_request();
    invalid.precision_profile = "nvint2";
    const auto invalid_plan = resolve_execution_plan(invalid, capabilities).plan;
    if (!require(!invalid_plan.accepted,
                 "experimental precision was admitted")) return 1;

    invalid = baseline_request();
    invalid.device.model_parallel_devices = {0, 0};
    const auto duplicate_devices = resolve_execution_plan(invalid, capabilities).plan;
    if (!require(!duplicate_devices.accepted,
                 "duplicate pipeline devices were admitted")) return 1;

    invalid = baseline_request();
    invalid.device.runtime = "cpu";
    invalid.device.required_arch = "host";
    invalid.precision_profile = "fp32";
    const auto cpu_plan = resolve_execution_plan(invalid, capabilities).plan;
    if (!require(!cpu_plan.accepted,
                 "disabled CPU Lion path was presented as runnable")) return 1;

    std::cout << "execution plan contract checks passed\n";
    return 0;
}
