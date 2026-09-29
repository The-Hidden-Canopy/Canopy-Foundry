#include "canopy_foundry/execution/plan.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

namespace canopy_foundry {

namespace {

using json = nlohmann::json;

constexpr std::array<std::uint32_t, 64> kSha256Constants{
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

constexpr std::uint32_t rotate_right(std::uint32_t value, unsigned amount) {
    return (value >> amount) | (value << (32u - amount));
}

class Sha256 {
public:
    Sha256()
        : state_{
              0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
              0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
          } {}

    void update(std::string_view value) {
        for (const unsigned char byte : value) {
            block_[block_size_++] = byte;
            if (block_size_ == block_.size()) {
                transform();
                bit_count_ += 512;
                block_size_ = 0;
            }
        }
    }

    std::string finish() {
        bit_count_ += static_cast<std::uint64_t>(block_size_) * 8u;
        block_[block_size_++] = 0x80u;
        if (block_size_ > 56) {
            while (block_size_ < block_.size()) block_[block_size_++] = 0;
            transform();
            block_size_ = 0;
        }
        while (block_size_ < 56) block_[block_size_++] = 0;
        for (int shift = 56; shift >= 0; shift -= 8) {
            block_[block_size_++] = static_cast<unsigned char>(bit_count_ >> shift);
        }
        transform();

        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (const std::uint32_t word : state_) output << std::setw(8) << word;
        return output.str();
    }

private:
    void transform() {
        std::array<std::uint32_t, 64> schedule{};
        for (std::size_t index = 0; index < 16; ++index) {
            const std::size_t offset = index * 4;
            schedule[index] =
                (static_cast<std::uint32_t>(block_[offset]) << 24) |
                (static_cast<std::uint32_t>(block_[offset + 1]) << 16) |
                (static_cast<std::uint32_t>(block_[offset + 2]) << 8) |
                static_cast<std::uint32_t>(block_[offset + 3]);
        }
        for (std::size_t index = 16; index < schedule.size(); ++index) {
            const std::uint32_t s0 =
                rotate_right(schedule[index - 15], 7) ^
                rotate_right(schedule[index - 15], 18) ^
                (schedule[index - 15] >> 3);
            const std::uint32_t s1 =
                rotate_right(schedule[index - 2], 17) ^
                rotate_right(schedule[index - 2], 19) ^
                (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16] + s0 + schedule[index - 7] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t index = 0; index < schedule.size(); ++index) {
            const std::uint32_t s1 =
                rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const std::uint32_t choose = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + s1 + choose +
                kSha256Constants[index] + schedule[index];
            const std::uint32_t s0 =
                rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    static constexpr std::size_t block_size = 64;
    std::array<unsigned char, block_size> block_{};
    std::array<std::uint32_t, 8> state_{};
    std::size_t block_size_{0};
    std::uint64_t bit_count_{0};
};

json capability_json(const CapabilityStatus& capability) {
    return {
        {"id", capability.id},
        {"state", capability_state_name(capability.state)},
        {"evidence", capability.evidence},
    };
}

json plan_json_value(const ExecutionPlan& plan) {
    return {
        {"schema", plan.schema},
        {"planner_revision", plan.planner_revision},
        {"device", {
            {"runtime", plan.device.runtime},
            {"backend", plan.device.backend},
            {"required_arch", plan.device.required_arch},
            {"devices", plan.device.devices},
            {"parallel_mode", plan.device.parallel_mode},
            {"transport", plan.device.transport},
            {"pipeline_split_layer", plan.device.pipeline_split_layer},
        }},
        {"model", {
            {"architecture_contract", plan.model.architecture_contract},
            {"family", plan.model.family},
            {"hidden_size", plan.model.hidden_size},
            {"intermediate_size", plan.model.intermediate_size},
            {"layers", plan.model.layers},
            {"heads", plan.model.heads},
            {"kv_heads", plan.model.kv_heads},
            {"vocab_size", plan.model.vocab_size},
            {"moe", plan.model.moe},
        }},
        {"precision", {
            {"profile", plan.precision.profile},
            {"master_storage", plan.precision.master_storage},
            {"forward_execution", plan.precision.forward_execution},
            {"backward_dx_execution", plan.precision.backward_dx_execution},
            {"gradients", plan.precision.gradients},
            {"accumulators", plan.precision.accumulators},
            {"optimizer_state", plan.precision.optimizer_state},
            {"normalization", plan.precision.normalization},
        }},
        {"optimizer", {
            {"type", plan.optimizer.type},
            {"state_precision", plan.optimizer.state_precision},
        }},
        {"attention_backend", plan.attention_backend},
        {"required_capabilities", plan.required_capabilities},
        {"admission", {
            {"accepted", plan.accepted},
            {"runtime_verification_required", plan.runtime_verification_required},
            {"rejection", plan.rejection.empty() ? json(nullptr) : json(plan.rejection)},
        }},
    };
}

void add_capability(
    RuntimeCapabilities& capabilities,
    std::string id,
    CapabilityState state,
    std::string evidence
) {
    capabilities.entries.push_back({std::move(id), state, std::move(evidence)});
}

}  // namespace

std::string capability_state_name(CapabilityState state) {
    switch (state) {
    case CapabilityState::Available: return "AVAILABLE";
    case CapabilityState::Unavailable: return "UNAVAILABLE";
    case CapabilityState::Experimental: return "EXPERIMENTAL";
    case CapabilityState::BuildDisabled: return "BUILD_DISABLED";
    case CapabilityState::HardwareUnsupported: return "HARDWARE_UNSUPPORTED";
    case CapabilityState::RuntimeUnverified: return "RUNTIME_UNVERIFIED";
    }
    return "UNAVAILABLE";
}

RuntimeCapabilities RuntimeCapabilities::from_build(
    bool cuda_enabled,
    bool opencl_enabled,
    bool cpu_enabled
) {
    RuntimeCapabilities capabilities{};
    add_capability(
        capabilities, "backend.cuda",
        cuda_enabled ? CapabilityState::Available : CapabilityState::BuildDisabled,
        cuda_enabled ? "CUDA target is present; device and architecture are not probed by this inventory"
                      : "CUDA target is disabled or unavailable in this build");
    add_capability(
        capabilities, "backend.opencl",
        opencl_enabled ? CapabilityState::Available : CapabilityState::BuildDisabled,
        opencl_enabled ? "OpenCL target is present; device topology is not probed by this inventory"
                       : "OpenCL target is disabled or unavailable in this build");
    add_capability(
        capabilities, "backend.cpu",
        cpu_enabled ? CapabilityState::Available : CapabilityState::BuildDisabled,
        cpu_enabled ? "CPU target is present; runtime execution policy remains separately admitted"
                    : "CPU target is disabled in this build");
    add_capability(
        capabilities, "attention.scalar",
        CapabilityState::Available,
        "baseline scalar attention contract");
    add_capability(
        capabilities, "optimizer.lion",
        CapabilityState::Available,
        "request contract admits Lion; execution parity remains a separate gate");
    add_capability(
        capabilities, "parallel.cuda_p2p",
        cuda_enabled ? CapabilityState::RuntimeUnverified : CapabilityState::BuildDisabled,
        cuda_enabled ? "requires runtime topology probe" : "requires a CUDA target");
    add_capability(
        capabilities, "parallel.host_staged",
        cuda_enabled ? CapabilityState::Available : CapabilityState::BuildDisabled,
        cuda_enabled ? "host-staged transport is a distinct CUDA pipeline mode"
                      : "requires a CUDA target");
    add_capability(
        capabilities, "precision.bf16",
        CapabilityState::Available,
        "baseline request representation");
    add_capability(
        capabilities, "precision.fp8",
        cuda_enabled ? CapabilityState::RuntimeUnverified : CapabilityState::BuildDisabled,
        cuda_enabled ? "profile-specific hardware evidence is required"
                      : "requires a CUDA target");
    add_capability(
        capabilities, "precision.nvfp4",
        CapabilityState::Experimental,
        "extension point only; not admitted by the planner foundation");
    add_capability(
        capabilities, "precision.mxfp2",
        CapabilityState::Experimental,
        "probe/research state; not a supported execution path");
    add_capability(
        capabilities, "precision.nvint2",
        CapabilityState::Experimental,
        "probe/research state; not a supported execution path");
    return capabilities;
}

const CapabilityStatus* RuntimeCapabilities::find(std::string_view id) const noexcept {
    const auto found = std::find_if(
        entries.begin(), entries.end(),
        [id](const CapabilityStatus& entry) { return entry.id == id; });
    return found == entries.end() ? nullptr : &*found;
}

CapabilityState RuntimeCapabilities::state(std::string_view id) const noexcept {
    const auto* capability = find(id);
    return capability == nullptr ? CapabilityState::Unavailable : capability->state;
}

std::string sha256_hex(std::string_view value) {
    Sha256 digest;
    digest.update(value);
    return digest.finish();
}

PlanResolution resolve_execution_plan(
    const ida_native::NativeRequest& request,
    RuntimeCapabilities capabilities
) {
    PlanResolution resolution{};
    resolution.capabilities = std::move(capabilities);
    auto& plan = resolution.plan;

    plan.device.runtime = request.device.runtime;
    plan.device.backend = request.device.runtime;
    plan.device.required_arch = request.device.required_arch;
    plan.device.devices = request.device.model_parallel_devices;
    plan.device.pipeline_split_layer = request.device.pipeline_split_layer;
    plan.model.architecture_contract = request.model.architecture_contract;
    plan.model.family = request.family;
    plan.model.hidden_size = request.model.hidden_size;
    plan.model.intermediate_size = request.model.intermediate_size;
    plan.model.layers = request.model.layers;
    plan.model.heads = request.model.heads;
    plan.model.kv_heads = request.model.kv_heads;
    plan.model.vocab_size = request.model.vocab_size;
    plan.model.moe = request.model.num_cognitive_routes > 0 ||
        request.model.num_personality_experts > 0 ||
        request.model.generic_moe_num_experts > 0;
    plan.precision.profile = request.precision_profile;
    plan.precision.gradients = request.gradient_buffer_precision;
    plan.precision.accumulators = request.gemm_accumulator_precision;
    plan.precision.optimizer_state = request.optimizer_state_precision;
    plan.precision.normalization = "legacy_request_default";
    plan.optimizer.type = request.optimizer_type;
    plan.optimizer.state_precision = request.optimizer_state_precision;
    plan.attention_backend = request.attention_backend.empty()
        ? "scalar_flash"
        : request.attention_backend;

    const auto reject = [&plan](std::string message) {
        if (plan.rejection.empty()) plan.rejection = std::move(message);
    };
    const auto require = [&plan, &resolution, &reject](std::string id) {
        plan.required_capabilities.push_back(id);
        const auto state = resolution.capabilities.state(id);
        if (state == CapabilityState::BuildDisabled ||
            state == CapabilityState::Unavailable ||
            state == CapabilityState::HardwareUnsupported) {
            reject("required capability is unavailable: " + id);
        }
        return state;
    };

    const auto backend_state = require("backend." + request.device.runtime);
    if (request.device.runtime != "cuda" && request.device.runtime != "opencl" &&
        request.device.runtime != "cpu") {
        reject("runtime is not recognized: " + request.device.runtime);
    }
    if (backend_state == CapabilityState::RuntimeUnverified) {
        plan.runtime_verification_required = true;
    }
    if (request.device.runtime == "cuda" || request.device.runtime == "opencl") {
        // The foundation inventory describes build capabilities only. A
        // device ordinal, architecture, and (for CUDA) topology probe still
        // belong to the runtime admission step.
        plan.runtime_verification_required = true;
    }

    require("attention.scalar");
    if (!request.attention_backend.empty() &&
        request.attention_backend != "scalar_flash") {
        reject("planner foundation admits only attention_backend=scalar_flash");
    }

    const auto optimizer_state = require("optimizer.lion");
    if (request.optimizer_type != "lion") {
        reject("only optimizer_type=lion is admitted");
    }
    if (optimizer_state == CapabilityState::RuntimeUnverified) {
        plan.runtime_verification_required = true;
    }

    if (request.device.runtime == "cuda") {
        if (request.precision_profile == "legacy_bf16") {
            plan.precision.master_storage = "bf16";
            plan.precision.forward_execution = "bf16";
            plan.precision.backward_dx_execution = "bf16";
        } else if (request.precision_profile == "legacy_fp8" ||
                   request.precision_profile == "ampere_fp8_packed") {
            plan.precision.master_storage = "unspecified_legacy";
            plan.precision.forward_execution = request.precision_profile;
            plan.precision.backward_dx_execution = request.precision_profile;
            const auto state = require("precision.fp8");
            if (state == CapabilityState::RuntimeUnverified) {
                plan.runtime_verification_required = true;
            }
        } else {
            reject("precision profile is not admitted by planner foundation: " +
                   request.precision_profile);
        }
    } else if (request.device.runtime == "cpu") {
        if (request.precision_profile != "fp32" &&
            request.precision_profile != "cpu_fp32") {
            reject("CPU planner foundation supports precision_profile=fp32 only");
        }
        plan.precision.master_storage = "f32";
        plan.precision.forward_execution = "f32";
        plan.precision.backward_dx_execution = "f32";
    } else if (request.device.runtime == "opencl") {
        if (request.precision_profile != "fp32" &&
            request.precision_profile != "opencl_fp32") {
            reject("OpenCL planner foundation supports precision_profile=fp32 only");
        }
        plan.precision.master_storage = "f32";
        plan.precision.forward_execution = "f32";
        plan.precision.backward_dx_execution = "f32";
    }

    if (plan.device.devices.empty()) {
        plan.device.parallel_mode = "single_device";
        plan.device.transport = "none";
    } else {
        plan.device.parallel_mode = "layer_pipeline";
        plan.device.transport = request.device.peer_transport;
        if (plan.device.devices.size() != 2) {
            reject("layer pipeline requires exactly two distinct devices");
        } else if (plan.device.devices[0] == plan.device.devices[1]) {
            reject("layer pipeline devices must be distinct");
        }
        if (request.device.runtime != "cuda") {
            reject("layer pipeline is currently admitted only for CUDA");
        }
        if (request.device.peer_transport != "auto" &&
            request.device.peer_transport != "cuda_p2p" &&
            request.device.peer_transport != "host_staged") {
            reject("peer transport must be auto, cuda_p2p, or host_staged");
        }
        if (request.device.peer_transport == "cuda_p2p") {
            const auto state = require("parallel.cuda_p2p");
            if (state == CapabilityState::RuntimeUnverified) {
                plan.runtime_verification_required = true;
            }
        } else if (request.device.peer_transport == "host_staged") {
            require("parallel.host_staged");
        } else {
            // AUTO is a planner choice only after topology probing. It is
            // intentionally retained as unresolved rather than silently
            // becoming P2P or host staging in this foundation pass.
            require("parallel.cuda_p2p");
            plan.runtime_verification_required = true;
        }
    }

    if (request.device.runtime == "cpu" || request.device.runtime == "opencl") {
        // The existing portability trainers are intentionally disabled before
        // their optimizer path. The plan must say so instead of presenting a
        // build target as an executable training capability.
        reject("selected portability backend has no admitted Lion execution path");
    }

    plan.accepted = plan.rejection.empty();
    return resolution;
}

std::string canonical_plan_json(const ExecutionPlan& plan) {
    return plan_json_value(plan).dump();
}

std::string execution_plan_sha256(const ExecutionPlan& plan) {
    return sha256_hex(canonical_plan_json(plan));
}

std::string plan_json(const ExecutionPlan& plan, bool pretty) {
    auto value = plan_json_value(plan);
    value["execution_plan_sha256"] = sha256_hex(value.dump());
    return value.dump(pretty ? 2 : -1);
}

std::string capabilities_json(
    const RuntimeCapabilities& capabilities,
    bool pretty
) {
    json value = {
        {"schema", "canopy-foundry-runtime-capabilities.v1"},
        {"source", capabilities.source},
        {"capabilities", json::array()},
    };
    for (const auto& capability : capabilities.entries) {
        value["capabilities"].push_back(capability_json(capability));
    }
    return value.dump(pretty ? 2 : -1);
}

}  // namespace canopy_foundry
