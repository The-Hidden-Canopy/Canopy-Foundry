#pragma once

#include "canopy_foundry/execution/capability.hpp"
#include "canopy_foundry/checkpoint/generation.hpp"
#include "canopy_foundry/memory/allocation_plan.hpp"
#include "canopy_foundry/optimizer/plan.hpp"
#include "canopy_foundry/parallel/plan.hpp"
#include "canopy_foundry/precision/execution_view.hpp"
#include "ida_native/request.hpp"

#include <string>
#include <vector>

namespace canopy_foundry {

struct DevicePlan {
    std::string runtime;
    std::string backend;
    std::string required_arch;
    std::vector<int> devices;
    std::string parallel_mode{"single_device"};
    std::string transport{"none"};
    int pipeline_split_layer{0};
};

struct ModelPlan {
    std::string architecture_contract;
    std::string family;
    int hidden_size{0};
    int intermediate_size{0};
    int layers{0};
    int heads{0};
    int kv_heads{0};
    int vocab_size{0};
    bool moe{false};
};

struct PrecisionPlan {
    // profile preserves the legacy request identity until the old execution
    // path is wrapped by a fully explicit precision planner.
    std::string profile;
    std::string master_storage;
    std::string forward_execution;
    std::string backward_dx_execution;
    std::string gradients;
    std::string accumulators;
    std::string optimizer_state;
    std::string normalization;
    // Explicit role identities are the stable contract used by execution
    // views. The legacy strings above remain for request compatibility.
    NumericFormat master_format{NumericFormat::Unknown};
    NumericFormat forward_format{NumericFormat::Unknown};
    NumericFormat backward_dx_format{NumericFormat::Unknown};
    NumericFormat packed_view_format{NumericFormat::Unknown};
    NumericFormat gradient_format{NumericFormat::Unknown};
    NumericFormat accumulator_format{NumericFormat::Unknown};
    NumericFormat optimizer_state_format{NumericFormat::Unknown};
    NumericFormat normalization_format{NumericFormat::Unknown};
};

struct MoePlan {
    std::string route_ownership{"none"};
    std::string dispatch_mode{"unresolved"};
    bool route_generation_required{false};
};

struct ExecutionPlan {
    std::string schema{"canopy-foundry-execution-plan.v1"};
    std::string planner_revision{"planner-foundation-v2-port-v1"};
    DevicePlan device;
    ModelPlan model;
    PrecisionPlan precision;
    OptimizerPlan optimizer;
    MoePlan moe;
    MemoryPlan memory;
    ParallelPlan parallel;
    CheckpointPlan checkpoint;
    std::string attention_backend;
    std::string receipt_schema{"canopy-foundry-execution-receipt.v1"};
    std::vector<std::string> required_capabilities;
    bool accepted{false};
    bool runtime_verification_required{false};
    std::string rejection;
};

struct PlanResolution {
    ExecutionPlan plan;
    RuntimeCapabilities capabilities;
};

PlanResolution resolve_execution_plan(
    const ida_native::NativeRequest& request,
    RuntimeCapabilities capabilities
);

// JSON uses ordered object keys from nlohmann::json so the returned compact
// representation is stable for identical plans.  The hash excludes the
// derived execution_plan_sha256 field.
std::string canonical_plan_json(const ExecutionPlan& plan);
std::string plan_json(const ExecutionPlan& plan, bool pretty = false);
std::string capabilities_json(
    const RuntimeCapabilities& capabilities,
    bool pretty = false
);
std::string execution_plan_sha256(const ExecutionPlan& plan);
std::string sha256_hex(std::string_view value);

}  // namespace canopy_foundry
