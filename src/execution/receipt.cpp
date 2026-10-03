#include "canopy_foundry/execution/receipt.hpp"

#include <nlohmann/json.hpp>

namespace canopy_foundry {

namespace {

using json = nlohmann::json;

}  // namespace

bool receipt_state_is_monotonic(const ExecutionReceipt& receipt) noexcept {
    if (receipt.resolved && !receipt.requested) return false;
    if (receipt.accepted && !receipt.resolved) return false;
    if (receipt.observed && !receipt.accepted) return false;
    if (receipt.persisted && !receipt.observed) return false;
    if (receipt.exact_resume_supported && !receipt.persisted) return false;
    if (receipt.model_export_complete && !receipt.persisted) return false;
    return true;
}

std::string receipt_json(const ExecutionReceipt& receipt, bool pretty) {
    const json value = {
        {"schema", receipt.schema},
        {"run_id", receipt.run_id},
        {"engine", {
            {"revision", receipt.engine_revision},
            {"binary_sha256", receipt.binary_sha256.empty()
                ? json(nullptr) : json(receipt.binary_sha256)},
        }},
        {"request_sha256", receipt.request_sha256},
        {"execution_plan_sha256", receipt.execution_plan_sha256},
        {"execution_profile_fingerprint", receipt.execution_profile_fingerprint},
        {"evidence_tier", evidence_tier_name(receipt.evidence_tier)},
        {"states", {
            {"requested", receipt.requested},
            {"resolved", receipt.resolved},
            {"accepted", receipt.accepted},
            {"observed", receipt.observed},
            {"persisted", receipt.persisted},
        }},
        {"checkpoint", {
            {"model_export_complete", receipt.model_export_complete},
            {"exact_resume_supported", receipt.exact_resume_supported},
        }},
        {"observed", {
            {"microsteps", receipt.observation.microsteps},
            {"optimizer_steps", receipt.observation.optimizer_steps},
            {"tokens", receipt.observation.tokens},
            {"skipped_updates", receipt.observation.skipped_updates},
            {"precision_pack_builds", receipt.observation.precision_pack_builds},
            {"precision_pack_reuses", receipt.observation.precision_pack_reuses},
            {"moe_route_plans", receipt.observation.moe_route_plans},
            {"moe_scheduler_builds", receipt.observation.moe_scheduler_builds},
            {"moe_scheduler_reuses", receipt.observation.moe_scheduler_reuses},
            {"forward_bytes", receipt.observation.forward_bytes},
            {"backward_bytes", receipt.observation.backward_bytes},
            {"transport", receipt.observation.transport},
        }},
    };
    return value.dump(pretty ? 2 : -1);
}

}  // namespace canopy_foundry
