#include "canopy_foundry/moe/route_plan.hpp"

#include <utility>

namespace canopy_foundry {

bool RoutePlan::valid() const noexcept {
    if (!forward_owned || !generation.valid() || experts <= 0 ||
        source_rows <= 0 || routes <= 0 || top_k <= 0 || top_k > experts) {
        return false;
    }
    if (counts_token == 0 || row_indices_token == 0 || offsets_token == 0 ||
        selected_experts_token == 0 || topk_probabilities_token == 0 ||
        schedule_fingerprint == 0) {
        return false;
    }
    if (active_expert_count < -1 || active_expert_count > experts ||
        actual_rows < -1 || actual_rows > source_rows || active_tiles < -1) {
        return false;
    }
    return routing_load_cv >= -1.0;
}

const RoutePlan& RoutePlanStore::publish(RoutePlan plan) {
    if (!plan.valid()) {
        throw RoutePlanError("invalid forward route plan");
    }
    if (plan.generation.value <= latest_generation_.value) {
        throw RoutePlanError(
            "route generation must advance monotonically at forward publication");
    }
    if (plans_.find(plan.key) != plans_.end()) {
        ++stats_.duplicate_rejections;
        throw RoutePlanError(
            "forward route plan already published for microstep/layer");
    }

    latest_generation_ = plan.generation;
    const auto [inserted, ok] = plans_.emplace(plan.key, std::move(plan));
    if (!ok) {
        throw RoutePlanError("route plan insertion failed");
    }
    ++stats_.published;
    return inserted->second;
}

const RoutePlan& RoutePlanStore::require_for_backward(
    RoutePlanKey key,
    RouteGeneration generation
) {
    if (!generation.valid()) {
        ++stats_.generation_mismatches;
        throw RoutePlanError("backward requires a non-zero route generation");
    }

    const auto found = plans_.find(key);
    if (found == plans_.end()) {
        ++stats_.missing_rejections;
        throw RoutePlanError("forward-owned route plan is missing");
    }
    if (found->second.generation != generation) {
        ++stats_.generation_mismatches;
        throw RoutePlanError(
            "backward route generation does not match forward route plan");
    }

    ++stats_.backward_reuses;
    return found->second;
}

void RoutePlanStore::clear() noexcept {
    plans_.clear();
    latest_generation_ = {};
}

}  // namespace canopy_foundry
