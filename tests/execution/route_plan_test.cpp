#include "canopy_foundry/moe/route_plan.hpp"

#include <iostream>

namespace {

using namespace canopy_foundry;

bool require(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

RoutePlan plan(std::uint64_t generation, std::uint64_t microstep = 3) {
    RoutePlan value{};
    value.key = {microstep, 2};
    value.generation = {generation};
    value.counts_token = 11;
    value.row_indices_token = 12;
    value.offsets_token = 13;
    value.selected_experts_token = 14;
    value.topk_probabilities_token = 15;
    value.schedule_fingerprint = 16;
    value.experts = 8;
    value.source_rows = 32;
    value.routes = 32;
    value.top_k = 2;
    value.active_expert_count = 6;
    value.actual_rows = 32;
    value.active_tiles = 12;
    value.routing_load_cv = 0.25;
    return value;
}

}  // namespace

int main() {
    RoutePlanStore store;
    const auto& forward = store.publish(plan(1));
    if (!require(forward.valid(), "valid forward route plan was rejected")) return 1;
    if (!require(store.size() == 1, "route plan was not retained")) return 1;

    const auto& backward = store.require_for_backward({3, 2}, {1});
    if (!require(&backward == &forward,
                 "backward did not consume the forward-owned plan")) return 1;
    if (!require(store.stats().backward_reuses == 1,
                 "backward route reuse was not counted")) return 1;

    bool generation_rejected = false;
    try {
        (void)store.require_for_backward({3, 2}, {2});
    } catch (const RoutePlanError&) {
        generation_rejected = true;
    }
    if (!require(generation_rejected,
                 "backward accepted a different route generation")) return 1;

    bool missing_rejected = false;
    try {
        (void)store.require_for_backward({4, 2}, {1});
    } catch (const RoutePlanError&) {
        missing_rejected = true;
    }
    if (!require(missing_rejected,
                 "backward accepted a route plan for another microstep")) return 1;

    bool duplicate_rejected = false;
    try {
        (void)store.publish(plan(2));
    } catch (const RoutePlanError&) {
        duplicate_rejected = true;
    }
    if (!require(duplicate_rejected,
                 "forward route plan was silently replaced")) return 1;

    bool non_monotonic_rejected = false;
    try {
        (void)store.publish(plan(1, 5));
    } catch (const RoutePlanError&) {
        non_monotonic_rejected = true;
    }
    if (!require(non_monotonic_rejected,
                 "non-monotonic route generation was accepted")) return 1;

    auto invalid = plan(2, 5);
    invalid.forward_owned = false;
    bool invalid_rejected = false;
    try {
        (void)store.publish(invalid);
    } catch (const RoutePlanError&) {
        invalid_rejected = true;
    }
    if (!require(invalid_rejected,
                 "non-forward-owned route state was promoted")) return 1;

    if (!require(store.stats().generation_mismatches == 1,
                 "generation mismatch evidence changed unexpectedly")) return 1;
    if (!require(store.stats().missing_rejections == 1,
                 "missing plan evidence changed unexpectedly")) return 1;
    if (!require(store.stats().duplicate_rejections == 1,
                 "duplicate plan evidence changed unexpectedly")) return 1;

    store.clear();
    if (!require(store.size() == 0, "route plan store did not clear")) return 1;
    return 0;
}
