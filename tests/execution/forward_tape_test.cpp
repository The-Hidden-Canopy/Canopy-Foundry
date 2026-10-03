#include "canopy_foundry/train/forward_tape.hpp"

#include <iostream>

namespace {

using namespace canopy_foundry;

bool require(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

RoutePlan route_plan() {
    RoutePlan plan{};
    plan.key = {9, 1};
    plan.generation = {3};
    plan.counts_token = 1;
    plan.row_indices_token = 2;
    plan.offsets_token = 3;
    plan.selected_experts_token = 4;
    plan.topk_probabilities_token = 5;
    plan.schedule_fingerprint = 6;
    plan.experts = 4;
    plan.source_rows = 16;
    plan.routes = 16;
    plan.top_k = 2;
    return plan;
}

LayerForwardTape dense_layer(std::uint32_t layer) {
    return {layer, layer + 10, layer + 20, layer + 30, std::nullopt, std::nullopt};
}

}  // namespace

int main() {
    if (!require(activation_policy_name(ActivationPolicy::Recompute) == "recompute",
                 "activation policy name changed")) return 1;

    RoutePlanStore routes;
    (void)routes.publish(route_plan());

    ForwardTape tape({7}, 9, ActivationPolicy::Recompute);
    tape.append_layer(dense_layer(0));
    tape.append_layer({
        1, 11, 21, 31,
        RoutePlanKey{9, 1},
        RouteGeneration{3},
    });
    tape.seal();
    if (!require(tape.valid() && tape.sealed(),
                 "valid forward tape did not seal")) return 1;

    validate_forward_tape_for_backward(tape, {7}, 9, routes);

    bool model_rejected = false;
    try {
        validate_forward_tape_for_backward(tape, {8}, 9, routes);
    } catch (const ForwardTapeError&) {
        model_rejected = true;
    }
    if (!require(model_rejected,
                 "backward accepted a different model generation")) return 1;

    bool microstep_rejected = false;
    try {
        validate_forward_tape_for_backward(tape, {7}, 10, routes);
    } catch (const ForwardTapeError&) {
        microstep_rejected = true;
    }
    if (!require(microstep_rejected,
                 "backward accepted a different microstep")) return 1;

    RoutePlanStore empty_routes;
    bool route_rejected = false;
    try {
        validate_forward_tape_for_backward(tape, {7}, 9, empty_routes);
    } catch (const ForwardTapeError&) {
        route_rejected = true;
    }
    if (!require(route_rejected,
                 "backward accepted a missing forward-owned route plan")) return 1;

    bool append_after_seal_rejected = false;
    try {
        tape.append_layer(dense_layer(2));
    } catch (const ForwardTapeError&) {
        append_after_seal_rejected = true;
    }
    if (!require(append_after_seal_rejected,
                 "sealed forward tape was mutated")) return 1;
    return 0;
}
