#include "canopy_foundry/train/forward_tape.hpp"

#include <string>

namespace canopy_foundry {

std::string_view activation_policy_name(ActivationPolicy policy) noexcept {
    switch (policy) {
    case ActivationPolicy::Resident: return "resident";
    case ActivationPolicy::Recompute: return "recompute";
    case ActivationPolicy::HostPinnedLayerBody: return "host_pinned_layer_body";
    }
    return "unknown";
}

bool LayerForwardTape::valid_for(std::uint32_t expected_layer) const noexcept {
    if (layer != expected_layer || input_token == 0 || normalized_token == 0 ||
        output_token == 0) {
        return false;
    }
    const bool has_route_key = route_key.has_value();
    const bool has_route_generation = route_generation.has_value();
    return has_route_key == has_route_generation &&
        (!has_route_generation || route_generation->valid());
}

ForwardTape::ForwardTape(
    ModelGeneration model_generation,
    std::uint64_t microstep,
    ActivationPolicy activation_policy
)
    : model_generation_(model_generation),
      microstep_(microstep),
      activation_policy_(activation_policy) {
    if (!model_generation_.valid()) {
        throw ForwardTapeError("forward tape requires a non-zero model generation");
    }
}

void ForwardTape::append_layer(LayerForwardTape layer) {
    if (sealed_) {
        throw ForwardTapeError("cannot append to a sealed forward tape");
    }
    if (!layer.valid_for(static_cast<std::uint32_t>(layers_.size()))) {
        throw ForwardTapeError("forward tape layer metadata is invalid or out of order");
    }
    if (layer.route_key.has_value() &&
        (layer.route_key->microstep != microstep_ ||
         layer.route_key->layer != layer.layer)) {
        throw ForwardTapeError("route plan identity does not match forward tape layer");
    }
    layers_.push_back(layer);
}

void ForwardTape::seal() {
    if (layers_.empty()) {
        throw ForwardTapeError("cannot seal an empty forward tape");
    }
    if (!valid()) {
        throw ForwardTapeError("cannot seal invalid forward tape metadata");
    }
    sealed_ = true;
}

bool ForwardTape::valid() const noexcept {
    if (!model_generation_.valid() || layers_.empty()) return false;
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        if (!layers_[index].valid_for(static_cast<std::uint32_t>(index))) {
            return false;
        }
    }
    return true;
}

void validate_forward_tape_for_backward(
    const ForwardTape& tape,
    ModelGeneration model_generation,
    std::uint64_t microstep,
    RoutePlanStore& route_plans
) {
    if (!tape.sealed() || !tape.valid()) {
        throw ForwardTapeError("backward requires a sealed valid forward tape");
    }
    if (tape.model_generation() != model_generation) {
        throw ForwardTapeError("backward model generation does not match forward tape");
    }
    if (tape.microstep() != microstep) {
        throw ForwardTapeError("backward microstep does not match forward tape");
    }

    for (const auto& layer : tape.layers()) {
        if (!layer.route_key.has_value()) continue;
        try {
            (void)route_plans.require_for_backward(
                *layer.route_key, *layer.route_generation);
        } catch (const RoutePlanError& error) {
            throw ForwardTapeError(
                std::string("forward-owned route validation failed: ") + error.what());
        }
    }
}

}  // namespace canopy_foundry
