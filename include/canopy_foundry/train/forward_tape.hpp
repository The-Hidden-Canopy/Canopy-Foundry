#pragma once

#include "canopy_foundry/moe/route_plan.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace canopy_foundry {

struct ModelGeneration {
    std::uint64_t value{0};

    [[nodiscard]] bool valid() const noexcept { return value != 0; }
    friend bool operator==(const ModelGeneration&, const ModelGeneration&) = default;
};

enum class ActivationPolicy : std::uint8_t {
    Resident,
    Recompute,
    HostPinnedLayerBody,
};

std::string_view activation_policy_name(ActivationPolicy policy) noexcept;

struct LayerForwardTape {
    std::uint32_t layer{0};
    std::uint64_t input_token{0};
    std::uint64_t normalized_token{0};
    std::uint64_t output_token{0};
    std::optional<RoutePlanKey> route_key;
    std::optional<RouteGeneration> route_generation;

    [[nodiscard]] bool valid_for(std::uint32_t expected_layer) const noexcept;
};

class ForwardTapeError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The tape is mutable only until seal().  Backward receives a sealed tape and
// must validate its model/microstep identity before consuming activations or
// any forward-owned route plan.
class ForwardTape {
public:
    ForwardTape(
        ModelGeneration model_generation,
        std::uint64_t microstep,
        ActivationPolicy activation_policy
    );

    void append_layer(LayerForwardTape layer);
    void seal();

    [[nodiscard]] bool sealed() const noexcept { return sealed_; }
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] ModelGeneration model_generation() const noexcept {
        return model_generation_;
    }
    [[nodiscard]] std::uint64_t microstep() const noexcept { return microstep_; }
    [[nodiscard]] ActivationPolicy activation_policy() const noexcept {
        return activation_policy_;
    }
    [[nodiscard]] const std::vector<LayerForwardTape>& layers() const noexcept {
        return layers_;
    }

private:
    ModelGeneration model_generation_{};
    std::uint64_t microstep_{0};
    ActivationPolicy activation_policy_{};
    std::vector<LayerForwardTape> layers_;
    bool sealed_{false};
};

// Validates the complete forward/backward ownership boundary.  In particular,
// a matching model generation is not sufficient when a route plan is absent or
// its forward generation does not match the tape.
void validate_forward_tape_for_backward(
    const ForwardTape& tape,
    ModelGeneration model_generation,
    std::uint64_t microstep,
    RoutePlanStore& route_plans
);

}  // namespace canopy_foundry
