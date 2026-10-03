#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>

namespace canopy_foundry {

struct RouteGeneration {
    std::uint64_t value{0};

    [[nodiscard]] bool valid() const noexcept { return value != 0; }
    friend bool operator==(const RouteGeneration&, const RouteGeneration&) = default;
    friend auto operator<=>(const RouteGeneration&, const RouteGeneration&) = default;
};

struct RoutePlanKey {
    std::uint64_t microstep{0};
    std::uint32_t layer{0};

    friend bool operator==(const RoutePlanKey&, const RoutePlanKey&) = default;
    friend auto operator<=>(const RoutePlanKey&, const RoutePlanKey&) = default;
};

// A route plan is the forward-owned identity of a sparse dispatch. Tokens are
// opaque backend storage identities; the CUDA/OpenCL adapter owns the buffers
// and is responsible for populating them in the declared order.
struct RoutePlan {
    RoutePlanKey key{};
    RouteGeneration generation{};
    std::uint64_t counts_token{0};
    std::uint64_t row_indices_token{0};
    std::uint64_t offsets_token{0};
    std::uint64_t selected_experts_token{0};
    std::uint64_t topk_probabilities_token{0};
    std::uint64_t schedule_fingerprint{0};
    int experts{0};
    int source_rows{0};
    int routes{0};
    int top_k{0};
    int active_expert_count{-1};
    int actual_rows{-1};
    int active_tiles{-1};
    double routing_load_cv{-1.0};
    bool forward_owned{true};

    [[nodiscard]] bool valid() const noexcept;
};

struct RoutePlanStoreStats {
    std::uint64_t published{0};
    std::uint64_t backward_reuses{0};
    std::uint64_t missing_rejections{0};
    std::uint64_t generation_mismatches{0};
    std::uint64_t duplicate_rejections{0};
};

class RoutePlanError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Forward publishes one immutable plan for a microstep/layer identity.
// Backward can only obtain that plan by supplying the same route generation;
// it cannot silently rebuild a different row ordering.
class RoutePlanStore {
public:
    const RoutePlan& publish(RoutePlan plan);

    const RoutePlan& require_for_backward(
        RoutePlanKey key,
        RouteGeneration generation
    );

    void clear() noexcept;

    [[nodiscard]] std::size_t size() const noexcept { return plans_.size(); }
    [[nodiscard]] const RoutePlanStoreStats& stats() const noexcept {
        return stats_;
    }

private:
    std::map<RoutePlanKey, RoutePlan> plans_;
    RouteGeneration latest_generation_{};
    RoutePlanStoreStats stats_{};
};

}  // namespace canopy_foundry
