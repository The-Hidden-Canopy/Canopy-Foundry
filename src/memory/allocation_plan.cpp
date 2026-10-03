#include "canopy_foundry/memory/allocation_plan.hpp"

#include <limits>

namespace canopy_foundry {

std::string residency_name(Residency residency) {
    switch (residency) {
    case Residency::Device: return "device";
    case Residency::HostPinned: return "host_pinned";
    case Residency::HostMapped: return "host_mapped";
    case Residency::Reconstructed: return "reconstructed";
    }
    return "unknown";
}

bool checked_add(
    std::uint64_t left,
    std::uint64_t right,
    std::uint64_t& result
) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool checked_multiply(
    std::uint64_t left,
    std::uint64_t right,
    std::uint64_t& result
) noexcept {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

namespace {

bool add_estimates(
    const AllocationEstimate& left,
    const AllocationEstimate& right,
    AllocationEstimate& result
) noexcept {
    return checked_add(left.device_bytes, right.device_bytes, result.device_bytes) &&
        checked_add(
            left.pinned_host_bytes,
            right.pinned_host_bytes,
            result.pinned_host_bytes) &&
        checked_add(
            left.pageable_host_bytes,
            right.pageable_host_bytes,
            result.pageable_host_bytes);
}

}  // namespace

MemoryPlan make_memory_plan(const MemoryInputs& inputs) {
    MemoryPlan plan{};
    plan.persistent = inputs.persistent;
    plan.per_microstep = inputs.per_microstep;
    plan.checkpoint_peak = inputs.checkpoint_peak;
    plan.device_reserve_bytes = inputs.device_reserve_bytes;
    plan.host_reserve_bytes = inputs.host_reserve_bytes;
    plan.master_residency = inputs.master_residency;
    plan.optimizer_state_residency = inputs.optimizer_state_residency;
    plan.activation_policy = inputs.activation_policy;

    if (plan.activation_policy.empty()) {
        plan.rejection = "activation policy must not be empty";
        return plan;
    }
    if (!add_estimates(plan.persistent, plan.per_microstep, plan.estimated_peak)) {
        plan.rejection = "memory estimate overflowed during persistent/per-microstep sum";
        return plan;
    }
    AllocationEstimate checkpoint_peak{};
    if (!add_estimates(plan.estimated_peak, plan.checkpoint_peak, checkpoint_peak)) {
        plan.rejection = "memory estimate overflowed during checkpoint peak sum";
        return plan;
    }
    // A checkpoint may overlap the training peak.  Record the conservative
    // combined result as the estimate while retaining each component above.
    plan.estimated_peak = checkpoint_peak;
    plan.estimate_valid = true;
    // Capacity is deliberately not inferred from a shape estimate.  A runtime
    // admission probe must set this after observing the actual device/host.
    plan.runtime_capacity_verified = false;
    return plan;
}

}  // namespace canopy_foundry
