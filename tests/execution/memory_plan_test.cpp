#include "canopy_foundry/memory/allocation_plan.hpp"

#include <iostream>
#include <limits>

namespace {

bool require(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

}  // namespace

int main() {
    using namespace canopy_foundry;

    std::uint64_t result = 0;
    if (!require(checked_add(4, 5, result) && result == 9,
                 "checked addition changed valid arithmetic")) return 1;
    if (!require(!checked_add(std::numeric_limits<std::uint64_t>::max(), 1, result),
                 "checked addition admitted overflow")) return 1;
    if (!require(!checked_multiply(std::numeric_limits<std::uint64_t>::max(), 2, result),
                 "checked multiplication admitted overflow")) return 1;

    MemoryInputs inputs{};
    inputs.persistent = {10, 20, 30};
    inputs.per_microstep = {40, 50, 60};
    inputs.checkpoint_peak = {70, 80, 90};
    inputs.master_residency = Residency::HostPinned;
    inputs.optimizer_state_residency = Residency::HostMapped;
    inputs.activation_policy = "recompute";
    const auto plan = make_memory_plan(inputs);
    if (!require(plan.estimate_valid, "valid memory estimate was rejected")) return 1;
    if (!require(plan.estimated_peak.device_bytes == 120 &&
                     plan.estimated_peak.pinned_host_bytes == 150 &&
                     plan.estimated_peak.pageable_host_bytes == 180,
                 "memory peak did not preserve all allocation classes")) return 1;
    if (!require(plan.master_residency == Residency::HostPinned &&
                     plan.optimizer_state_residency == Residency::HostMapped,
                 "memory residency was not retained")) return 1;
    if (!require(!plan.runtime_capacity_verified,
                 "shape estimate was presented as runtime capacity evidence")) return 1;

    inputs.activation_policy.clear();
    const auto invalid_policy = make_memory_plan(inputs);
    if (!require(!invalid_policy.estimate_valid,
                 "empty activation policy was admitted")) return 1;

    inputs.activation_policy = "resident";
    inputs.persistent.device_bytes = std::numeric_limits<std::uint64_t>::max();
    inputs.per_microstep.device_bytes = 1;
    const auto overflow = make_memory_plan(inputs);
    if (!require(!overflow.estimate_valid && !overflow.rejection.empty(),
                 "memory peak overflow was not rejected")) return 1;

    return 0;
}
