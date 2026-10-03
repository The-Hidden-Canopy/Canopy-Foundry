#pragma once

#include <cstdint>
#include <string>

namespace canopy_foundry {

enum class Residency {
    Device,
    HostPinned,
    HostMapped,
    Reconstructed,
};

std::string residency_name(Residency residency);

struct AllocationEstimate {
    std::uint64_t device_bytes{0};
    std::uint64_t pinned_host_bytes{0};
    std::uint64_t pageable_host_bytes{0};
};

struct MemoryInputs {
    AllocationEstimate persistent;
    AllocationEstimate per_microstep;
    AllocationEstimate checkpoint_peak;
    std::uint64_t device_reserve_bytes{0};
    std::uint64_t host_reserve_bytes{0};
    Residency master_residency{Residency::Device};
    Residency optimizer_state_residency{Residency::Device};
    std::string activation_policy{"resident"};
};

struct MemoryPlan {
    AllocationEstimate persistent;
    AllocationEstimate per_microstep;
    AllocationEstimate checkpoint_peak;
    AllocationEstimate estimated_peak;
    std::uint64_t device_reserve_bytes{0};
    std::uint64_t host_reserve_bytes{0};
    Residency master_residency{Residency::Device};
    Residency optimizer_state_residency{Residency::Device};
    std::string activation_policy{"resident"};
    bool estimate_valid{false};
    bool runtime_capacity_verified{false};
    std::string rejection;
};

// These helpers make shape-based estimates fail closed on integer overflow.
bool checked_add(
    std::uint64_t left,
    std::uint64_t right,
    std::uint64_t& result
) noexcept;
bool checked_multiply(
    std::uint64_t left,
    std::uint64_t right,
    std::uint64_t& result
) noexcept;

MemoryPlan make_memory_plan(const MemoryInputs& inputs);

}  // namespace canopy_foundry
