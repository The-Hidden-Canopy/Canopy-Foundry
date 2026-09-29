#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace canopy_foundry {

// Capability state is deliberately richer than a boolean.  In particular,
// source or build presence must not be reported as hardware evidence.
enum class CapabilityState {
    Available,
    Unavailable,
    Experimental,
    BuildDisabled,
    HardwareUnsupported,
    RuntimeUnverified,
};

struct CapabilityStatus {
    std::string id;
    CapabilityState state{CapabilityState::Unavailable};
    std::string evidence;
};

struct RuntimeCapabilities {
    std::string source{"build_inventory"};
    std::vector<CapabilityStatus> entries;

    static RuntimeCapabilities from_build(
        bool cuda_enabled,
        bool opencl_enabled,
        bool cpu_enabled
    );

    const CapabilityStatus* find(std::string_view id) const noexcept;
    CapabilityState state(std::string_view id) const noexcept;
};

std::string capability_state_name(CapabilityState state);

}  // namespace canopy_foundry
