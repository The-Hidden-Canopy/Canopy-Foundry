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

// Evidence is an assertion boundary, not a performance grade.  A build or
// source declaration must never be serialized as hardware or burn evidence.
enum class EvidenceTier {
    SourceOnly,
    UnitVerified,
    HardwareObserved,
    BurnObserved,
    PromotionEligible,
};

struct CapabilityStatus {
    std::string id;
    CapabilityState state{CapabilityState::Unavailable};
    std::string evidence;
    EvidenceTier evidence_tier{EvidenceTier::SourceOnly};
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
std::string evidence_tier_name(EvidenceTier tier);

}  // namespace canopy_foundry
