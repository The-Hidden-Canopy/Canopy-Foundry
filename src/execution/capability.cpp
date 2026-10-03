#include "canopy_foundry/execution/capability.hpp"

namespace canopy_foundry {

std::string capability_state_name(CapabilityState state) {
    switch (state) {
    case CapabilityState::Available: return "AVAILABLE";
    case CapabilityState::Unavailable: return "UNAVAILABLE";
    case CapabilityState::Experimental: return "EXPERIMENTAL";
    case CapabilityState::BuildDisabled: return "BUILD_DISABLED";
    case CapabilityState::HardwareUnsupported: return "HARDWARE_UNSUPPORTED";
    case CapabilityState::RuntimeUnverified: return "RUNTIME_UNVERIFIED";
    }
    return "UNAVAILABLE";
}

std::string evidence_tier_name(EvidenceTier tier) {
    switch (tier) {
    case EvidenceTier::SourceOnly: return "SOURCE_ONLY";
    case EvidenceTier::UnitVerified: return "UNIT_VERIFIED";
    case EvidenceTier::HardwareObserved: return "HARDWARE_OBSERVED";
    case EvidenceTier::BurnObserved: return "BURN_OBSERVED";
    case EvidenceTier::PromotionEligible: return "PROMOTION_ELIGIBLE";
    }
    return "SOURCE_ONLY";
}

}  // namespace canopy_foundry
