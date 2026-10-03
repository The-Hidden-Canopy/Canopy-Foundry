#pragma once

#include "canopy_foundry/execution/capability.hpp"

#include <cstdint>
#include <string>

namespace canopy_foundry {

struct ExecutionObservation {
    std::uint64_t microsteps{0};
    std::uint64_t optimizer_steps{0};
    std::uint64_t tokens{0};
    std::uint64_t skipped_updates{0};
    std::uint64_t precision_pack_builds{0};
    std::uint64_t precision_pack_reuses{0};
    std::uint64_t moe_route_plans{0};
    std::uint64_t moe_scheduler_builds{0};
    std::uint64_t moe_scheduler_reuses{0};
    std::uint64_t forward_bytes{0};
    std::uint64_t backward_bytes{0};
    std::string transport{"none"};
};

// The receipt keeps requested/resolved/accepted/observed/persisted separate.
// A plan or a synthetic shell run may populate only the first two states.
struct ExecutionReceipt {
    std::string schema{"canopy-foundry-execution-receipt.v1"};
    std::string run_id;
    std::string engine_revision;
    std::string binary_sha256;
    std::string request_sha256;
    std::string execution_plan_sha256;
    std::string execution_profile_fingerprint;
    EvidenceTier evidence_tier{EvidenceTier::SourceOnly};
    bool requested{false};
    bool resolved{false};
    bool accepted{false};
    bool observed{false};
    bool persisted{false};
    bool model_export_complete{false};
    bool exact_resume_supported{false};
    ExecutionObservation observation;
};

[[nodiscard]] bool receipt_state_is_monotonic(
    const ExecutionReceipt& receipt
) noexcept;
std::string receipt_json(const ExecutionReceipt& receipt, bool pretty = false);

}  // namespace canopy_foundry
