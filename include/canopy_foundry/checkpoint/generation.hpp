#pragma once

#include <cstdint>
#include <string>

namespace canopy_foundry {

enum class CheckpointPurpose {
    ModelExport,
    ExactResume,
};

std::string checkpoint_purpose_name(CheckpointPurpose purpose);

struct CheckpointGeneration {
    std::uint64_t generation{0};
    std::uint64_t optimizer_step{0};
    std::string execution_plan_sha256;
    std::string model_sha256;
    std::string optimizer_sha256;
    std::string data_cursor_sha256;

    [[nodiscard]] bool complete_for(CheckpointPurpose purpose) const noexcept;
};

struct CheckpointValidation {
    bool valid{false};
    std::string reason;
};

// Validate a candidate generation before it is called resumable or
// exportable.  An optional expected generation is compared field-for-field for
// exact resume; paths and file contents never enter this contract.
CheckpointValidation validate_checkpoint_generation(
    const CheckpointGeneration& candidate,
    CheckpointPurpose purpose,
    const CheckpointGeneration* expected = nullptr
);

struct CheckpointPlan {
    CheckpointPurpose purpose{CheckpointPurpose::ModelExport};
    bool atomic_publication{true};
    bool generation_validation_required{true};
    bool exact_resume_requested{false};
    bool exact_resume_supported{false};
    bool model_export_complete{false};
};

}  // namespace canopy_foundry
