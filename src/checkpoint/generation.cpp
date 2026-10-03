#include "canopy_foundry/checkpoint/generation.hpp"

#include <cctype>

namespace canopy_foundry {

std::string checkpoint_purpose_name(CheckpointPurpose purpose) {
    switch (purpose) {
    case CheckpointPurpose::ModelExport: return "model_export";
    case CheckpointPurpose::ExactResume: return "exact_resume";
    }
    return "unknown";
}

namespace {

bool sha256(const std::string& value) noexcept {
    if (value.size() != 64) return false;
    for (const unsigned char character : value) {
        if (!std::isxdigit(character)) return false;
    }
    return true;
}

}  // namespace

bool CheckpointGeneration::complete_for(CheckpointPurpose purpose) const noexcept {
    if (generation == 0 || !sha256(execution_plan_sha256) || !sha256(model_sha256)) {
        return false;
    }
    if ((!optimizer_sha256.empty() && !sha256(optimizer_sha256)) ||
        (!data_cursor_sha256.empty() && !sha256(data_cursor_sha256))) {
        return false;
    }
    if (purpose == CheckpointPurpose::ModelExport) return true;
    return sha256(optimizer_sha256) && sha256(data_cursor_sha256);
}

CheckpointValidation validate_checkpoint_generation(
    const CheckpointGeneration& candidate,
    CheckpointPurpose purpose,
    const CheckpointGeneration* expected
) {
    if (!candidate.complete_for(purpose)) {
        return {false, "checkpoint generation is incomplete for requested purpose"};
    }
    if (purpose == CheckpointPurpose::ExactResume && expected != nullptr) {
        if (!expected->complete_for(CheckpointPurpose::ExactResume)) {
            return {false, "expected checkpoint generation is incomplete"};
        }
        if (candidate.generation != expected->generation ||
            candidate.optimizer_step != expected->optimizer_step ||
            candidate.execution_plan_sha256 != expected->execution_plan_sha256 ||
            candidate.model_sha256 != expected->model_sha256 ||
            candidate.optimizer_sha256 != expected->optimizer_sha256 ||
            candidate.data_cursor_sha256 != expected->data_cursor_sha256) {
            return {false, "checkpoint members do not agree on exact generation"};
        }
    }
    return {true, {}};
}

}  // namespace canopy_foundry
