#include "canopy_foundry/precision/execution_view.hpp"

#include <string>
#include <utility>

namespace canopy_foundry {

std::string_view execution_role_name(ExecutionRole role) noexcept {
    switch (role) {
    case ExecutionRole::Forward: return "forward";
    case ExecutionRole::BackwardDx: return "backward_dx";
    case ExecutionRole::BackwardDwOperand: return "backward_dw_operand";
    case ExecutionRole::ExpertForward: return "expert_forward";
    case ExecutionRole::ExpertDx: return "expert_dx";
    case ExecutionRole::ExpertDw: return "expert_dw";
    }
    return "unknown";
}

std::string_view numeric_format_name(NumericFormat format) noexcept {
    switch (format) {
    case NumericFormat::Unknown: return "unknown";
    case NumericFormat::F32: return "f32";
    case NumericFormat::BF16: return "bf16";
    case NumericFormat::FP8E4M3: return "fp8_e4m3";
    case NumericFormat::MXFP8E4M3UE8M0K32:
        return "mxfp8_e4m3_ue8m0_k32";
    case NumericFormat::NVFP4E2M1UE4M3:
        return "nvfp4_e2m1_ue4m3";
    }
    return "unknown";
}

NumericFormat numeric_format_from_name(std::string_view name) noexcept {
    if (name == "f32" || name == "fp32") return NumericFormat::F32;
    if (name == "bf16") return NumericFormat::BF16;
    if (name == "fp8_e4m3" || name == "e4m3") {
        return NumericFormat::FP8E4M3;
    }
    if (name == "mxfp8_e4m3_ue8m0_k32" || name == "mxfp8") {
        return NumericFormat::MXFP8E4M3UE8M0K32;
    }
    if (name == "nvfp4_e2m1_ue4m3" || name == "nvfp4") {
        return NumericFormat::NVFP4E2M1UE4M3;
    }
    return NumericFormat::Unknown;
}

bool PackedExecutionView::valid_for(
    const ExecutionViewKey& expected_key,
    TensorGeneration expected_generation
) const noexcept {
    return ready && key == expected_key &&
        source_generation == expected_generation &&
        expected_generation.valid();
}

const PackedExecutionView* ExecutionViewCache::lookup(
    const ExecutionViewKey& key,
    TensorGeneration generation
) const noexcept {
    const auto latest = generations_.find(key.tensor);
    if (latest != generations_.end() && generation.value < latest->second.value) return nullptr;
    const auto found = views_.find(key);
    if (found == views_.end() || !found->second.valid_for(key, generation)) {
        return nullptr;
    }
    return &found->second;
}

const PackedExecutionView& ExecutionViewCache::require(
    const ExecutionViewKey& key,
    TensorGeneration generation,
    const ExecutionViewBuilder& builder
) {
    if (!generation.valid()) {
        throw ExecutionViewError(
            "execution view requires a non-zero source generation");
    }

    const auto latest = generations_.find(key.tensor);
    if (latest != generations_.end() && generation.value < latest->second.value) {
        ++stats_.stale_rejections;
        throw ExecutionViewError("execution view source generation moved backwards");
    }
    if (latest == generations_.end() || generation.value > latest->second.value) {
        // A newer generation invalidates every role/layout for this tensor,
        // not merely the key that first observed the mutation.
        for (auto it = views_.begin(); it != views_.end();) {
            if (it->first.tensor == key.tensor) {
                it = views_.erase(it);
                ++stats_.stale_rejections;
            } else {
                ++it;
            }
        }
        generations_[key.tensor] = generation;
    }

    const auto found = views_.find(key);
    if (found != views_.end()) {
        if (found->second.valid_for(key, generation)) {
            ++stats_.reuses;
            return found->second;
        }
        ++stats_.stale_rejections;
        views_.erase(found);
    }

    if (!builder) {
        ++stats_.unavailable_rejections;
        throw ExecutionViewError(
            "execution view is missing or stale and no rebuild is available");
    }

    PackedExecutionView rebuilt = builder(key, generation);
    const auto latest_after_build = generations_.find(key.tensor);
    if (latest_after_build == generations_.end() || latest_after_build->second != generation) {
        ++stats_.stale_rejections;
        throw ExecutionViewError("execution view generation changed during rebuild");
    }
    if (!rebuilt.valid_for(key, generation)) {
        ++stats_.stale_rejections;
        throw ExecutionViewError(
            "execution view builder returned mismatched or incomplete metadata");
    }

    const auto [inserted, ok] = views_.emplace(key, std::move(rebuilt));
    if (!ok) {
        throw ExecutionViewError("execution view cache insertion failed");
    }
    ++stats_.builds;
    return inserted->second;
}

void ExecutionViewCache::invalidate(
    TensorId tensor,
    TensorGeneration new_generation
) {
    if (!new_generation.valid()) {
        throw ExecutionViewError(
            "execution view invalidation requires a non-zero generation");
    }

    const auto latest = generations_.find(tensor);
    if (latest != generations_.end() && new_generation.value < latest->second.value) {
        ++stats_.stale_rejections;
        throw ExecutionViewError("execution view invalidation generation moved backwards");
    }
    generations_[tensor] = new_generation;

    for (auto it = views_.begin(); it != views_.end();) {
        if (it->first.tensor == tensor) {
            it = views_.erase(it);
            ++stats_.invalidations;
        } else {
            ++it;
        }
    }
}

void ExecutionViewCache::invalidate_all() {
    stats_.invalidations += views_.size();
    views_.clear();
}

void ExecutionViewCache::reset_for_new_model() {
    invalidate_all();
    generations_.clear();
}

}  // namespace canopy_foundry
