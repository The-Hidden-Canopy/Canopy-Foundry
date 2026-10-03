#include "canopy_foundry/optimizer/plan.hpp"

#include <algorithm>

namespace canopy_foundry {

std::string optimizer_owner_name(OptimizerOwner owner) {
    switch (owner) {
    case OptimizerOwner::Device: return "device";
    case OptimizerOwner::CpuThread: return "cpu_thread";
    case OptimizerOwner::HostMappedGpuMath: return "host_mapped_gpu_math";
    }
    return "unknown";
}

namespace {

bool overlaps(
    const OptimizerParameterRange& left,
    const OptimizerParameterRange& right
) noexcept {
    return left.tensor_id == right.tensor_id &&
        left.begin < right.end && right.begin < left.end;
}

}  // namespace

OptimizerOwnershipValidation validate_optimizer_groups(
    const std::vector<OptimizerGroup>& groups
) {
    for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
        const auto& group = groups[group_index];
        for (const auto& range : group.ranges) {
            if (!range.valid()) {
                return {false, "optimizer group contains an invalid parameter range"};
            }
        }
        for (std::size_t range_index = 0; range_index < group.ranges.size(); ++range_index) {
            for (std::size_t other_range = 0; other_range < range_index; ++other_range) {
                if (overlaps(group.ranges[other_range], group.ranges[range_index])) {
                    return {false, "optimizer group contains overlapping parameter ranges"};
                }
            }
        }
        for (std::size_t other_index = 0; other_index < group_index; ++other_index) {
            for (const auto& left : groups[other_index].ranges) {
                for (const auto& right : group.ranges) {
                    if (overlaps(left, right)) {
                        return {false, "optimizer groups own overlapping parameter ranges"};
                    }
                }
            }
        }
    }
    return {true, {}};
}

}  // namespace canopy_foundry
