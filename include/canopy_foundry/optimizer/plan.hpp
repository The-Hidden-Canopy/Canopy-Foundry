#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace canopy_foundry {

enum class OptimizerOwner {
    Device,
    CpuThread,
    HostMappedGpuMath,
};

std::string optimizer_owner_name(OptimizerOwner owner);

struct OptimizerParameterRange {
    std::uint64_t tensor_id{0};
    std::uint64_t begin{0};
    std::uint64_t end{0};

    [[nodiscard]] bool valid() const noexcept {
        return tensor_id != 0 && begin < end;
    }
};

struct OptimizerGroup {
    std::vector<OptimizerParameterRange> ranges;
    OptimizerOwner owner{OptimizerOwner::Device};
};

struct OptimizerPlan {
    std::string type;
    std::string state_precision;
    OptimizerOwner owner{OptimizerOwner::Device};
    bool overlap_enabled{false};
    std::vector<OptimizerGroup> groups;
};

struct OptimizerOwnershipValidation {
    bool valid{false};
    std::string reason;
};

OptimizerOwnershipValidation validate_optimizer_groups(
    const std::vector<OptimizerGroup>& groups
);

}  // namespace canopy_foundry
