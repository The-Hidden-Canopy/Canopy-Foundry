#include "canopy_foundry/optimizer/plan.hpp"

#include <iostream>

namespace {

bool require(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

}  // namespace

int main() {
    using namespace canopy_foundry;
    if (!require(optimizer_owner_name(OptimizerOwner::CpuThread) == "cpu_thread",
                 "optimizer owner name changed")) return 1;

    const std::vector<OptimizerGroup> disjoint{
        {{{7, 0, 10}}, OptimizerOwner::Device},
        {{{7, 10, 20}}, OptimizerOwner::CpuThread},
        {{{8, 0, 4}}, OptimizerOwner::HostMappedGpuMath},
    };
    if (!require(validate_optimizer_groups(disjoint).valid,
                 "disjoint optimizer ownership was rejected")) return 1;

    const std::vector<OptimizerGroup> overlap{
        {{{7, 0, 10}}, OptimizerOwner::Device},
        {{{7, 9, 20}}, OptimizerOwner::CpuThread},
    };
    const auto invalid = validate_optimizer_groups(overlap);
    if (!require(!invalid.valid && invalid.reason.find("overlapping") != std::string::npos,
                 "overlapping optimizer ownership was admitted")) return 1;

    const std::vector<OptimizerGroup> malformed{
        {{{7, 5, 5}}, OptimizerOwner::Device},
    };
    if (!require(!validate_optimizer_groups(malformed).valid,
                 "empty optimizer range was admitted")) return 1;

    const std::vector<OptimizerGroup> duplicate_owner{
        {{{7, 0, 10}, {7, 9, 12}}, OptimizerOwner::Device},
    };
    if (!require(!validate_optimizer_groups(duplicate_owner).valid,
                 "same-owner overlapping ranges were admitted")) return 1;
    return 0;
}
