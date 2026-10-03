#include "canopy_foundry/parallel/plan.hpp"

#include <iostream>

namespace {

bool require(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

}  // namespace

int main() {
    using namespace canopy_foundry;
    if (!require(parallel_mode_name(ParallelMode::LayerPipeline) == "layer_pipeline",
                 "parallel mode name changed")) return 1;
    if (!require(transport_kind_name(TransportKind::Unresolved) == "unresolved",
                 "unresolved transport was not explicit")) return 1;
    if (!require(pipeline_schedule_name(PipelineSchedule::OneForwardOneBackward) == "1f1b",
                 "pipeline schedule name changed")) return 1;

    ParallelPlan plan{};
    plan.mode = ParallelMode::LayerPipeline;
    plan.transport = TransportKind::HostStaged;
    plan.devices = {0, 1};
    plan.pipeline_split_layer = 4;
    plan.topology_probe_required = true;
    plan.transport_explicit = true;
    if (!require(plan.devices.size() == 2 && plan.transport_explicit,
                 "parallel plan lost explicit transport identity")) return 1;
    if (!require(!plan.runtime_verified,
                 "parallel plan presented topology as observed")) return 1;
    return 0;
}
