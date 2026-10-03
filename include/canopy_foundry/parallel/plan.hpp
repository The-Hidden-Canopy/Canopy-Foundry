#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace canopy_foundry {

enum class ParallelMode {
    SingleDevice,
    LayerPipeline,
    ExpertSharded,
};

enum class TransportKind {
    None,
    Unresolved,
    CudaP2P,
    HostStaged,
};

enum class PipelineSchedule {
    Sequential,
    OneForwardOneBackward,
};

std::string parallel_mode_name(ParallelMode mode);
std::string transport_kind_name(TransportKind kind);
std::string pipeline_schedule_name(PipelineSchedule schedule);

struct ParallelPlan {
    ParallelMode mode{ParallelMode::SingleDevice};
    TransportKind transport{TransportKind::None};
    PipelineSchedule schedule{PipelineSchedule::Sequential};
    std::vector<int> devices;
    int pipeline_split_layer{0};
    bool topology_probe_required{false};
    bool transport_explicit{false};
    bool runtime_verified{false};
};

}  // namespace canopy_foundry
