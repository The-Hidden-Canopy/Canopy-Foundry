#include "canopy_foundry/parallel/plan.hpp"

namespace canopy_foundry {

std::string parallel_mode_name(ParallelMode mode) {
    switch (mode) {
    case ParallelMode::SingleDevice: return "single_device";
    case ParallelMode::LayerPipeline: return "layer_pipeline";
    case ParallelMode::ExpertSharded: return "expert_sharded";
    }
    return "unknown";
}

std::string transport_kind_name(TransportKind kind) {
    switch (kind) {
    case TransportKind::None: return "none";
    case TransportKind::Unresolved: return "unresolved";
    case TransportKind::CudaP2P: return "cuda_p2p";
    case TransportKind::HostStaged: return "host_staged";
    }
    return "unknown";
}

std::string pipeline_schedule_name(PipelineSchedule schedule) {
    switch (schedule) {
    case PipelineSchedule::Sequential: return "sequential";
    case PipelineSchedule::OneForwardOneBackward: return "1f1b";
    }
    return "unknown";
}

}  // namespace canopy_foundry
