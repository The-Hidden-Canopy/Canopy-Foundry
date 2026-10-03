#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace canopy_foundry::shell {

enum class SimulationStage {
    Idle,
    Requested,
    Resolved,
    Accepted,
    Routing,
    Executing,
    Checkpointing,
    Persisted,
};

struct SimulationState {
    bool active{false};
    bool paused{false};
    bool completed{false};
    SimulationStage stage{SimulationStage::Idle};
    std::uint64_t tick{0};
    int progress_percent{0};
    int active_layer{0};
    int active_token{0};
    int memory_percent{8};
    std::array<int, 4> expert_load{0, 0, 0, 0};
    std::string scenario{"Northstar / toy MoE"};
};

[[nodiscard]] const char* simulation_stage_name(SimulationStage stage) noexcept;

void start_simulation(SimulationState& state) noexcept;
void reset_simulation(SimulationState& state) noexcept;
void toggle_simulation_pause(SimulationState& state) noexcept;
void advance_simulation(SimulationState& state) noexcept;

}  // namespace canopy_foundry::shell
