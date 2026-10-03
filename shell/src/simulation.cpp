#include "canopy_foundry/shell/simulation.hpp"

#include <algorithm>

namespace canopy_foundry::shell {

const char* simulation_stage_name(SimulationStage stage) noexcept {
    switch (stage) {
        case SimulationStage::Idle:
            return "IDLE";
        case SimulationStage::Requested:
            return "REQUESTED";
        case SimulationStage::Resolved:
            return "RESOLVED";
        case SimulationStage::Accepted:
            return "ACCEPTED";
        case SimulationStage::Routing:
            return "ROUTING";
        case SimulationStage::Executing:
            return "EXECUTING";
        case SimulationStage::Checkpointing:
            return "CHECKPOINTING";
        case SimulationStage::Persisted:
            return "PERSISTED";
    }
    return "IDLE";
}

void start_simulation(SimulationState& state) noexcept {
    state = {};
    state.active = true;
    state.stage = SimulationStage::Requested;
    state.progress_percent = 3;
    state.memory_percent = 8;
    state.scenario = "Northstar / toy MoE";
}

void reset_simulation(SimulationState& state) noexcept {
    state = {};
}

void toggle_simulation_pause(SimulationState& state) noexcept {
    if (state.active && !state.completed) state.paused = !state.paused;
}

void advance_simulation(SimulationState& state) noexcept {
    if (!state.active || state.paused || state.completed) return;

    ++state.tick;
    const auto tick = state.tick;
    if (tick < 10) {
        state.stage = SimulationStage::Requested;
        state.progress_percent = 4 + static_cast<int>(tick * 0.9);
        state.memory_percent = 8 + static_cast<int>(tick / 4);
    } else if (tick < 20) {
        state.stage = SimulationStage::Resolved;
        state.progress_percent = 14 + static_cast<int>((tick - 10) * 1.2);
        state.memory_percent = 12 + static_cast<int>((tick - 10) / 3);
    } else if (tick < 29) {
        state.stage = SimulationStage::Accepted;
        state.progress_percent = 27 + static_cast<int>((tick - 20) * 1.2);
        state.memory_percent = 16 + static_cast<int>((tick - 20) / 3);
    } else if (tick < 43) {
        state.stage = SimulationStage::Routing;
        state.progress_percent = 38 + static_cast<int>((tick - 29) * 1.0);
        state.memory_percent = 22 + static_cast<int>((tick - 29) / 2);
    } else if (tick < 75) {
        state.stage = SimulationStage::Executing;
        state.progress_percent = 52 + static_cast<int>((tick - 43) * 0.9);
        state.memory_percent = 34 + static_cast<int>((tick - 43) * 0.55);
    } else if (tick < 86) {
        state.stage = SimulationStage::Checkpointing;
        state.progress_percent = 81 + static_cast<int>((tick - 75) * 1.2);
        state.memory_percent = 58 + static_cast<int>((tick - 75) / 2);
    } else if (tick < 96) {
        state.stage = SimulationStage::Persisted;
        state.progress_percent = 94 + static_cast<int>((tick - 86) * 0.6);
        state.memory_percent = 54 - static_cast<int>((tick - 86) / 3);
    } else {
        state.stage = SimulationStage::Persisted;
        state.progress_percent = 100;
        state.memory_percent = 48;
        state.active = false;
        state.completed = true;
    }

    state.progress_percent = std::clamp(state.progress_percent, 0, 100);
    state.memory_percent = std::clamp(state.memory_percent, 5, 92);
    state.active_layer = static_cast<int>((tick / 4) % 8);
    state.active_token = static_cast<int>((tick * 7) % 128);
    for (std::size_t expert = 0; expert < state.expert_load.size(); ++expert) {
        const auto wave = static_cast<int>((tick + expert * 9) % 13);
        state.expert_load[expert] = (state.stage == SimulationStage::Routing ||
                                     state.stage == SimulationStage::Executing)
            ? 2 + wave
            : static_cast<int>((tick + expert) % 4);
    }
}

}  // namespace canopy_foundry::shell
