#include "canopy_foundry/shell/simulation.hpp"

#include <cassert>
#include <string>

int main() {
    using namespace canopy_foundry::shell;

    SimulationState state;
    assert(state.stage == SimulationStage::Idle);
    assert(!state.active);
    assert(simulation_status_name(state) == std::string("READY"));
    toggle_simulation_pause(state); // pause has no effect before starting
    assert(simulation_status_name(state) == std::string("READY"));

    start_simulation(state);
    assert(state.active);
    assert(!state.paused);
    assert(state.stage == SimulationStage::Requested);
    assert(state.progress_percent == 3);
    assert(simulation_status_name(state) == std::string("RUNNING"));

    advance_simulation(state);
    const auto first_tick = state.tick;
    assert(first_tick == 1);
    assert(state.progress_percent > 3);

    toggle_simulation_pause(state);
    assert(state.paused);
    assert(simulation_status_name(state) == std::string("PAUSED"));
    advance_simulation(state);
    assert(state.tick == first_tick);
    toggle_simulation_pause(state);
    assert(!state.paused);
    assert(simulation_status_name(state) == std::string("RUNNING"));

    for (int index = 0; index < 120; ++index) advance_simulation(state);
    assert(state.completed);
    assert(!state.active);
    assert(state.stage == SimulationStage::Persisted);
    assert(state.progress_percent == 100);
    assert(state.memory_percent >= 5);
    assert(simulation_stage_name(state.stage) == std::string("PERSISTED"));
    assert(simulation_status_name(state) == std::string("COMPLETE"));
    toggle_simulation_pause(state); // pause has no effect after completion
    assert(simulation_status_name(state) == std::string("COMPLETE"));

    start_simulation(state);
    toggle_simulation_pause(state);
    assert(simulation_status_name(state) == std::string("PAUSED"));
    start_simulation(state); // restarting clears pause and progress
    assert(simulation_status_name(state) == std::string("RUNNING"));
    assert(state.tick == 0);
    reset_simulation(state);
    assert(!state.active);
    assert(!state.completed);
    assert(state.stage == SimulationStage::Idle);
    assert(state.tick == 0);
    assert(simulation_status_name(state) == std::string("READY"));
    return 0;
}
