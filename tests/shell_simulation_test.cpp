#include "canopy_foundry/shell/simulation.hpp"

#include <cassert>
#include <string>

int main() {
    using namespace canopy_foundry::shell;

    SimulationState state;
    assert(state.stage == SimulationStage::Idle);
    assert(!state.active);

    start_simulation(state);
    assert(state.active);
    assert(!state.paused);
    assert(state.stage == SimulationStage::Requested);
    assert(state.progress_percent == 3);

    advance_simulation(state);
    const auto first_tick = state.tick;
    assert(first_tick == 1);
    assert(state.progress_percent > 3);

    toggle_simulation_pause(state);
    assert(state.paused);
    advance_simulation(state);
    assert(state.tick == first_tick);
    toggle_simulation_pause(state);
    assert(!state.paused);

    for (int index = 0; index < 120; ++index) advance_simulation(state);
    assert(state.completed);
    assert(!state.active);
    assert(state.stage == SimulationStage::Persisted);
    assert(state.progress_percent == 100);
    assert(state.memory_percent >= 5);
    assert(simulation_stage_name(state.stage) == std::string("PERSISTED"));

    reset_simulation(state);
    assert(!state.active);
    assert(!state.completed);
    assert(state.stage == SimulationStage::Idle);
    assert(state.tick == 0);
    return 0;
}
