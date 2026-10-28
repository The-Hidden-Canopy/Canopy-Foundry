#include "canopy_foundry/shell/model.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
    using namespace canopy_foundry::shell;
    ShellState state;
    assert(state.mode == WorkspaceMode::Ida);
    assert(state.view == View::Overview);
    state.document.loaded = true;
    state.document.name = "selected.json";
    state.local_run_id = "owned-fixture";
    state.simulation.active = true;
    state.simulation.tick = 17;
    state.view = View::Evidence;
    switch_workspace_mode(state, WorkspaceMode::TrainBrain);
    assert(state.view == View::Plan);
    assert(state.document.name == "selected.json");
    assert(state.simulation.active && state.simulation.tick == 17);
    state.view = View::Capabilities;
    switch_workspace_mode(state, WorkspaceMode::Ida);
    assert(state.view == View::Evidence);
    switch_workspace_mode(state, WorkspaceMode::TrainBrain);
    assert(state.view == View::Capabilities);
    switch_workspace_mode(state, WorkspaceMode::TrainBrain);
    assert(state.view == View::Capabilities);
    assert(state.document.loaded && state.simulation.active && state.simulation.tick == 17);
    assert(state.local_run_id == "owned-fixture");
}
