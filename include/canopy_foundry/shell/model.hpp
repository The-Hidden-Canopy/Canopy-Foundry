#pragma once

#include "canopy_foundry/shell/sprite.hpp"
#include "canopy_foundry/shell/simulation.hpp"

#include <filesystem>
#include <string>

namespace canopy_foundry::shell {

enum class View {
    Overview,
    Plan,
    Capabilities,
    Evidence,
    Boundaries,
    Simulation,
};

enum class WorkspaceMode { Ida, TrainBrain };

struct DocumentSummary {
    bool loaded{false};
    std::filesystem::path path;
    std::string name{"None loaded"};
    std::string type{"no document"};
    std::string schema{"—"};
    std::string backend{"—"};
    std::string required_arch{"—"};
    std::string model_contract{"—"};
    std::string optimizer{"—"};
    std::string planner_revision{"—"};
    std::string parallel_mode{"—"};
    std::string transport{"—"};
    std::string posture{"CONTRACT-ONLY"};
    bool observed{false};
    bool persisted{false};
};

struct PlanSummary {
    bool resolved{false};
    bool accepted{false};
    std::string rejection;
    std::string planner_revision{"-"};
    std::string parallel_mode{"-"};
    std::string transport{"-"};
    std::string master{"-"};
    std::string forward{"-"};
    std::string backward{"-"};
    std::string gradients{"-"};
    std::string accumulators{"-"};
    std::string optimizer_state{"-"};
};

struct ShellState {
    View view{View::Overview};
    WorkspaceMode mode{WorkspaceMode::Ida};
    View ida_view{View::Overview};
    View training_view{View::Plan};
    bool companion_visible{true};
    bool guide_visible{true};
    std::filesystem::path guide_state_directory;
    std::filesystem::path frontend_root;
    bool frontend_present{false};
    SpriteSheet ida_sprite;
    std::uint64_t animation_tick{0};
    SimulationState simulation;
    DocumentSummary document;
    PlanSummary plan;
    std::string local_run_id;
    std::string status{"Ready. No local document loaded."};
};

// Presentation changes never own or mutate an execution, document, or simulation.
inline void switch_workspace_mode(ShellState& state, WorkspaceMode mode) noexcept {
    if (state.mode == mode) return;
    if (state.mode == WorkspaceMode::Ida) state.ida_view = state.view;
    else state.training_view = state.view;
    state.mode = mode;
    state.view = mode == WorkspaceMode::Ida ? state.ida_view : state.training_view;
}

void resolve_selected_request(ShellState& state);
void inspect_selected_resume(ShellState& state);
void check_selected_run(ShellState& state);
void start_selected_run(ShellState& state);
void refresh_selected_run(ShellState& state);
void cancel_selected_run(ShellState& state);

[[nodiscard]] const char* view_name(View view) noexcept;
[[nodiscard]] bool frontend_available(const std::filesystem::path& root);
[[nodiscard]] bool load_document(
    ShellState& state,
    const std::filesystem::path& path,
    std::string& error
);

}  // namespace canopy_foundry::shell
