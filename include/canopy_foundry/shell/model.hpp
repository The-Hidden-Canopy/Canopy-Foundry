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

struct ShellState {
    View view{View::Overview};
    std::filesystem::path frontend_root;
    bool frontend_present{false};
    SpriteSheet ida_sprite;
    std::uint64_t animation_tick{0};
    SimulationState simulation;
    DocumentSummary document;
    std::string status{"Ready. No local document loaded."};
};

[[nodiscard]] const char* view_name(View view) noexcept;
[[nodiscard]] bool frontend_available(const std::filesystem::path& root);
[[nodiscard]] bool load_document(
    ShellState& state,
    const std::filesystem::path& path,
    std::string& error
);

}  // namespace canopy_foundry::shell
