#include "canopy_foundry/shell/guide.hpp"
#include "canopy_foundry/shell/model.hpp"
#include <filesystem>
#include <fstream>

int main(int argc, char** argv) {
    using namespace canopy_foundry::shell;
    if (argc != 2) return 2;
    const auto root = std::filesystem::path(argv[1]);
    if (!std::filesystem::create_directory(root)) return 2;
    const auto frontend = root / "frontend";
    const auto docs = root / "docs";
    std::filesystem::create_directory(frontend);
    std::filesystem::create_directory(docs);
    if (!guide_document_path(frontend, GuideDocument::Starter).empty()) return 1;
    { std::ofstream out(root / "STARTER_GUIDE.md"); out << "fixture"; }
    { std::ofstream out(docs / "neural-foundry-guide.md"); out << "fixture"; }
    if (guide_document_path(frontend, GuideDocument::Starter).filename() != "STARTER_GUIDE.md") return 1;
    if (guide_document_path(frontend, GuideDocument::Foundry).filename() != "neural-foundry-guide.md") return 1;
    if (!guide_document_path(frontend, static_cast<GuideDocument>(99)).empty()) return 1;
    const auto pref = root / "preferences";
    if (guide_seen(pref)) return 1;
    ShellState state;
    state.guide_state_directory = pref;
    state.document.loaded = true;
    state.document.name = "selection.json";
    state.mode = WorkspaceMode::TrainBrain;
    state.view = View::Plan;
    state.simulation.active = true;
    close_guide(state);
    if (state.guide_visible || !guide_seen(pref)) return 1;
    if (state.document.name != "selection.json" || !state.simulation.active || state.view != View::Plan) return 1;
    ShellState returning;
    returning.guide_state_directory = pref;
    returning.guide_visible = !guide_seen(pref);
    if (returning.guide_visible) return 1;
    returning.guide_visible = true; // User reopens help.
    close_guide(returning);
    if (returning.guide_visible || !guide_seen(pref)) return 1;
    const auto blocked = root / "blocked";
    { std::ofstream out(blocked); out << "fixture"; }
    ShellState failure;
    failure.guide_state_directory = blocked / "preferences";
    close_guide(failure);
    if (failure.guide_visible || failure.status.find("Could not") == std::string::npos) return 1;
    std::filesystem::remove(blocked);
    std::filesystem::remove(root / "STARTER_GUIDE.md");
    std::filesystem::remove(docs / "neural-foundry-guide.md");
    std::filesystem::remove(docs);
    std::filesystem::remove(frontend);
    std::filesystem::remove(pref / "guide-dismissed-v1");
    std::filesystem::remove(pref);
    if (!std::filesystem::remove(root)) return 1;
}
