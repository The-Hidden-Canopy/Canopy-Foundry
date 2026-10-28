#include "canopy_foundry/shell/guide.hpp"
#include "canopy_foundry/shell/model.hpp"
#include <cstdlib>
#include <system_error>

namespace canopy_foundry::shell {
std::filesystem::path guide_document_path(const std::filesystem::path& frontend_root, GuideDocument document) {
    const auto root = frontend_root.parent_path();
    std::filesystem::path path;
    switch (document) {
        case GuideDocument::Starter: path = root / "STARTER_GUIDE.md"; break;
        case GuideDocument::Foundry: path = root / "docs" / "neural-foundry-guide.md"; break;
        default: return {};
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec) || ec) return {};
    const auto absolute = std::filesystem::absolute(path, ec);
    return ec ? std::filesystem::path{} : absolute;
}
std::filesystem::path default_guide_state_directory() {
#ifdef _WIN32
    const char* base = std::getenv("LOCALAPPDATA");
    return base && *base ? std::filesystem::path(base) / "CanopyFoundry" : std::filesystem::path{};
#else
    const char* base = std::getenv("XDG_CONFIG_HOME");
    if (base && *base) return std::filesystem::path(base) / "canopy-foundry";
    base = std::getenv("HOME");
    return base && *base ? std::filesystem::path(base) / ".config" / "canopy-foundry" : std::filesystem::path{};
#endif
}
bool guide_seen(const std::filesystem::path& directory) {
    if (directory.empty()) return false;
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(directory / "guide-dismissed-v1", ec);
    return !ec && std::filesystem::is_directory(status);
}
void close_guide(ShellState& state) {
    std::string error;
    if (!dismiss_guide(state.guide_state_directory, error)) state.status = error;
    state.guide_visible = false;
}
bool dismiss_guide(const std::filesystem::path& directory, std::string& error) {
    if (directory.empty()) { error = "Guide preference directory is unavailable."; return false; }
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) { error = "Could not save guide preference."; return false; }
    const auto marker = directory / "guide-dismissed-v1";
    std::filesystem::create_directory(marker, ec);
    if (ec || !guide_seen(directory)) { error = "Could not save guide preference."; return false; }
    return true;
}
}
