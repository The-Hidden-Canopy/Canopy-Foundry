#pragma once
#include <filesystem>
#include <string>
namespace canopy_foundry::shell {
struct ShellState;
enum class GuideDocument { Starter, Foundry };
std::filesystem::path guide_document_path(const std::filesystem::path& frontend_root, GuideDocument document);
std::filesystem::path default_guide_state_directory();
bool guide_seen(const std::filesystem::path& directory);
void close_guide(ShellState& state);
bool dismiss_guide(const std::filesystem::path& directory, std::string& error);
}
