#include "canopy_foundry/shell/model.hpp"

#include <nlohmann/json.hpp>

#include <cctype>
#include <fstream>
#include <sstream>
#include <string_view>
#include <vector>

namespace canopy_foundry::shell {
namespace {

using json = nlohmann::json;

std::string lower(std::string value) {
    for (char& character : value) {
        character = static_cast<char>(
            std::tolower(static_cast<unsigned char>(character)));
    }
    return value;
}

const json* find_path(const json& root, std::string_view path) {
    const json* current = &root;
    std::size_t start = 0;
    while (start < path.size()) {
        const auto separator = path.find('.', start);
        const auto end = separator == std::string_view::npos
            ? path.size()
            : separator;
        const std::string key(path.substr(start, end - start));
        if (!current->is_object() || !current->contains(key)) return nullptr;
        current = &(*current)[key];
        if (separator == std::string_view::npos) break;
        start = separator + 1;
    }
    return current;
}

std::string string_at(
    const json& root,
    std::initializer_list<std::string_view> paths,
    std::string fallback = "—"
) {
    for (const auto path : paths) {
        const auto* value = find_path(root, path);
        if (value == nullptr || value->is_null()) continue;
        if (value->is_string()) return value->get<std::string>();
        if (value->is_boolean()) return value->get<bool>() ? "true" : "false";
        if (value->is_number()) return value->dump();
    }
    return fallback;
}

bool contains_any(const json& root, std::initializer_list<std::string_view> paths) {
    for (const auto path : paths) {
        if (find_path(root, path) != nullptr) return true;
    }
    return false;
}

std::string classify(const json& root) {
    const auto schema = lower(string_at(root, {"schema", "schema_version"}, ""));
    if (schema.find("receipt") != std::string::npos ||
        contains_any(root, {"observed", "persisted"})) {
        return "execution receipt";
    }
    if (schema.find("capabilit") != std::string::npos ||
        (root.contains("capabilities") && root["capabilities"].is_array())) {
        return "capability inventory";
    }
    if (schema.find("plan") != std::string::npos ||
        contains_any(root, {"device", "precision", "required_capabilities"})) {
        return "execution plan";
    }
    return "local JSON document";
}

}  // namespace

const char* view_name(View view) noexcept {
    switch (view) {
        case View::Overview:
            return "Overview";
        case View::Plan:
            return "Plan lens";
        case View::Capabilities:
            return "Capabilities";
        case View::Evidence:
            return "Evidence";
        case View::Boundaries:
            return "Ownership";
        case View::Simulation:
            return "Simulation";
    }
    return "Overview";
}

bool frontend_available(const std::filesystem::path& root) {
    std::error_code error;
    return std::filesystem::is_regular_file(root / "index.html", error) &&
        std::filesystem::is_regular_file(root / "styles.css", error) &&
        std::filesystem::is_regular_file(root / "app.js", error);
}

bool load_document(
    ShellState& state,
    const std::filesystem::path& path,
    std::string& error
) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "could not open local JSON document";
        return false;
    }

    std::ostringstream contents;
    contents << input.rdbuf();

    json root;
    try {
        root = json::parse(contents.str());
    } catch (const json::parse_error& parse_error) {
        error = std::string("invalid JSON: ") + parse_error.what();
        return false;
    }
    if (!root.is_object()) {
        error = "local JSON document must be an object";
        return false;
    }

    DocumentSummary summary;
    summary.loaded = true;
    summary.path = path;
    summary.name = path.filename().string();
    summary.type = classify(root);
    summary.schema = string_at(root, {"schema", "schema_version"});
    summary.backend = string_at(
        root,
        {"backend", "device.backend", "execution.backend"}
    );
    summary.required_arch = string_at(
        root,
        {"required_arch", "device.required_arch"}
    );
    summary.model_contract = string_at(
        root,
        {"model.architecture_contract", "model_contract_id"}
    );
    summary.optimizer = string_at(
        root,
        {"optimizer_type", "optimizer.type", "optimizer"}
    );
    summary.planner_revision = string_at(
        root,
        {"planner_revision", "execution_plan.planner_revision"}
    );
    summary.parallel_mode = string_at(
        root,
        {"parallel_mode", "device.parallel_mode"}
    );
    summary.transport = string_at(
        root,
        {"transport", "device.transport"}
    );
    summary.observed = contains_any(root, {"observed", "observation", "metrics"});
    summary.persisted = contains_any(root, {"persisted", "checkpoint", "receipt_sha256"});
    summary.posture = summary.persisted
        ? "RECEIPT-SHAPED"
        : summary.observed
            ? "OBSERVED FIELDS"
            : "REQUEST PREVIEW";

    state.document = std::move(summary);
    state.status = "Loaded locally: " + state.document.name +
        ". No upload or runtime claim was made.";
    return true;
}

}  // namespace canopy_foundry::shell
