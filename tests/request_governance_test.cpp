#include "ida_native/request.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

fs::path test_root() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path root = fs::temp_directory_path() /
        ("ida_native_request_governance_" + std::to_string(stamp));
    fs::create_directories(root);
    return root;
}

void write_request(const fs::path& path, const json& payload) {
    std::ofstream output(path);
    output << payload.dump(2) << '\n';
}

bool rejects(const fs::path& path) {
    try {
        (void)ida_native::load_request(path);
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

}  // namespace

int main() {
    const fs::path root = test_root();
    const fs::path request_path = root / "request.json";
    try {
        const json valid = {
            {"backend", "native"},
            {"optimizer_type", "lion"},
            {"output_dir", (root / "output").string()},
            {"status_file", (root / "output" / "status.json").string()},
        };
        write_request(request_path, valid);
        const auto request = ida_native::load_request(request_path);
        if (request.output_dir != root / "output" ||
            request.status_file != root / "output" / "status.json") {
            std::cerr << "valid local runtime paths were not preserved\n";
            fs::remove_all(root);
            return 1;
        }

        for (const std::string& optimizer : {"adam", "adamw"}) {
            json invalid = valid;
            invalid["optimizer_type"] = optimizer;
            write_request(request_path, invalid);
            if (!rejects(request_path)) {
                std::cerr << "accepted disabled optimizer: " << optimizer << '\n';
                fs::remove_all(root);
                return 1;
            }
        }

        for (const std::string& key : {
                 "org", "role", "justification", "audit", "domain_event",
                 "promotion_enabled", "telemetry", "socket", "serve", "worker",
                 "ontology_required", "analytics_required", "ontology_path",
                 "analytics_path", "analytics_contract",
             }) {
            json invalid = valid;
            invalid[key] = "untrusted";
            write_request(request_path, invalid);
            if (!rejects(request_path)) {
                std::cerr << "accepted forbidden governance field: " << key << '\n';
                fs::remove_all(root);
                return 1;
            }
        }

        json nested = valid;
        nested["model"] = {{"RoLe", "admin"}};
        write_request(request_path, nested);
        if (!rejects(request_path)) {
            std::cerr << "accepted nested governance field\n";
            fs::remove_all(root);
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "unexpected request governance test error: " << error.what() << '\n';
        fs::remove_all(root);
        return 1;
    }

    fs::remove_all(root);
    return 0;
}
