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

        // JSON paths are UTF-8 even when the Windows ANSI code page is not.
        const auto unicode_path = root / fs::u8path("asset-\xCE\xA9");
        const auto utf8_name = unicode_path.u8string();
        const std::string encoded_path(utf8_name.begin(), utf8_name.end());
        json unicode_request = valid;
        unicode_request["output_dir"] = encoded_path;
        unicode_request["input"] = {{"token_blocks", encoded_path}};
        write_request(request_path, unicode_request);
        const auto unicode_loaded = ida_native::load_request(request_path);
        if (unicode_loaded.output_dir != unicode_path || unicode_loaded.input.token_blocks != unicode_path)
            throw std::runtime_error("UTF-8 request paths did not round-trip");

        // Preserve actual CRLF bytes as parsed, independent of Windows text mode.
        std::string windows_json;
        for (const char ch : valid.dump(2)) {
            if (ch == '\n') windows_json += '\r';
            windows_json += ch;
        }
        windows_json += "\r\n";
        { std::ofstream output(request_path, std::ios::binary); output << windows_json; }
        std::string parsed_bytes;
        (void)ida_native::load_request(request_path, parsed_bytes);
        if (parsed_bytes != windows_json) {
            std::cerr << "parsed request bytes were not preserved\n";
            fs::remove_all(root);
            return 1;
        }

        constexpr std::size_t limit = 2u << 20;
        std::string at_limit = valid.dump();
        at_limit.append(limit - at_limit.size(), ' ');
        { std::ofstream output(request_path, std::ios::binary); output << at_limit; }
        parsed_bytes.clear();
        (void)ida_native::load_request_bounded(request_path, parsed_bytes, limit);
        if (parsed_bytes != at_limit) throw std::runtime_error("bounded read changed an exact-limit request");
        { std::ofstream output(request_path, std::ios::binary | std::ios::app); output << ' '; }
        parsed_bytes = "stale";
        bool oversized_rejected = false;
        try { (void)ida_native::load_request_bounded(request_path, parsed_bytes, limit); }
        catch (const std::exception&) { oversized_rejected = true; }
        if (!oversized_rejected || !parsed_bytes.empty()) throw std::runtime_error("oversized request was not safely rejected");
        // The established general loader retains its prior, unbounded API.
        (void)ida_native::load_request(request_path);

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
