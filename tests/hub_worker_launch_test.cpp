#include "canopy_foundry/execution/hub_worker.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {
namespace fs = std::filesystem;

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

void write(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << text;
}

std::string read(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string report(const std::string& body) {
    return R"({"schema":"canopy_foundry.worker_once_report.v1",)" + body + "}";
}
}  // namespace

int main(int argc, char** argv) {
    using canopy_foundry::interpret_hub_worker_report;
    const fs::path root = fs::temp_directory_path() / "foundry-hub-worker-test";
    fs::remove_all(root);
    const fs::path path = root / "report.json";

    assert(interpret_hub_worker_report(2, path).exit_code == 1);
    assert(interpret_hub_worker_report(0, path).exit_code == 4);  // missing report: unknown, never success
    write(path, "{not json");
    assert(interpret_hub_worker_report(0, path).exit_code == 4);
    write(path, R"({"schema":"other","claimed":false})");
    assert(interpret_hub_worker_report(0, path).exit_code == 4);
    write(path, report(R"("claimed":false,"run_id":null,"hub_status":null)"));
    auto result = interpret_hub_worker_report(0, path);
    assert(result.exit_code == 4 && result.message.find("No job is queued") != std::string::npos);
    write(path, report(R"("claimed":true,"run_id":"nf-0123456789","hub_status":"succeeded")"));
    result = interpret_hub_worker_report(0, path);
    assert(result.exit_code == 0 && result.run_id == "nf-0123456789" && result.job_stopped);
    write(path, report(R"("claimed":true,"run_id":"nf-0123456789","hub_status":"failed")"));
    assert(interpret_hub_worker_report(0, path).exit_code == 1);
    write(path, report(R"("claimed":true,"run_id":"nf-0123456789","hub_status":"running")"));
    result = interpret_hub_worker_report(0, path);
    assert(result.exit_code == 4 && !result.job_stopped);
    write(path, report(R"("claimed":true,"run_id":"nf-0123456789","hub_status":null)"));
    assert(interpret_hub_worker_report(0, path).exit_code == 4);

    // Discovery walks up from the executable to the checkout holding the worker.
    const fs::path checkout = root / "checkout";
    write(checkout / "scripts" / "neural_forge_worker.py",
          "import json, os, sys\n"
          "report = sys.argv[sys.argv.index('--once-report') + 1]\n"
          "print('argv=' + ' '.join(sys.argv[1:]))\n"
          "print('cwd-has-scripts=' + str(os.path.isdir('scripts')))\n"
          "print('token-present=' + str(bool(os.environ.get('NEURAL_FORGE_WORKER_TOKEN'))))\n"
          "json.dump({'schema': 'canopy_foundry.worker_once_report.v1', 'claimed': True,\n"
          "           'run_id': 'nf-fake-run-01', 'hub_status': 'succeeded'}, open(report, 'w'))\n");
    set_env("CANOPY_FOUNDRY_ROOT", nullptr);
    auto launch = canopy_foundry::hub_worker_launch_for(checkout / "build" / "bin" / "foundry");
    assert(fs::equivalent(launch.foundry_root, checkout));
    bool threw = false;
    try { canopy_foundry::hub_worker_launch_for(root / "elsewhere" / "foundry"); } catch (const std::exception&) { threw = true; }
    assert(threw);

    // Required settings are checked by presence before anything starts.
    for (const char* name : {"NEURAL_FORGE_HUB_URL", "NEURAL_FORGE_WORKER_TOKEN", "NEURAL_FORGE_ORG"}) set_env(name, nullptr);
    assert(!canopy_foundry::hub_worker_configuration_problem().empty());
    result = canopy_foundry::run_next_hub_job(launch);
    assert(result.exit_code == 4 && !fs::exists(launch.log_path));
    set_env("NEURAL_FORGE_HUB_URL", "https://hub.invalid");
    set_env("NEURAL_FORGE_ORG", "org-test");
    set_env("NEURAL_FORGE_WORKER_TOKEN", "secret-token-value");
    assert(canopy_foundry::hub_worker_configuration_problem().empty());

    if (argc < 2) return 0;  // no interpreter supplied: launch path not exercised
    set_env("CANOPY_FOUNDRY_PYTHON", argv[1]);
    launch = canopy_foundry::hub_worker_launch_for(checkout / "build" / "bin" / "foundry");
    write(launch.report_path, report(R"("claimed":true,"run_id":"nf-stale-report","hub_status":"succeeded")"));
    result = canopy_foundry::run_next_hub_job(launch);
    assert(result.exit_code == 0);
    assert(result.run_id == "nf-fake-run-01");  // the stale report was cleared first
    const auto log = read(launch.log_path);
    assert(log.find("--once --once-report") != std::string::npos);
    assert(log.find("cwd-has-scripts=True") != std::string::npos);
    assert(log.find("token-present=True") != std::string::npos);
    assert(log.find("secret-token-value") == std::string::npos);  // never on the command line

    set_env("CANOPY_FOUNDRY_PYTHON", (root / "missing-python").string().c_str());
    launch = canopy_foundry::hub_worker_launch_for(checkout / "build" / "bin" / "foundry");
    assert(canopy_foundry::run_next_hub_job(launch).exit_code == 4);
    fs::remove_all(root);
    return 0;
}
