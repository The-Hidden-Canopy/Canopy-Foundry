#pragma once

#include "canopy_foundry/execution/service.hpp"

#include <filesystem>
#include <string>

namespace canopy_foundry {
// Runs scripts/neural_forge_worker.py once. No job is chosen here and no local
// request is used. Worker settings stay in the inherited environment and are
// never placed on a command line or in the report.
struct HubWorkerLaunch {
    std::filesystem::path python;        // interpreter; bare names are resolved on PATH
    std::filesystem::path foundry_root;  // working directory; contains scripts/
    std::filesystem::path report_path;   // worker --once-report output
    std::filesystem::path log_path;      // worker stdout/stderr
};

// Locates the worker from CANOPY_FOUNDRY_ROOT, else from the ancestors of
// `executable`. Python comes from CANOPY_FOUNDRY_PYTHON or the platform default.
// Throws when the worker cannot be found.
HubWorkerLaunch hub_worker_launch_for(const std::filesystem::path& executable);
// Same, from the running executable's own location.
HubWorkerLaunch hub_worker_launch_for_this_process();
// Empty when the worker's required settings are present (values are not read).
std::string hub_worker_configuration_problem();
// Maps the worker exit code and its --once report to a frontend result:
// 0 succeeded; 1 worker or job failed; 4 nothing queued or outcome unknown.
LocalOperationResult interpret_hub_worker_report(int worker_exit_code,
                                                 const std::filesystem::path& report_path);
// Runs the worker once in owned containment and waits for it to finish.
LocalOperationResult run_next_hub_job(const HubWorkerLaunch& launch);
}
