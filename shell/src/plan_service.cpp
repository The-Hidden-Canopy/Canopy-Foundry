#include "canopy_foundry/shell/model.hpp"
#include "canopy_foundry/execution/hub_worker.hpp"
#include "canopy_foundry/execution/service.hpp"
#include <exception>
#include <mutex>
#include <optional>
#include <thread>

namespace canopy_foundry::shell {
struct HubJobTask {
    std::mutex mutex;
    std::optional<LocalOperationResult> result;
};

void start_next_hub_job(ShellState& state, std::function<LocalOperationResult()> runner) {
    if (state.hub_job) {
        state.status = "A job attempt is already running; wait for it to finish.";
        return;
    }
    auto task = std::make_shared<HubJobTask>();
    state.hub_job = task;
    state.status = "Running the next job through the worker. The selected local request is not used.";
    std::thread([task, runner = std::move(runner)] {
        LocalOperationResult result;
        try { result = runner(); }
        catch (const std::exception& error) { result = {4, std::string("Run next job failed to start: ") + error.what()}; }
        std::lock_guard lock(task->mutex);
        task->result = std::move(result);
    }).detach();
}
void start_next_hub_job(ShellState& state) {
    start_next_hub_job(state, [] { return run_next_hub_job(hub_worker_launch_for_this_process()); });
}
bool poll_next_hub_job(ShellState& state) {
    const auto task = state.hub_job;  // outlives the lock below
    if (!task) return false;
    {
        std::lock_guard lock(task->mutex);
        if (!task->result) return false;
        state.status = task->result->message;
    }
    state.hub_job.reset();
    return true;
}
void check_selected_run(ShellState& state) {
    // The request can change on disk after Resolve. The run check reloads it,
    // so the previously displayed plan must not appear to describe that check.
    state.plan = {};
    if (!state.document.loaded) {
        state.status = "Select a request JSON before checking Run.";
        return;
    }
    state.status = check_local_run(state.document.path).message;
}
void start_selected_run(ShellState& state) {
    state.plan = {};
    if (!state.document.loaded) {
        state.status = "Select a request JSON before starting Run.";
        return;
    }
    apply_start_result(state, start_local_run(state.document.path));
}
void apply_start_result(ShellState& state, const LocalOperationResult& result) {
    // Unknown outcomes must use current-owner reconciliation even when an
    // unverified run ID accompanied the response.
    if (result.exit_code == 4) state.local_run_id.clear();
    else if (!result.run_id.empty()) state.local_run_id = result.run_id;
    state.status = result.message;
}
void cancel_selected_run(ShellState& state) {
    const auto result = cancel_local_run(state.local_run_id);
    if (result.job_stopped || result.exit_code == 4) state.local_run_id.clear();
    state.status = result.message;
}
void refresh_selected_run(ShellState& state) {
    const auto result = status_local_run(state.local_run_id);
    if (result.job_stopped || result.exit_code == 4) state.local_run_id.clear();
    else if (!result.run_id.empty()) state.local_run_id = result.run_id;
    state.status = result.message;
}
void inspect_selected_resume(ShellState& state) {
    if (!state.document.loaded) {
        state.status = "Select a request JSON before inspecting its resume checkpoint.";
        return;
    }
    try {
        const auto request = ida_native::load_request(state.document.path);
        if (request.parent.resume_from_checkpoint.empty()) {
            state.status = "Request has no resume checkpoint.";
            return;
        }
        state.status = inspect_checkpoint(request.parent.resume_from_checkpoint).message;
    } catch (const std::exception& error) {
        state.status = std::string("Checkpoint inspection failed: ") + error.what();
    }
}
void resolve_selected_request(ShellState& state) {
    state.plan = {};
    if (!state.document.loaded) {
        state.status = "Select a request JSON before resolving a plan.";
        return;
    }
    try {
        const auto result = plan_request(state.document.path, build_capabilities());
        const auto& plan = result.plan;
        state.plan.resolved = true;
        state.plan.accepted = plan.accepted;
        state.plan.rejection = plan.rejection;
        state.plan.planner_revision = plan.planner_revision;
        state.plan.parallel_mode = plan.device.parallel_mode;
        state.plan.transport = plan.device.transport;
        state.plan.master = plan.precision.master_storage;
        state.plan.forward = plan.precision.forward_execution;
        state.plan.backward = plan.precision.backward_dx_execution;
        state.plan.gradients = plan.precision.gradients;
        state.plan.accumulators = plan.precision.accumulators;
        state.plan.optimizer_state = plan.precision.optimizer_state;
        state.status = plan.accepted
            ? "Plan resolved for the current file. Resolve again after edits; runtime and hardware verification remain separate."
            : "Plan rejected: " + plan.rejection;
    } catch (const std::exception& error) {
        state.status = std::string("Plan resolution failed: ") + error.what();
    }
}
}
