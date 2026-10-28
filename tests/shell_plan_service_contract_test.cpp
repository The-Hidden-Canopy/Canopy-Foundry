#include "canopy_foundry/shell/model.hpp"
#include "canopy_foundry/execution/service.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <memory>
#include <stdexcept>
#include <utility>

// These stubs isolate the shell/service adapter. They do not establish native
// planner, filesystem, device, execution, or checkpoint correctness.
static bool changed_request = false;
namespace ida_native {
NativeRequest load_request(const std::filesystem::path& path) {
    if (path == "malformed") throw std::runtime_error("malformed request");
    NativeRequest request{};
    request.family = changed_request ? "unsupported" : path.string();
    return request;
}
}
namespace canopy_foundry {
RuntimeCapabilities RuntimeCapabilities::from_build(bool, bool, bool) { return {}; }
PlanResolution resolve_execution_plan(const ida_native::NativeRequest& request,
                                      RuntimeCapabilities capabilities) {
    PlanResolution result{};
    result.capabilities = std::move(capabilities);
    result.plan.accepted = request.family == "accepted";
    result.plan.rejection = result.plan.accepted ? "" : "unsupported backend";
    result.plan.precision.master_storage = "bf16";
    result.plan.precision.forward_execution = "bf16";
    result.plan.precision.gradients = "fp32";
    result.plan.precision.optimizer_state = "fp32";
    return result;
}
std::string execution_plan_binding_error(const ida_native::NativeRequest&, const PlanResolution&) { return {}; }
}
// A fake owner tests routing and state only. It does not launch or contain a
// process, validate native admission, or prove that cancellation works.
struct FakeOwner : canopy_foundry::OwnedRunBackend {
    int starts = 0;
    int cancels = 0;
    bool unknown_on_cancel = false;
    bool pending_on_cancel = false;
    bool throw_on_start = false;
    canopy_foundry::OwnedRunObservation start(const ida_native::NativeRequest&,
                                              const canopy_foundry::PlanResolution&) override {
        ++starts;
        if (throw_on_start) throw std::runtime_error("fixture launch error");
        return {canopy_foundry::OwnedRunState::running, "fixture-1", "fixture"};
    }
    canopy_foundry::OwnedRunObservation observe() override {
        return {canopy_foundry::OwnedRunState::running, "fixture-1", "fixture"};
    }
    canopy_foundry::OwnedRunObservation cancel(std::chrono::milliseconds timeout) override {
        assert(timeout == std::chrono::seconds(3));
        ++cancels;
        return {unknown_on_cancel ? canopy_foundry::OwnedRunState::unknown
                : pending_on_cancel ? canopy_foundry::OwnedRunState::running
                                    : canopy_foundry::OwnedRunState::stopped,
                "fixture-1", "fixture"};
    }
};
int main() {
    using namespace canopy_foundry;
    using namespace canopy_foundry::shell;
    ShellState state;
    state.document.loaded = true;
    state.document.path = "accepted";
    const auto cli_resolution = plan_request(state.document.path, build_capabilities());
    resolve_selected_request(state);
    assert(state.plan.resolved && state.plan.accepted == cli_resolution.plan.accepted);
    assert(state.plan.master == "bf16" && state.plan.gradients == "fp32");
    assert(state.status.find("verification") != std::string::npos);
    state.document.path = "unsupported";
    resolve_selected_request(state);
    assert(state.plan.resolved && !state.plan.accepted);
    assert(state.status.find("unsupported backend") != std::string::npos);
    state.document.path = "malformed";
    resolve_selected_request(state);
    assert(!state.plan.resolved && state.plan.master == "-");
    assert(state.status.find("malformed request") != std::string::npos);
    state.document.path = "accepted";
    resolve_selected_request(state);
    assert(state.plan.resolved && state.plan.accepted);
    const auto run_check = check_local_run(state.document.path);
    assert(run_check.exit_code == 4);
    check_selected_run(state);
    assert(state.status == run_check.message);
    assert(!state.plan.resolved);
    start_selected_run(state);
    assert(state.status == start_local_run(state.document.path).message);
    assert(state.local_run_id.empty());
    refresh_selected_run(state);
    assert(state.status == status_local_run("").message);
    // A request may change on disk without changing its selected path. The
    // next check must replan it and remove the old accepted presentation.
    resolve_selected_request(state);
    assert(state.plan.resolved && state.plan.accepted);
    changed_request = true;
    check_selected_run(state);
    assert(!state.plan.resolved && state.plan.master == "-");
    assert(state.status.find("Run blocked") != std::string::npos);
    changed_request = false;
    const auto cancel_check = cancel_local_run("");
    assert(cancel_check.exit_code == 4);
    cancel_selected_run(state);
    assert(state.status == cancel_check.message);
    state.document.path = "unsupported";
    assert(check_local_run(state.document.path).exit_code == 3);
    check_selected_run(state);
    assert(state.status.find("Run blocked") != std::string::npos);
    state.document.loaded = false;
    resolve_selected_request(state);
    assert(!state.plan.resolved);
    check_selected_run(state);
    assert(state.status.find("Select a request") != std::string::npos);

    auto owner = std::make_unique<FakeOwner>();
    auto* owner_ptr = owner.get();
    LocalRunService lifecycle(std::move(owner));
    assert(lifecycle.start("unsupported").exit_code == 3 && owner_ptr->starts == 0);
    assert(lifecycle.start("accepted").exit_code == 0 && owner_ptr->starts == 1);
    assert(lifecycle.start("accepted").exit_code == 4 && owner_ptr->starts == 1);
    assert(lifecycle.cancel("another-run").exit_code == 4 && owner_ptr->cancels == 0);
    assert(lifecycle.status("fixture-1").exit_code == 0);
    owner_ptr->pending_on_cancel = true;
    assert(lifecycle.cancel("fixture-1").exit_code == 4 && owner_ptr->cancels == 1);
    owner_ptr->pending_on_cancel = false;
    const auto stopped = lifecycle.cancel("fixture-1");
    assert(stopped.exit_code == 0 && stopped.job_stopped && owner_ptr->cancels == 2);
    assert(lifecycle.status("fixture-1").job_stopped);
    assert(lifecycle.cancel("fixture-1").job_stopped && owner_ptr->cancels == 2);

    auto uncertain_owner = std::make_unique<FakeOwner>();
    auto* uncertain_ptr = uncertain_owner.get();
    uncertain_ptr->unknown_on_cancel = true;
    LocalRunService uncertain(std::move(uncertain_owner));
    assert(uncertain.start("accepted").exit_code == 0);
    assert(uncertain.cancel("fixture-1").exit_code == 4);
    assert(!uncertain.may_shutdown());
    assert(uncertain.start("accepted").exit_code == 4 && uncertain_ptr->starts == 1);
    assert(uncertain.cancel("fixture-1").exit_code == 4 && uncertain_ptr->cancels == 1);

    auto failing_owner = std::make_unique<FakeOwner>();
    auto* failing_ptr = failing_owner.get();
    failing_ptr->throw_on_start = true;
    LocalRunService failing(std::move(failing_owner));
    assert(failing.start("accepted").exit_code == 4);
    assert(!failing.may_shutdown());
    assert(failing.start("accepted").exit_code == 4 && failing_ptr->starts == 1);
}
