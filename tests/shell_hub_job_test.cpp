#include "canopy_foundry/execution/service.hpp"
#include "canopy_foundry/shell/model.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <atomic>
#include <cassert>
#include <chrono>
#include <string>
#include <thread>

namespace shell = canopy_foundry::shell;

namespace {
void wait_for(shell::ShellState& state) {
    for (int attempt = 0; attempt < 500 && !shell::poll_next_hub_job(state); ++attempt)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
}
}  // namespace

int main() {
    shell::ShellState state;
    state.document.loaded = true;
    state.document.path = "selected-request.json";
    std::atomic<bool> release{false};
    std::atomic<int> calls{0};

    shell::start_next_hub_job(state, [&] {
        ++calls;
        while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return canopy_foundry::LocalOperationResult{0, "Job nf-test succeeded.", "nf-test", true};
    });
    assert(shell::hub_job_active(state));
    assert(state.status.find("selected local request is not used") != std::string::npos);
    assert(!shell::poll_next_hub_job(state));  // still running: UI thread is not blocked

    shell::start_next_hub_job(state, [&] { ++calls; return canopy_foundry::LocalOperationResult{}; });
    assert(state.status.find("already running") != std::string::npos);

    release = true;
    wait_for(state);
    assert(!shell::hub_job_active(state));
    assert(state.status == "Job nf-test succeeded.");
    assert(calls == 1);  // the second request never ran
    assert(state.local_run_id.empty());  // worker runs are not local owner runs

    shell::start_next_hub_job(state, []() -> canopy_foundry::LocalOperationResult { throw std::runtime_error("boom"); });
    wait_for(state);
    assert(!shell::hub_job_active(state));
    assert(state.status.find("failed to start: boom") != std::string::npos);
    return 0;
}
