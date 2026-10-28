#include "canopy_foundry/execution/service.hpp"
#include <exception>
#include <utility>

namespace canopy_foundry {
namespace { LocalRunService& default_local_run_service(); }
RuntimeCapabilities build_capabilities() {
#if defined(CANOPY_FOUNDRY_BUILD_CUDA) && CANOPY_FOUNDRY_BUILD_CUDA
    constexpr bool cuda = true;
#else
    constexpr bool cuda = false;
#endif
#if defined(CANOPY_FOUNDRY_BUILD_OPENCL) && CANOPY_FOUNDRY_BUILD_OPENCL
    constexpr bool opencl = true;
#else
    constexpr bool opencl = false;
#endif
#if defined(CANOPY_FOUNDRY_BUILD_CPU) && CANOPY_FOUNDRY_BUILD_CPU
    constexpr bool cpu = true;
#else
    constexpr bool cpu = false;
#endif
    return RuntimeCapabilities::from_build(cuda, opencl, cpu);
}
PlanResolution plan_request(const std::filesystem::path& path, RuntimeCapabilities capabilities) {
    return resolve_execution_plan(ida_native::load_request(path), std::move(capabilities));
}
LocalOperationResult check_local_run(const std::filesystem::path& path) {
    try {
        const auto request = ida_native::load_request(path);
        const auto resolution = resolve_execution_plan(request, build_capabilities());
        if (!resolution.plan.accepted)
            return {3, "Run blocked: " + resolution.plan.rejection};
        if (!request.parent.resume_from_checkpoint.empty()) {
            const auto checkpoint = inspect_checkpoint(request.parent.resume_from_checkpoint);
            if (!checkpoint.requires_native_validation)
                return {3, "Run blocked: " + checkpoint.message};
            return {4, "Run unavailable: the checkpoint needs native validation and the local run service is not connected."};
        }
        return {4, "Run unavailable: the local owned-run service is not connected; check Help for supported paths."};
    } catch (const std::exception& error) {
        return {1, std::string("Run check failed: ") + error.what()};
    }
}
LocalOperationResult cancel_local_run(const std::string& run_id) {
    // Implemented below through the same owner used by start_local_run.
    return default_local_run_service().cancel(run_id);
}

LocalRunService::LocalRunService(std::unique_ptr<OwnedRunBackend> backend)
    : backend_(std::move(backend)) {}

LocalOperationResult LocalRunService::accept_observation(OwnedRunObservation observation) {
    if (observation.run_id.empty() || (!run_id_.empty() && observation.run_id != run_id_)) {
        uncertain_ = true;
        return {4, "Run outcome unknown: owned-run identity could not be verified."};
    }
    run_id_ = std::move(observation.run_id);
    switch (observation.state) {
    case OwnedRunState::running:
        return {0, "Run " + run_id_ + " is active. " + observation.message, run_id_};
    case OwnedRunState::stopped:
        occupied_ = false;
        stopped_ = true;
        return {0, "Owned job " + run_id_ + " stopped. Training outcome requires a separate receipt. " + observation.message, run_id_, true};
    case OwnedRunState::unknown:
        uncertain_ = true;
        return {4, "Run " + run_id_ + " outcome unknown. " + observation.message, run_id_};
    }
    uncertain_ = true;
    return {4, "Run outcome unknown: unrecognized owner state."};
}

LocalOperationResult LocalRunService::start(const std::filesystem::path& path) {
    if (occupied_ || uncertain_)
        return {4, "Run unavailable: the previous owned run has not been reconciled."};
    if (!backend_) return check_local_run(path);
    try {
        const auto request = ida_native::load_request(path);
        const auto resolution = resolve_execution_plan(request, build_capabilities());
        if (!resolution.plan.accepted) return {3, "Run blocked: " + resolution.plan.rejection};
        const auto binding_error = execution_plan_binding_error(request, resolution);
        if (!binding_error.empty()) return {3, "Run blocked: " + binding_error};
        if (!request.parent.resume_from_checkpoint.empty()) {
            const auto checkpoint = inspect_checkpoint(request.parent.resume_from_checkpoint);
            if (!checkpoint.requires_native_validation) return {3, "Run blocked: " + checkpoint.message};
        }
        // Mark ownership before calling the backend: an exception may happen
        // after a child is created, so it cannot safely free the slot.
        occupied_ = true;
        stopped_ = false;
        run_id_.clear();
        try { return accept_observation(backend_->start(request, resolution)); }
        catch (...) {
            uncertain_ = true;
            return {4, "Run outcome unknown: the owned backend failed during launch."};
        }
    } catch (const std::exception& error) {
        return {1, std::string("Run check failed: ") + error.what()};
    }
}

LocalOperationResult LocalRunService::current() {
    if (!backend_) return {4, "Status unavailable: no local run service is connected."};
    if (uncertain_) return {4, "Run outcome unknown: the owned run requires reconciliation.", run_id_};
    if (run_id_.empty()) return {4, "Status unavailable: no local owned run."};
    return status(run_id_);
}

LocalOperationResult LocalRunService::status(const std::string& run_id) {
    if (!backend_) return {4, "Status unavailable: no local run service is connected."};
    if (stopped_ && !uncertain_ && !run_id.empty() && run_id == run_id_)
        return {0, "Owned job " + run_id_ + " stopped. Training outcome requires a separate receipt.", run_id_, true};
    if (!occupied_ || uncertain_ || run_id.empty() || run_id != run_id_)
        return {4, "Status unavailable: no matching verified owned run."};
    try { return accept_observation(backend_->observe()); }
    catch (...) { uncertain_ = true; return {4, "Run outcome unknown: status could not be verified."}; }
}

LocalOperationResult LocalRunService::cancel(const std::string& run_id) {
    if (!backend_) return {4, "Cancel unavailable: no local run service is connected."};
    if (stopped_ && !uncertain_ && !run_id.empty() && run_id == run_id_)
        return status(run_id);
    if (!occupied_ || uncertain_ || run_id.empty() || run_id != run_id_)
        return {4, "Cancel unavailable: no matching verified owned run."};
    try {
        const auto observation = backend_->cancel(std::chrono::seconds(3));
        auto result = accept_observation(observation);
        if (observation.state == OwnedRunState::running && result.exit_code == 0) {
            result.exit_code = 4;
            result.message = "Cancellation pending; " + result.message;
        }
        return result;
    }
    catch (...) { uncertain_ = true; return {4, "Run outcome unknown: cancellation could not be verified."}; }
}

namespace {
LocalRunService& default_local_run_service() {
    static LocalRunService service;
    return service;
}
}
LocalOperationResult start_local_run(const std::filesystem::path& path) {
    return default_local_run_service().start(path);
}
LocalOperationResult status_local_run(const std::string& run_id) {
    return default_local_run_service().status(run_id);
}
}
