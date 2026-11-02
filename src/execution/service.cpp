#include "canopy_foundry/execution/service.hpp"
#if defined(CANOPY_FOUNDRY_LOCAL_OWNER_TRANSPORT) && CANOPY_FOUNDRY_LOCAL_OWNER_TRANSPORT
#include "canopy_foundry/execution/owner_pipe.hpp"
#include <cstdlib>
#endif
#include <exception>
#include <utility>

namespace canopy_foundry {
namespace {
LocalRunService& default_local_run_service();
constexpr std::size_t local_request_limit = 2u << 20;
}
bool local_owner_transport_enabled() noexcept {
#if defined(CANOPY_FOUNDRY_LOCAL_OWNER_TRANSPORT) && CANOPY_FOUNDRY_LOCAL_OWNER_TRANSPORT
    return true;
#else
    return false;
#endif
}
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
#if defined(CANOPY_FOUNDRY_LOCAL_OWNER_TRANSPORT) && CANOPY_FOUNDRY_LOCAL_OWNER_TRANSPORT
        const auto* endpoint = _wgetenv(L"CANOPY_FOUNDRY_OWNER_ENDPOINT");
        const bool configured_owner = endpoint && *endpoint;
#else
        constexpr bool configured_owner = false;
#endif
        std::string admitted_bytes;
        const auto request = ida_native::load_request_bounded(path, admitted_bytes, local_request_limit);
        const auto resolution = resolve_execution_plan(request, build_capabilities());
        if (!resolution.plan.accepted) {
            if (configured_owner)
                return {3, "Client plan blocked: " + resolution.plan.rejection +
                           ". The configured owner has not been checked and may use different capabilities."};
            return {3, "Run blocked: " + resolution.plan.rejection};
        }
        if (!request.parent.resume_from_checkpoint.empty()) {
            const auto checkpoint = inspect_checkpoint(request.parent.resume_from_checkpoint);
            if (!checkpoint.requires_native_validation)
                return {3, "Run blocked: " + checkpoint.message};
            if (configured_owner)
                return {4, "Client plan accepted; checkpoint needs native validation. The configured owner has not been checked."};
            return {4, "Run unavailable: the checkpoint needs native validation and the local run service is not connected."};
        }
        if (configured_owner)
            return {4, "Client plan accepted; the configured owner has not been checked. Check Run does not start a run."};
        return {4, "Run unavailable: the local owned-run service is not connected; check Help for supported paths."};
    } catch (const std::exception& error) {
        return {1, std::string("Run check failed: ") + error.what()};
    }
}
LocalOperationResult cancel_local_run(const std::string& run_id) {
#if defined(CANOPY_FOUNDRY_LOCAL_OWNER_TRANSPORT) && CANOPY_FOUNDRY_LOCAL_OWNER_TRANSPORT
    if (const auto* endpoint = _wgetenv(L"CANOPY_FOUNDRY_OWNER_ENDPOINT"); endpoint && *endpoint) {
        try { return owner_file_call(std::filesystem::path(endpoint), OwnerCommand::cancel, run_id, std::chrono::seconds(5)); }
        catch (...) { return {4, "Owner endpoint unavailable or unverified; reconcile the retained owner."}; }
    }
#endif
    // Implemented below through the same owner used by start_local_run.
    return default_local_run_service().cancel(run_id);
}

LocalRunService::LocalRunService(std::unique_ptr<OwnedRunBackend> backend, RuntimeCapabilities capabilities)
    : backend_(std::move(backend)), capabilities_(std::move(capabilities)) {}

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
        terminal_message_ = "Owned run " + run_id_ + " is no longer active. Training outcome requires a separate receipt. " + observation.message;
        return {0, terminal_message_, run_id_, true};
    case OwnedRunState::unknown:
    case OwnedRunState::rejected: // rejection after launch is not quiescence
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
        std::string admitted_bytes;
        // Match the backend's existing 2 MiB pinned-source limit before the
        // serialized owner allocates and parses the request.
        const auto request = ida_native::load_request_bounded(path, admitted_bytes, local_request_limit);
        auto resolution = resolve_execution_plan(request, capabilities_);
        resolution.local_source_request_bytes = std::move(admitted_bytes);
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
        terminal_message_.clear();
        try {
            auto observation = backend_->start(request, resolution);
            if (observation.state == OwnedRunState::rejected) {
                occupied_ = false;
                return {3, "Run blocked before process creation: " + observation.message};
            }
            return accept_observation(std::move(observation));
        }
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
    if (uncertain_) {
        // Explicit current is reconciliation through the retained backend's
        // exact handles, never a new launch or a PID-based guess.
        try {
            auto observation = backend_->observe();
            if (observation.run_id.empty() || (!run_id_.empty() && observation.run_id != run_id_) ||
                (observation.state != OwnedRunState::running && observation.state != OwnedRunState::stopped))
                return {4, "Run outcome unknown: " + observation.message, run_id_};
            uncertain_ = false;
            occupied_ = true;
            return accept_observation(std::move(observation));
        } catch (...) { return {4, "Run outcome unknown: reconciliation could not verify ownership.", run_id_}; }
    }
    if (run_id_.empty()) return {4, "Status unavailable: no local owned run."};
    return status(run_id_);
}

LocalOperationResult LocalRunService::status(const std::string& run_id) {
    if (!backend_) return {4, "Status unavailable: no local run service is connected."};
    if (stopped_ && !uncertain_ && !run_id.empty() && run_id == run_id_)
        return {0, terminal_message_, run_id_, true};
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
#if defined(CANOPY_FOUNDRY_LOCAL_OWNER_TRANSPORT) && CANOPY_FOUNDRY_LOCAL_OWNER_TRANSPORT
    if (const auto* endpoint = _wgetenv(L"CANOPY_FOUNDRY_OWNER_ENDPOINT"); endpoint && *endpoint) {
        try { return owner_file_start(std::filesystem::path(endpoint), std::filesystem::absolute(path), std::chrono::seconds(5)); }
        catch (...) { return {4, "Owner endpoint unavailable or unverified; reconcile the retained owner."}; }
    }
#endif
    return default_local_run_service().start(path);
}
LocalOperationResult status_local_run(const std::string& run_id) {
#if defined(CANOPY_FOUNDRY_LOCAL_OWNER_TRANSPORT) && CANOPY_FOUNDRY_LOCAL_OWNER_TRANSPORT
    if (const auto* endpoint = _wgetenv(L"CANOPY_FOUNDRY_OWNER_ENDPOINT"); endpoint && *endpoint) {
        try { return owner_file_call(std::filesystem::path(endpoint), run_id.empty() ? OwnerCommand::current : OwnerCommand::status,
                                     run_id, std::chrono::seconds(5)); }
        catch (...) { return {4, "Owner endpoint unavailable or unverified; reconcile the retained owner."}; }
    }
#endif
    return default_local_run_service().status(run_id);
}
}
