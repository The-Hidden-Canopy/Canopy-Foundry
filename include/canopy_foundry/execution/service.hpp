#pragma once

#include "canopy_foundry/execution/plan.hpp"
#include <chrono>
#include <filesystem>
#include <memory>

namespace canopy_foundry {
// A build inventory is not hardware verification. Both entry points use the
// same planner and capability snapshot; execution retains its own admission.
struct CheckpointInspection {
    bool requires_native_validation{false};
    std::string message;
};
// Shared frontend result. This preflight never starts a process or creates a
// receipt; exit 3 is rejected, exit 4 means execution remains unavailable.
struct LocalOperationResult {
    int exit_code{4};
    std::string message;
    std::string run_id;
    bool job_stopped{false};
};
// Read-only, bounded header/presence checks only; never admission to resume.
CheckpointInspection inspect_checkpoint(const std::filesystem::path& directory);
RuntimeCapabilities build_capabilities();
PlanResolution plan_request(const std::filesystem::path& request_path,
                            RuntimeCapabilities capabilities);
LocalOperationResult check_local_run(const std::filesystem::path& request_path);
LocalOperationResult cancel_local_run(const std::string& run_id);

// Lifecycle observations are local process ownership facts, not training
// receipts. In particular, stopped does not mean a successful training run.
enum class OwnedRunState { running, stopped, unknown };
struct OwnedRunObservation {
    OwnedRunState state{OwnedRunState::unknown};
    std::string run_id;
    std::string message;
};

// An implementation must retain ownership and independently attest runtime,
// checkpoint, request/plan binding and output isolation before starting. It
// must not return stopped unless its containment can verify the owned job is
// empty. External descendants and Hub authority are outside this contract.
class OwnedRunBackend {
public:
    virtual ~OwnedRunBackend() = default;
    virtual OwnedRunObservation start(const ida_native::NativeRequest& request,
                                      const PlanResolution& resolution) = 0;
    virtual OwnedRunObservation observe() = 0;
    virtual OwnedRunObservation cancel(std::chrono::milliseconds timeout) = 0;
};

// One retained owner for one local run. Calls must be serialized by its owner.
// This is not cross-process transport;
// a durable owner/control channel is required before CLI execution is enabled.
// A null backend is the production default until containment is verified.
class LocalRunService {
public:
    explicit LocalRunService(std::unique_ptr<OwnedRunBackend> backend = {});
    LocalOperationResult start(const std::filesystem::path& request_path);
    LocalOperationResult current();
    LocalOperationResult status(const std::string& run_id);
    LocalOperationResult cancel(const std::string& run_id);
    // An unknown launch or cleanup is not quiescent.
    bool may_shutdown() const noexcept { return !occupied_ && !uncertain_; }
private:
    LocalOperationResult accept_observation(OwnedRunObservation observation);
    std::unique_ptr<OwnedRunBackend> backend_;
    bool occupied_{false};
    bool uncertain_{false};
    bool stopped_{false};
    std::string run_id_;
};
// Both frontends use the same default service; no backend is installed here.
LocalOperationResult start_local_run(const std::filesystem::path& request_path);
LocalOperationResult status_local_run(const std::string& run_id);
}
