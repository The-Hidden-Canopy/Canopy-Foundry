#pragma once

#include "canopy_foundry/execution/service.hpp"
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace canopy_foundry {

// Local transport for a retained owner process. The endpoint is a capability:
// keep it in operator-private storage, not in a Hub event or public receipt.
// The process creation time binds it to one owner incarnation, not merely a PID.
struct OwnerEndpoint {
    std::wstring pipe_name;
    std::uint32_t process_id{0};
    std::uint64_t process_creation_time{0};
};
enum class OwnerCommand { current, status, cancel };
enum class OwnerServeResult { handled, timeout, error };

// No production code creates this transport until the containment and runtime
// gates pass. One owner serially serves a retained LocalRunService; the service
// and its owned backend must outlive this pipe. Never use it for Hub authority.
class OwnerPipe {
public:
    OwnerPipe();
    ~OwnerPipe();
    OwnerPipe(const OwnerPipe&) = delete;
    OwnerPipe& operator=(const OwnerPipe&) = delete;
    bool open(OwnerEndpoint& endpoint, std::string& error);
    OwnerServeResult serve_once(LocalRunService& owner,
                                std::chrono::milliseconds timeout,
                                std::string& error);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The client verifies the server process ID and creation time against the
// endpoint before sending anything. Timeout or broken transport means unknown,
// never a completed cancellation.
LocalOperationResult owner_pipe_call(const OwnerEndpoint& endpoint,
                                     OwnerCommand command,
                                     const std::string& run_id,
                                     std::chrono::milliseconds timeout);
// Start passes a local absolute request path to the retained owner. The owner
// reloads and admits it; a lost response is unknown, never retried implicitly.
LocalOperationResult owner_pipe_start(const OwnerEndpoint& endpoint,
                                      const std::filesystem::path& request,
                                      std::chrono::milliseconds timeout);

// The caller supplies a trusted, operator-private directory. Creation is
// exclusive and never replaces an existing file. Reading validates the file's
// owner and restrictive ACL; neither function turns an endpoint into a receipt.
bool save_owner_endpoint(const std::filesystem::path& file, const OwnerEndpoint& endpoint, std::string& error);
bool load_owner_endpoint(const std::filesystem::path& file, OwnerEndpoint& endpoint, std::string& error);
LocalOperationResult owner_file_call(const std::filesystem::path& file,
                                     OwnerCommand command, const std::string& run_id,
                                     std::chrono::milliseconds timeout);
LocalOperationResult owner_file_start(const std::filesystem::path& file,
                                       const std::filesystem::path& request,
                                       std::chrono::milliseconds timeout);

// Host-side bootstrap: a future owner process keeps this object alive and
// repeatedly calls serve_once. open() publishes an endpoint before accepting
// work, so a lost start response can be reconciled via OwnerCommand::current.
// No backend is installed by default and this class never launches a daemon.
class OwnerHost {
public:
    explicit OwnerHost(std::unique_ptr<OwnedRunBackend> backend = {});
    bool open(const std::filesystem::path& endpoint_file, std::string& error);
    OwnerServeResult serve_once(std::chrono::milliseconds timeout, std::string& error);
    LocalOperationResult current();
    bool may_shutdown() const noexcept;
    // stop_requested must be a nonthrowing, local signal. A stop request never
    // closes a running or uncertain owner; the loop continues until verified
    // quiescence. The caller must keep the process alive while it runs.
    OwnerServeResult serve_until_quiescent(bool (*stop_requested)(void*) noexcept, void* context,
                                           std::chrono::milliseconds poll_interval, std::string& error);
private:
    LocalRunService owner_;
    OwnerPipe pipe_;
    bool ready_{false};
};
}
