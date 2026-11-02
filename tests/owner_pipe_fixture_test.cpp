#include "canopy_foundry/execution/owner_pipe.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Synthetic transport fixture only: no trainer, receipt, GPU, network or
// actual production process containment is established by this test.
namespace ida_native {
NativeRequest load_request(const std::filesystem::path&) { return {}; }
NativeRequest load_request(const std::filesystem::path& path, std::string& source_bytes) {
    source_bytes = "synthetic request";
    return load_request(path);
}
NativeRequest load_request_bounded(const std::filesystem::path& path, std::string& source_bytes, std::size_t maximum_bytes) {
    assert(maximum_bytes == (2u << 20));
    return load_request(path, source_bytes);
}
}
namespace canopy_foundry {
RuntimeCapabilities RuntimeCapabilities::from_build(bool, bool, bool) { return {}; }
PlanResolution resolve_execution_plan(const ida_native::NativeRequest&, RuntimeCapabilities capabilities) {
    PlanResolution result{}; result.plan.accepted = true; result.capabilities = std::move(capabilities); return result;
}
std::string execution_plan_binding_error(const ida_native::NativeRequest&, const PlanResolution&) { return {}; }
CheckpointInspection inspect_checkpoint(const std::filesystem::path&) { return {}; }
}
struct FakeOwner final : canopy_foundry::OwnedRunBackend {
    canopy_foundry::OwnedRunObservation start(const ida_native::NativeRequest&, const canopy_foundry::PlanResolution&) override {
        return {canopy_foundry::OwnedRunState::running, "fixture-1", "synthetic owner"};
    }
    canopy_foundry::OwnedRunObservation observe() override {
        return {canopy_foundry::OwnedRunState::running, "fixture-1", "synthetic owner"};
    }
    canopy_foundry::OwnedRunObservation cancel(std::chrono::milliseconds) override {
        return {canopy_foundry::OwnedRunState::stopped, "fixture-1", "synthetic owner"};
    }
};

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

std::wstring executable_path() {
    std::vector<wchar_t> path(32768);
    const DWORD count = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    assert(count && count < path.size());
    return std::wstring(path.data(), count);
}
void run_child(canopy_foundry::OwnerHost& host,
               const std::filesystem::path& endpoint_file, const std::filesystem::path& request_file,
               const std::wstring& command) {
    const auto exe = executable_path();
    std::wstring args = L"\"" + exe + L"\" --client \"" + endpoint_file.wstring() + L"\" " + command +
                        L" \"" + request_file.wstring() + L"\"";
    STARTUPINFOW startup{}; startup.cb = sizeof(startup);
    PROCESS_INFORMATION child{};
    assert(CreateProcessW(exe.c_str(), args.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                          nullptr, nullptr, &startup, &child));
    CloseHandle(child.hThread);
    std::string error;
    const auto result = host.serve_once(std::chrono::seconds(8), error);
    const auto waited = WaitForSingleObject(child.hProcess, 8000);
    if (waited == WAIT_TIMEOUT) {
        // The test owns this exact child handle; no PID-based cleanup.
        TerminateProcess(child.hProcess, 1);
        WaitForSingleObject(child.hProcess, 3000);
    }
    DWORD exit_code = 1;
    const bool got_exit = GetExitCodeProcess(child.hProcess, &exit_code) != FALSE;
    CloseHandle(child.hProcess);
    assert(command == L"wrong-owner" ? result != canopy_foundry::OwnerServeResult::handled
                                      : result == canopy_foundry::OwnerServeResult::handled);
    assert(waited == WAIT_OBJECT_0 && got_exit && exit_code == 0);
}

int wmain(int argc, wchar_t** argv) {
    using namespace canopy_foundry;
    if (argc == 5 && std::wstring(argv[1]) == L"--client") {
        const std::wstring command(argv[3]);
        if (command == L"current-env") {
            const auto result = status_local_run("");
            return result.exit_code == 0 && result.run_id == "fixture-1" && !result.job_stopped ? 0 : 1;
        }
        const bool cancel = command == L"cancel";
        const bool current = command == L"current" || command == L"current-stopped";
        if (command == L"wrong-owner") {
            OwnerEndpoint endpoint; std::string error;
            if (!load_owner_endpoint(std::filesystem::path(argv[2]), endpoint, error)) return 2;
            ++endpoint.process_id;
            return owner_pipe_call(endpoint, OwnerCommand::status, "fixture-1", std::chrono::seconds(5)).exit_code == 4 ? 0 : 1;
        }
        const auto result = command == L"start"
            ? owner_file_start(std::filesystem::path(argv[2]), std::filesystem::path(argv[4]), std::chrono::seconds(5))
            : owner_file_call(std::filesystem::path(argv[2]), current ? OwnerCommand::current :
                              cancel ? OwnerCommand::cancel : OwnerCommand::status,
                              current ? "" : "fixture-1", std::chrono::seconds(5));
        return result.exit_code == 0 && result.run_id == "fixture-1" &&
               result.job_stopped == (cancel || command == L"current-stopped") ? 0 : 1;
    }
    assert(argc == 2);
    const auto fixture_directory = std::filesystem::absolute(argv[1]);
    std::filesystem::create_directories(fixture_directory);
    const auto endpoint_file = fixture_directory / (L"owner-\x03a9-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
                                                    std::to_wstring(GetTickCount64()) + L".json");
    const auto request_file = fixture_directory / L"synthetic-request.json";
    OwnerHost host(std::make_unique<FakeOwner>());
    std::string error;
    assert(host.open(endpoint_file, error));
    assert(host.may_shutdown());
    OwnerEndpoint endpoint;
    assert(load_owner_endpoint(endpoint_file, endpoint, error));
    assert(!save_owner_endpoint(endpoint_file, endpoint, error)); // never replace an existing endpoint
    assert(owner_pipe_call(endpoint, OwnerCommand::cancel, "bad/id", std::chrono::milliseconds(1)).exit_code == 4);
    run_child(host, endpoint_file, request_file, L"wrong-owner");
    run_child(host, endpoint_file, request_file, L"start");
    assert(!host.may_shutdown());
    run_child(host, endpoint_file, request_file, L"current");
    assert(SetEnvironmentVariableW(L"CANOPY_FOUNDRY_OWNER_ENDPOINT", endpoint_file.c_str()));
    run_child(host, endpoint_file, request_file, L"current-env");
    assert(SetEnvironmentVariableW(L"CANOPY_FOUNDRY_OWNER_ENDPOINT", nullptr));
    // No host is serving here: Check Run must stay a client-only check.
    assert(_wputenv_s(L"CANOPY_FOUNDRY_OWNER_ENDPOINT", endpoint_file.c_str()) == 0);
    const auto check = check_local_run(request_file);
    assert(check.exit_code == 4 && check.message.find("configured owner has not been checked") != std::string::npos);
    assert(_wputenv_s(L"CANOPY_FOUNDRY_OWNER_ENDPOINT", L"") == 0);
    run_child(host, endpoint_file, request_file, L"status");
    run_child(host, endpoint_file, request_file, L"cancel");
    assert(host.may_shutdown());
    run_child(host, endpoint_file, request_file, L"current-stopped");
    assert(host.current().job_stopped);
    { std::ofstream corrupt(endpoint_file, std::ios::binary | std::ios::trunc); corrupt << "{"; }
    OwnerEndpoint restored;
    assert(!load_owner_endpoint(endpoint_file, restored, error));
    assert(std::filesystem::remove(endpoint_file));
    return 0;
}
#else
int main() { return 0; }
#endif
