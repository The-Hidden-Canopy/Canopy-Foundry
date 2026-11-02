#include "canopy_foundry/execution/local_backend.hpp"
#include <nlohmann/json.hpp>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>
#include <string>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winioctl.h>
#endif

// This executable impersonates the pinned CLI protocol only in private test
// directories. It contains no CUDA/trainer/Hub path and cannot prove learning.
#ifdef _WIN32
namespace {
using json = nlohmann::json;
std::filesystem::path self() {
    std::vector<wchar_t> buffer(32768);
    const auto size = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    assert(size && size < buffer.size()); return std::wstring(buffer.data(), size);
}
std::string bytes(const std::filesystem::path& file) {
    std::ifstream stream(file, std::ios::binary);
    assert(stream); return {std::istreambuf_iterator<char>(stream), {}};
}
void write(const std::filesystem::path& file, const std::string& value) {
    std::ofstream stream(file, std::ios::binary); stream << value; stream.close(); assert(stream);
}
void make_sparse(const std::filesystem::path& file) {
    HANDLE handle = CreateFileW(file.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    assert(handle != INVALID_HANDLE_VALUE);
    DWORD returned = 0;
    assert(DeviceIoControl(handle, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr));
    LARGE_INTEGER size{}; size.QuadPart = 4ll << 30;
    assert(SetFilePointerEx(handle, size, nullptr, FILE_BEGIN) && SetEndOfFile(handle));
    CloseHandle(handle);
}
int fixture_child(int argc, wchar_t** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::wstring(argv[i]) == L"--linger") { Sleep(60000); return 0; }
        if (std::wstring(argv[i]) == L"--self-check") {
            if (std::filesystem::exists(self().parent_path() / "probe-pauses")) {
                write(self().parent_path() / "probe-entered", "synthetic fixture");
                for (int i = 0; i < 3000 && !std::filesystem::exists(self().parent_path() / "probe-release"); ++i) Sleep(10);
                if (!std::filesystem::exists(self().parent_path() / "probe-release")) return 8;
            }
            if (std::filesystem::exists(self().parent_path() / "probe-waits")) {
                write(self().parent_path() / "probe-entered", "synthetic fixture");
                Sleep(60000);
            }
            return std::filesystem::exists(self().parent_path() / "probe-fails") ? 7 : 0;
        }
    }
    assert(argc >= 3 && std::wstring(argv[1]) == L"--request-json");
    const auto request = json::parse(bytes(argv[2]));
    const auto output = std::filesystem::u8path(request.at("output_dir").get<std::string>());
    const auto exe = self();
    if (std::filesystem::exists(exe.parent_path() / "trainer-crashes")) {
        write(output / "status.json", "{malformed");
        std::filesystem::create_directory(output / "partial-checkpoint");
        write(output / "partial-checkpoint/model.safetensors", "partial");
        return 9;
    }
    std::wstring command = L"\"" + exe.wstring() + L"\" --linger";
    STARTUPINFOW startup{}; startup.cb = sizeof(startup);
    PROCESS_INFORMATION child{};
    assert(CreateProcessW(exe.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &child));
    CloseHandle(child.hThread); CloseHandle(child.hProcess);
    // Give the test an identity-bound root HANDLE, never a PID-based kill.
    const auto owner_pid = static_cast<DWORD>(std::stoul(bytes(exe.parent_path() / "test-owner-pid")));
    HANDLE owner = OpenProcess(PROCESS_DUP_HANDLE, FALSE, owner_pid), duplicated = nullptr;
    assert(owner && DuplicateHandle(GetCurrentProcess(), GetCurrentProcess(), owner, &duplicated, SYNCHRONIZE, FALSE, 0));
    CloseHandle(owner);
    write(output / "synthetic-root-handle.tmp", std::to_string(reinterpret_cast<std::uintptr_t>(duplicated)));
    std::filesystem::rename(output / "synthetic-root-handle.tmp", output / "synthetic-root-handle.txt");
    return 0; // the descendant deliberately remains in the inherited job
}
struct Case {
    std::filesystem::path directory, request_path, run_root, binary, token;
    canopy_foundry::LocalBackendConfig config;
    Case(const std::filesystem::path& base, const char* name, const std::filesystem::path& repository) {
        directory = base / name; assert(std::filesystem::create_directory(directory));
        binary = directory / "ida_native_train.exe";
        assert(CopyFileW(self().c_str(), binary.c_str(), TRUE));
        write(directory / "test-owner-pid", std::to_string(GetCurrentProcessId()));
        token = directory / "tokens.bin";
        write(token, std::string(8192, '\0')); write(directory / "labels.bin", std::string(8192, '\0'));
        run_root = directory / "runs"; assert(std::filesystem::create_directory(run_root));
        auto payload = json::parse(bytes(repository / "configs/examples/planning_only_request.json"));
        payload["input"]["token_blocks"] = token.string();
        payload["input"]["label_blocks"] = (directory / "labels.bin").string();
        payload["output_dir"] = (directory / "must-not-be-used").string();
        payload["status_file"] = (directory / "must-not-be-used/status.json").string();
        request_path = directory / "request.json"; write(request_path, payload.dump(2));
        config.binary = binary; config.binary_sha256 = canopy_foundry::sha256_hex(bytes(binary));
        config.run_root = run_root; config.maximum_runtime = std::chrono::seconds(15);
    }
};
}
int wmain(int argc, wchar_t** argv) {
    if (argc >= 2 && std::wstring(argv[1]).rfind(L"--", 0) == 0) return fixture_child(argc, argv);
    assert(argc == 3);
    const auto base = std::filesystem::absolute(argv[1]) / (L"run-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(GetTickCount64()));
    std::filesystem::create_directories(base);
    const auto repository = std::filesystem::absolute(argv[2]);
    using namespace canopy_foundry;
    const auto capabilities = RuntimeCapabilities::from_build(true, false, false);

    Case uppercase_pin(base, "uppercase-pin", repository);
    for (char& ch : uppercase_pin.config.binary_sha256) if (ch >= 'a' && ch <= 'f') ch = static_cast<char>(ch - ('a' - 'A'));
    assert(make_local_native_backend(uppercase_pin.config)); // standard uppercase SHA-256 is accepted

    Case wrong_pin(base, "wrong-pin", repository);
    wrong_pin.config.binary_sha256 = std::string(64, '0');
    LocalRunService rejected(make_local_native_backend(wrong_pin.config), capabilities);
    const auto wrong_started = rejected.start(wrong_pin.request_path);
    assert(wrong_started.exit_code == 0);
    LocalOperationResult wrong_result;
    for (int i = 0; i < 300; ++i) { wrong_result = rejected.status(wrong_started.run_id); if (wrong_result.job_stopped) break; Sleep(10); }
    assert(wrong_result.job_stopped && wrong_result.message.find("before process creation") != std::string::npos);
    assert(rejected.may_shutdown());
    assert(std::filesystem::is_empty(wrong_pin.run_root));

    Case changed_request(base, "changed-request", repository);
    std::string before_bytes;
    const auto before = ida_native::load_request(changed_request.request_path, before_bytes);
    auto before_plan = resolve_execution_plan(before, capabilities);
    before_plan.local_source_request_bytes = before_bytes;
    auto changed = json::parse(bytes(changed_request.request_path));
    // Seed is deliberately omitted from the plan hash; plan equality alone
    // must not authorize changed request bytes.
    changed["seed"] = before.seed + 1;
    write(changed_request.request_path, changed.dump(2));
    const auto after = ida_native::load_request(changed_request.request_path);
    assert(execution_plan_sha256(before_plan.plan) == execution_plan_sha256(resolve_execution_plan(after, capabilities).plan));
    auto changed_backend = make_local_native_backend(changed_request.config);
    const auto changed_started = changed_backend->start(before, before_plan);
    assert(changed_started.state == OwnedRunState::running);
    OwnedRunObservation changed_result;
    for (int i = 0; i < 300; ++i) { changed_result = changed_backend->observe(); if (changed_result.state == OwnedRunState::stopped) break; Sleep(10); }
    assert(changed_result.state == OwnedRunState::stopped && changed_result.message.find("source request changed after admission") != std::string::npos);
    assert(std::filesystem::is_empty(changed_request.run_root));
    HANDLE editable = CreateFileW(changed_request.token.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                  nullptr, OPEN_EXISTING, 0, nullptr);
    assert(editable != INVALID_HANDLE_VALUE); CloseHandle(editable); // prelaunch rejection releases locks

    Case foreign(base, "foreign-contract", repository);
    auto foreign_payload = json::parse(bytes(foreign.request_path));
    foreign_payload["model"]["architecture_contract"] = "external_graph_v1";
    write(foreign.request_path, foreign_payload.dump(2));
    LocalRunService foreign_service(make_local_native_backend(foreign.config), capabilities);
    // The retained backend admits preparation asynchronously. A foreign model
    // must stop during preparation, before it creates a run directory or probe.
    assert(ida_native::load_request(foreign.request_path).model.architecture_contract == "external_graph_v1");
    const auto foreign_started = foreign_service.start(foreign.request_path);
    assert(foreign_started.exit_code == 0 && !foreign_started.run_id.empty());
    assert(!foreign_service.may_shutdown());
    LocalOperationResult foreign_result;
    for (int i = 0; i < 300; ++i) {
        foreign_result = foreign_service.status(foreign_started.run_id);
        if (foreign_result.job_stopped) break;
        Sleep(10);
    }
    assert(foreign_result.job_stopped && foreign_result.run_id == foreign_started.run_id);
    assert(foreign_result.message.find("before process creation") != std::string::npos);
    assert(foreign_result.message.find("local model layout does not match a supported public contract") != std::string::npos);
    assert(foreign_service.may_shutdown());
    assert(std::filesystem::is_empty(foreign.run_root));

    Case cancel_preparation(base, "cancel-preparation", repository);
    make_sparse(cancel_preparation.token);
    make_sparse(cancel_preparation.directory / "labels.bin");
    LocalRunService preparing(make_local_native_backend(cancel_preparation.config), capabilities);
    const auto preparing_run = preparing.start(cancel_preparation.request_path);
    assert(preparing_run.exit_code == 0 && preparing_run.message.find("preparation active") != std::string::npos);
    auto preparation_stop = preparing.cancel(preparing_run.run_id);
    for (int i = 0; i < 300 && !preparation_stop.job_stopped; ++i) { Sleep(10); preparation_stop = preparing.current(); }
    assert(preparation_stop.job_stopped && preparing.may_shutdown());
    assert(preparation_stop.message.find("before process creation") != std::string::npos);
    const auto preparation_dir = cancel_preparation.run_root / preparing_run.run_id;
    assert(!std::filesystem::exists(preparation_dir / "launch-provenance.json"));
    assert(!std::filesystem::exists(preparation_dir / "probe-stdout.txt"));

    Case cancel_probe(base, "cancel-probe", repository);
    write(cancel_probe.directory / "probe-waits", "synthetic fixture");
    LocalRunService probe(make_local_native_backend(cancel_probe.config), capabilities);
    const auto probing = probe.start(cancel_probe.request_path);
    assert(probing.exit_code == 0 && !probing.run_id.empty());
    assert(probing.message.find("training has not started") != std::string::npos);
    auto probe_stop = probe.cancel(probing.run_id);
    if (!probe_stop.job_stopped) {
        std::cerr << "probe cancellation initially fenced: " << probe_stop.message << "\n";
        assert(!probe.may_shutdown());
        assert(probe.start(cancel_probe.request_path).exit_code == 4);
        for (int i = 0; i < 300 && !probe_stop.job_stopped; ++i) {
            Sleep(10);
            probe_stop = probe.current();
            assert(probe_stop.run_id == probing.run_id);
        }
    }
    assert(probe_stop.job_stopped && probe.may_shutdown());
    assert(!std::filesystem::exists(cancel_probe.run_root / probing.run_id / "trainer-stdout.txt"));

    Case partial_record(base, "partial-record", repository);
    write(partial_record.directory / "probe-waits", "synthetic fixture");
    LocalRunService partial(make_local_native_backend(partial_record.config), capabilities);
    const auto partial_run = partial.start(partial_record.request_path);
    assert(partial_run.exit_code == 0);
    for (int i = 0; i < 300 && !std::filesystem::exists(partial_record.directory / "probe-entered"); ++i) Sleep(10);
    assert(std::filesystem::exists(partial_record.directory / "probe-entered"));
    const auto partial_dir = partial_record.run_root / partial_run.run_id;
    write(partial_dir / "probe-process.json", "{partial");
    const auto fenced = partial.cancel(partial_run.run_id);
    assert(fenced.exit_code == 4 && !partial.may_shutdown());
    assert(bytes(partial_dir / "probe-process.json") == "{partial");
    LocalOperationResult recovered;
    for (int i = 0; i < 300; ++i) { recovered = partial.current(); if (recovered.job_stopped) break; Sleep(10); }
    assert(recovered.job_stopped && partial.may_shutdown());
    assert(bytes(partial_dir / "probe-process.json") == "{partial");
    const auto recovery_record = json::parse(bytes(partial_dir / "probe-process-recovery-1.json"));
    assert(recovery_record["training_outcome"] == "unverified" && recovery_record["promotion_eligible"] == false);
    assert(!std::filesystem::exists(partial_dir / "trainer-stdout.txt"));

    Case setup_failure(base, "trainer-setup-failure", repository);
    write(setup_failure.directory / "probe-pauses", "synthetic fixture");
    LocalRunService setup(make_local_native_backend(setup_failure.config), capabilities);
    const auto setup_run = setup.start(setup_failure.request_path);
    assert(setup_run.exit_code == 0);
    for (int i = 0; i < 300 && !std::filesystem::exists(setup_failure.directory / "probe-entered"); ++i) Sleep(10);
    assert(std::filesystem::exists(setup_failure.directory / "probe-entered"));
    const auto setup_dir = setup_failure.run_root / setup_run.run_id;
    write(setup_dir / "trainer-stdout.txt", "preexisting synthetic conflict");
    write(setup_failure.directory / "probe-release", "synthetic fixture");
    LocalOperationResult setup_result;
    for (int i = 0; i < 300; ++i) { setup_result = setup.status(setup_run.run_id); if (setup_result.job_stopped) break; Sleep(10); }
    assert(setup_result.job_stopped && setup_result.message.find("trainer process was not created") != std::string::npos);
    assert(bytes(setup_dir / "trainer-stdout.txt") == "preexisting synthetic conflict");
    assert(!std::filesystem::exists(setup_dir / "trainer-process.json"));

    Case failed_probe(base, "failed-probe", repository);
    write(failed_probe.directory / "probe-fails", "synthetic fixture");
    LocalRunService failed(make_local_native_backend(failed_probe.config), capabilities);
    const auto failed_run = failed.start(failed_probe.request_path);
    assert(failed_run.exit_code == 0);
    LocalOperationResult failure;
    for (int i = 0; i < 300; ++i) { failure = failed.status(failed_run.run_id); if (failure.job_stopped) break; Sleep(10); }
    assert(failure.job_stopped && failure.message.find("self-check failed") != std::string::npos);
    assert(failed.status(failed_run.run_id).message == failure.message); // failure survives repeated refresh
    assert(!std::filesystem::exists(failed_probe.run_root / failed_run.run_id / "trainer-stdout.txt"));

    Case crashed_trainer(base, "crashed-trainer", repository);
    write(crashed_trainer.directory / "trainer-crashes", "synthetic fixture");
    LocalRunService crashed(make_local_native_backend(crashed_trainer.config), capabilities);
    const auto crash_run = crashed.start(crashed_trainer.request_path);
    assert(crash_run.exit_code == 0);
    LocalOperationResult crash_result;
    for (int i = 0; i < 300; ++i) { crash_result = crashed.status(crash_run.run_id); if (crash_result.job_stopped) break; Sleep(10); }
    assert(crash_result.job_stopped && crash_result.message.find("exited with code 9") != std::string::npos);
    const auto crash_directory = crashed_trainer.run_root / crash_run.run_id;
    const auto crash_record = json::parse(bytes(crash_directory / "trainer-process.json"));
    assert(crash_record["exit_code"] == 9 && crash_record["training_outcome"] == "unverified");
    assert(crash_record["promotion_eligible"] == false);
    assert(!inspect_checkpoint(crash_directory / "output/partial-checkpoint").requires_native_validation);
    assert(bytes(crash_directory / "output/status.json") == "{malformed");

    Case descendants(base, "descendants", repository);
    LocalRunService service(make_local_native_backend(descendants.config), capabilities);
    const auto started = service.start(descendants.request_path);
    assert(started.exit_code == 0 && !service.may_shutdown());
    assert(service.start(descendants.request_path).exit_code == 4);
    assert(service.cancel("different-run").exit_code == 4);
    const auto directory = descendants.run_root / started.run_id;
    const auto marker = directory / "output/synthetic-root-handle.txt";
    for (int i = 0; i < 300 && !std::filesystem::exists(marker); ++i) {
        assert(service.status(started.run_id).exit_code == 0); Sleep(10);
    }
    assert(std::filesystem::exists(marker));
    // The root handle was duplicated by this exact synthetic root before exit.
    const auto root = reinterpret_cast<HANDLE>(static_cast<std::uintptr_t>(std::stoull(bytes(marker))));
    assert(WaitForSingleObject(root, 3000) == WAIT_OBJECT_0); CloseHandle(root);
    assert(!service.status(started.run_id).job_stopped); // live descendant
    HANDLE overwrite = CreateFileW(descendants.token.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    assert(overwrite == INVALID_HANDLE_VALUE); // inputs stay frozen until job empties
    const auto stopped = service.cancel(started.run_id);
    assert(stopped.job_stopped && service.may_shutdown());
    assert(!std::filesystem::exists(descendants.directory / "must-not-be-used"));
    const auto provenance = json::parse(bytes(directory / "launch-provenance.json"));
    const auto result = json::parse(bytes(directory / "trainer-process.json"));
    assert(provenance["binary_sha256"] == descendants.config.binary_sha256);
    assert(provenance["inputs"].size() == 2 && result["job_empty"] == true);
    assert(result["cancel_requested"] == true && result["training_outcome"] == "unverified");
    assert(result["termination_requested"] == true && result["stop_reason"] == "operator");
    assert(result["promotion_eligible"] == false);
    return 0; // preserve private fixtures and diagnostics for review
}
#else
int main() { return 0; }
#endif
