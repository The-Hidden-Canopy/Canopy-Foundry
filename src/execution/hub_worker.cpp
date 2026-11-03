#include "canopy_foundry/execution/hub_worker.hpp"

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace canopy_foundry {
namespace {
constexpr std::string_view report_schema = "canopy_foundry.worker_once_report.v1";
constexpr std::uintmax_t report_limit = 64u << 10;
const std::filesystem::path worker_relative = std::filesystem::path("scripts") / "neural_forge_worker.py";

bool environment_set(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    std::size_t length = 0;
    const bool present = _dupenv_s(&value, &length, name) == 0 && value != nullptr && value[0] != '\0';
    std::free(value);
    return present;
#else
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
#endif
}

std::filesystem::path environment_path(const char* name) {
#ifdef _WIN32
    const std::wstring wide(name, name + std::char_traits<char>::length(name));
    const DWORD length = GetEnvironmentVariableW(wide.c_str(), nullptr, 0);
    if (length == 0) return {};
    std::wstring value(length, L'\0');
    const DWORD written = GetEnvironmentVariableW(wide.c_str(), value.data(), length);
    value.resize(written);
    return std::filesystem::path(value);
#else
    const char* value = std::getenv(name);
    return value ? std::filesystem::path(value) : std::filesystem::path();
#endif
}

#ifdef _WIN32
std::wstring quote(const std::wstring& value) {
    std::wstring result = L"\"";
    std::size_t slashes = 0;
    for (const auto ch : value) {
        if (ch == L'\\') { ++slashes; continue; }
        result.append(ch == L'"' ? slashes * 2 + 1 : slashes, L'\\');
        result += ch; slashes = 0;
    }
    result.append(slashes * 2, L'\\'); result += L'"'; return result;
}

struct Handle {
    HANDLE value{nullptr};
    explicit Handle(HANDLE handle = nullptr) : value(handle == INVALID_HANDLE_VALUE ? nullptr : handle) {}
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    ~Handle() { if (value) CloseHandle(value); }
};

int spawn_and_wait(const HubWorkerLaunch& launch) {
    // Kill-on-close keeps the worker and its trainer inside this frontend's ownership.
    Handle job(CreateJobObjectW(nullptr, nullptr));
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!job.value || !SetInformationJobObject(job.value, JobObjectExtendedLimitInformation, &limits, sizeof(limits)))
        throw std::runtime_error("worker containment unavailable");
    SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
    Handle input(CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_EXISTING, 0, nullptr));
    Handle log(CreateFileW(launch.log_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &inherit, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr));
    if (!input.value || !log.value) throw std::runtime_error("worker log unavailable");
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = input.value;
    startup.hStdOutput = log.value;
    startup.hStdError = log.value;
    std::wstring command = quote(launch.python.wstring()) + L" " + quote((launch.foundry_root / worker_relative).wstring())
        + L" --once --once-report " + quote(launch.report_path.wstring());
    if (command.size() >= 32767) throw std::runtime_error("worker command exceeds Windows bounds");
    PROCESS_INFORMATION child{};
    // Worker settings come from the inherited environment.
    if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED | CREATE_NO_WINDOW,
                        nullptr, launch.foundry_root.c_str(), &startup, &child))
        throw std::runtime_error("the worker interpreter could not be started; set CANOPY_FOUNDRY_PYTHON");
    Handle process(child.hProcess);
    Handle thread(child.hThread);
    if (!AssignProcessToJobObject(job.value, process.value)) {
        TerminateProcess(process.value, 1);
        WaitForSingleObject(process.value, 3000);
        throw std::runtime_error("worker could not be assigned to containment");
    }
    if (ResumeThread(thread.value) == static_cast<DWORD>(-1)) {
        TerminateJobObject(job.value, 1);
        throw std::runtime_error("worker could not resume inside containment");
    }
    WaitForSingleObject(process.value, INFINITE);
    DWORD code = 1;
    if (!GetExitCodeProcess(process.value, &code)) code = 1;
    return static_cast<int>(code);
}
#else
int spawn_and_wait(const HubWorkerLaunch& launch) {
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    const std::string log = launch.log_path.string();
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
    const std::string python = launch.python.string();
    const std::string worker = (launch.foundry_root / worker_relative).string();
    const std::string report = launch.report_path.string();
    std::string once = "--once", once_report = "--once-report";
    char* argv[] = {const_cast<char*>(python.c_str()), const_cast<char*>(worker.c_str()), once.data(),
                    once_report.data(), const_cast<char*>(report.c_str()), nullptr};
    pid_t child = 0;
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 29))
    const std::string root = launch.foundry_root.string();
    posix_spawn_file_actions_addchdir_np(&actions, root.c_str());
    const int spawned = posix_spawnp(&child, python.c_str(), &actions, nullptr, argv, environ);
#else
    // Without a spawn-time chdir the worker path is absolute, but its default
    // relative roots follow this process's working directory.
    const auto previous = std::filesystem::current_path();
    std::filesystem::current_path(launch.foundry_root);
    const int spawned = posix_spawnp(&child, python.c_str(), &actions, nullptr, argv, environ);
    std::filesystem::current_path(previous);
#endif
    posix_spawn_file_actions_destroy(&actions);
    if (spawned != 0) throw std::runtime_error("the worker interpreter could not be started; set CANOPY_FOUNDRY_PYTHON");
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) throw std::runtime_error("worker exit status unavailable");
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
#endif
}  // namespace

HubWorkerLaunch hub_worker_launch_for(const std::filesystem::path& executable) {
    HubWorkerLaunch launch;
    auto root = environment_path("CANOPY_FOUNDRY_ROOT");
    if (root.empty()) {
        for (auto candidate = std::filesystem::absolute(executable).parent_path(); !candidate.empty();
             candidate = candidate.parent_path()) {
            if (std::filesystem::is_regular_file(candidate / worker_relative)) { root = candidate; break; }
            if (candidate == candidate.root_path()) break;
        }
    }
    if (root.empty() || !std::filesystem::is_regular_file(root / worker_relative))
        throw std::runtime_error("the worker was not found; set CANOPY_FOUNDRY_ROOT to the Foundry checkout");
    launch.foundry_root = std::filesystem::absolute(root).lexically_normal().make_preferred();
    launch.python = environment_path("CANOPY_FOUNDRY_PYTHON");
#ifdef _WIN32
    if (launch.python.empty()) launch.python = "python";
#else
    if (launch.python.empty()) launch.python = "python3";
#endif
    const auto directory = launch.foundry_root / "runs" / "hub-next";
    launch.report_path = directory / "once-report.json";
    launch.log_path = directory / "worker.log";
    return launch;
}

HubWorkerLaunch hub_worker_launch_for_this_process() {
#ifdef _WIN32
    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) throw std::runtime_error("the Foundry executable path is unavailable");
    buffer.resize(length);
    return hub_worker_launch_for(std::filesystem::path(buffer));
#else
    std::error_code ec;
    const auto self = std::filesystem::read_symlink("/proc/self/exe", ec);
    return hub_worker_launch_for(ec ? std::filesystem::current_path() / "foundry" : self);
#endif
}

std::string hub_worker_configuration_problem() {
    for (const char* name : {"NEURAL_FORGE_HUB_URL", "NEURAL_FORGE_WORKER_TOKEN", "NEURAL_FORGE_ORG"}) {
        if (!environment_set(name))
            return std::string("Run next job unavailable: ") + name + " is not set for the worker.";
    }
    return {};
}

LocalOperationResult interpret_hub_worker_report(int worker_exit_code, const std::filesystem::path& report_path) {
    if (worker_exit_code != 0)
        return {1, "Worker exited with code " + std::to_string(worker_exit_code) + "; see the worker log."};
    std::error_code ec;
    const auto size = std::filesystem::file_size(report_path, ec);
    if (ec || size == 0 || size > report_limit)
        return {4, "Worker finished without a readable report; the outcome is unknown."};
    std::ifstream in(report_path, std::ios::binary);
    const std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto report = nlohmann::json::parse(data, nullptr, false);
    if (!report.is_object() || report.value("schema", "") != report_schema || !report.contains("claimed")
        || !report["claimed"].is_boolean())
        return {4, "Worker report is malformed; the outcome is unknown."};
    if (!report["claimed"].get<bool>())
        return {4, "No job is queued for this worker. Nothing was started."};
    const std::string run_id = report.contains("run_id") && report["run_id"].is_string() ? report["run_id"].get<std::string>() : "";
    if (run_id.empty())
        return {4, "Worker reported a job without a run ID; the outcome is unknown."};
    if (!report.contains("hub_status") || !report["hub_status"].is_string())
        return {4, "Job " + run_id + " ran, but its final status was not confirmed.", run_id};
    const auto status = report["hub_status"].get<std::string>();
    if (status == "succeeded")
        return {0, "Job " + run_id + " succeeded. Evaluation and promotion remain separate.", run_id, true};
    if (status == "failed" || status == "cancelled")
        return {1, "Job " + run_id + " ended " + status + ".", run_id, true};
    return {4, "Job " + run_id + " is " + status + "; it is not finished.", run_id};
}

LocalOperationResult run_next_hub_job(const HubWorkerLaunch& launch) {
    if (const auto problem = hub_worker_configuration_problem(); !problem.empty()) return {4, problem};
    try {
        std::filesystem::create_directories(launch.report_path.parent_path());
        std::error_code ec;
        std::filesystem::remove(launch.report_path, ec);  // never read a previous attempt's report
        if (ec) return {4, "A previous worker report could not be cleared; nothing was started."};
        const int code = spawn_and_wait(launch);
        auto result = interpret_hub_worker_report(code, launch.report_path);
        result.message += " Worker log: " + launch.log_path.string();
        return result;
    } catch (const std::exception& error) {
        return {4, std::string("Run next job could not start the worker: ") + error.what()};
    }
}
}
