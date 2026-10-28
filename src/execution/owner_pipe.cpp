#include "canopy_foundry/execution/owner_pipe.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>
#include <aclapi.h>
#include <sddl.h>
#endif

namespace canopy_foundry {
#ifdef _WIN32
namespace {
using json = nlohmann::json;
constexpr std::size_t max_message = 4096;
constexpr wchar_t pipe_prefix[] = L"\\\\.\\pipe\\CanopyFoundry-";

bool valid_timeout(std::chrono::milliseconds timeout) {
    return timeout.count() >= 0 && timeout <= std::chrono::seconds(30);
}
bool valid_id(const std::string& id) {
    return !id.empty() && id.size() <= 128 && std::all_of(id.begin(), id.end(), [](unsigned char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
               (ch >= '0' && ch <= '9') || ch == '-' || ch == '_';
    });
}
bool valid_pipe_name(const std::wstring& name) {
    const std::wstring prefix(pipe_prefix);
    if (name.size() != prefix.size() + 32 || name.compare(0, prefix.size(), prefix) != 0) return false;
    return std::all_of(name.begin() + prefix.size(), name.end(), [](wchar_t ch) {
        return (ch >= L'0' && ch <= L'9') || (ch >= L'a' && ch <= L'f');
    });
}
std::uint64_t creation_time(HANDLE process) {
    FILETIME created{}, exited{}, kernel{}, user{};
    if (!GetProcessTimes(process, &created, &exited, &kernel, &user)) return 0;
    return (static_cast<std::uint64_t>(created.dwHighDateTime) << 32) | created.dwLowDateTime;
}
bool current_user_descriptor(PSECURITY_DESCRIPTOR& descriptor) {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return false;
    DWORD length = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &length);
    std::vector<unsigned char> data(length);
    const bool read = length && GetTokenInformation(token, TokenUser, data.data(), length, &length);
    CloseHandle(token);
    if (!read) return false;
    LPWSTR sid = nullptr;
    if (!ConvertSidToStringSidW(reinterpret_cast<TOKEN_USER*>(data.data())->User.Sid, &sid)) return false;
    const std::wstring sddl = std::wstring(L"D:P(A;;GA;;;") + sid + L")";
    LocalFree(sid);
    return ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.c_str(), SDDL_REVISION_1,
                                                                 &descriptor, nullptr) != FALSE;
}
bool private_file_security(HANDLE file) {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return false;
    DWORD length = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &length);
    std::vector<unsigned char> data(length);
    const bool read = length && GetTokenInformation(token, TokenUser, data.data(), length, &length);
    CloseHandle(token);
    if (!read) return false;
    const PSID current = reinterpret_cast<TOKEN_USER*>(data.data())->User.Sid;
    PSID owner = nullptr;
    PACL dacl = nullptr;
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    const DWORD security = GetSecurityInfo(file, SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
                                           &owner, nullptr, &dacl, nullptr, &descriptor);
    bool safe = false;
    if (security == ERROR_SUCCESS && descriptor && owner && dacl && EqualSid(owner, current)) {
        SECURITY_DESCRIPTOR_CONTROL control{};
        DWORD revision = 0;
        ACL_SIZE_INFORMATION info{};
        if (GetSecurityDescriptorControl(descriptor, &control, &revision) && (control & SE_DACL_PROTECTED) &&
            GetAclInformation(dacl, &info, sizeof(info), AclSizeInformation) && info.AceCount == 1) {
            void* raw = nullptr;
            if (GetAce(dacl, 0, &raw)) {
                const auto* header = static_cast<ACE_HEADER*>(raw);
                if (header->AceType == ACCESS_ALLOWED_ACE_TYPE && !(header->AceFlags & INHERITED_ACE)) {
                    const auto* ace = static_cast<ACCESS_ALLOWED_ACE*>(raw);
                    safe = EqualSid(const_cast<DWORD*>(&ace->SidStart), current) != FALSE;
                }
            }
        }
    }
    if (descriptor) LocalFree(descriptor);
    return safe;
}
bool read_message(HANDLE pipe, ULONGLONG deadline, std::string& message) {
    std::array<char, max_message + 1> buffer{};
    do {
        DWORD read = 0;
        if (ReadFile(pipe, buffer.data(), static_cast<DWORD>(buffer.size()), &read, nullptr)) {
            if (read == 0 || read > max_message) return false;
            message.assign(buffer.data(), read);
            return true;
        }
        const DWORD error = GetLastError();
        if (error != ERROR_NO_DATA && error != ERROR_PIPE_LISTENING) return false;
        if (GetTickCount64() >= deadline) break;
        Sleep(10);
    } while (true);
    return false;
}
bool write_message(HANDLE pipe, const std::string& message) {
    if (message.empty() || message.size() > max_message) return false;
    DWORD written = 0;
    return WriteFile(pipe, message.data(), static_cast<DWORD>(message.size()), &written, nullptr) &&
           written == message.size();
}
LocalOperationResult transport_unknown() {
    return {4, "Owner transport unavailable or outcome unknown; reconcile the retained owner."};
}
}
struct OwnerPipe::Impl { HANDLE pipe{INVALID_HANDLE_VALUE}; ~Impl() { if (pipe != INVALID_HANDLE_VALUE) CloseHandle(pipe); } };
#else
struct OwnerPipe::Impl {};
#endif

OwnerPipe::OwnerPipe() : impl_(std::make_unique<Impl>()) {}
OwnerPipe::~OwnerPipe() = default;

bool OwnerPipe::open(OwnerEndpoint& endpoint, std::string& error) {
#ifdef _WIN32
    endpoint = {};
    if (!impl_ || impl_->pipe != INVALID_HANDLE_VALUE) { error = "owner pipe already open"; return false; }
    std::array<unsigned char, 16> nonce{};
    if (BCryptGenRandom(nullptr, nonce.data(), static_cast<ULONG>(nonce.size()), BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0) {
        error = "owner pipe random identity unavailable"; return false;
    }
    static constexpr wchar_t hex[] = L"0123456789abcdef";
    std::wstring name(pipe_prefix);
    for (const auto byte : nonce) { name.push_back(hex[byte >> 4]); name.push_back(hex[byte & 15]); }
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    if (!current_user_descriptor(descriptor)) { error = "owner pipe user security unavailable"; return false; }
    SECURITY_ATTRIBUTES attributes{sizeof(attributes), descriptor, FALSE};
    HANDLE pipe = CreateNamedPipeW(name.c_str(), PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
        PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_NOWAIT | PIPE_REJECT_REMOTE_CLIENTS,
        1, static_cast<DWORD>(max_message), static_cast<DWORD>(max_message), 0, &attributes);
    LocalFree(descriptor);
    if (pipe == INVALID_HANDLE_VALUE) { error = "owner pipe creation failed"; return false; }
    const auto birth = creation_time(GetCurrentProcess());
    if (!birth) { CloseHandle(pipe); error = "owner process identity unavailable"; return false; }
    impl_->pipe = pipe;
    endpoint = {std::move(name), GetCurrentProcessId(), birth};
    error.clear();
    return true;
#else
    (void)endpoint; error = "local owner pipe unavailable on this platform"; return false;
#endif
}

OwnerServeResult OwnerPipe::serve_once(LocalRunService& owner, std::chrono::milliseconds timeout, std::string& error) {
#ifdef _WIN32
    if (!impl_ || impl_->pipe == INVALID_HANDLE_VALUE || !valid_timeout(timeout)) {
        error = "owner pipe not ready"; return OwnerServeResult::error;
    }
    const ULONGLONG deadline = GetTickCount64() + static_cast<ULONGLONG>(timeout.count());
    for (;;) {
        if (ConnectNamedPipe(impl_->pipe, nullptr) || GetLastError() == ERROR_PIPE_CONNECTED) break;
        const DWORD code = GetLastError();
        if (code != ERROR_PIPE_LISTENING && code != ERROR_NO_DATA) { error = "owner connection failed"; return OwnerServeResult::error; }
        if (code == ERROR_NO_DATA) DisconnectNamedPipe(impl_->pipe);
        if (GetTickCount64() >= deadline) return OwnerServeResult::timeout;
        Sleep(10);
    }
    std::string input;
    bool handled = false;
    if (read_message(impl_->pipe, deadline, input)) {
        try {
            const auto request = json::parse(input);
            if (request.is_object() && request.size() == 3 && request.value("version", 0) == 1 &&
                request.contains("command") && request["command"].is_string()) {
                const auto command = request["command"].get<std::string>();
                LocalOperationResult result;
                bool valid = false;
                if (command == "current" && request.contains("run_id") && request["run_id"].is_string() &&
                    request["run_id"].get<std::string>().empty()) {
                    result = owner.current(); valid = true;
                } else if ((command == "status" || command == "cancel") && request.contains("run_id") && request["run_id"].is_string()) {
                    const auto id = request["run_id"].get<std::string>();
                    if (valid_id(id)) { result = command == "status" ? owner.status(id) : owner.cancel(id); valid = true; }
                } else if (command == "start" && request.contains("request") && request["request"].is_string()) {
                    const auto path = request["request"].get<std::string>();
                    if (!path.empty() && path.find('\0') == std::string::npos) {
                        const auto local_path = std::filesystem::u8path(path);
                        if (local_path.is_absolute()) { result = owner.start(local_path); valid = true; }
                    }
                }
                if (valid) {
                    const json response = {{"version", 1}, {"code", result.exit_code}, {"message", result.message},
                                           {"run_id", result.run_id}, {"job_stopped", result.job_stopped}};
                    // DisconnectNamedPipe discards unread bytes. Wait for the
                    // client to acknowledge the response before disconnecting,
                    // but never wait past this request's deadline.
                    std::string acknowledgement;
                    handled = write_message(impl_->pipe, response.dump()) &&
                              read_message(impl_->pipe, deadline, acknowledgement) &&
                              acknowledgement == "ack";
                }
            }
        } catch (...) { /* malformed requests never reach a backend */ }
    }
    DisconnectNamedPipe(impl_->pipe);
    if (!handled) { error = "owner request or response was not completed"; return OwnerServeResult::error; }
    error.clear();
    return OwnerServeResult::handled;
#else
    (void)owner; (void)timeout; error = "local owner pipe unavailable on this platform"; return OwnerServeResult::error;
#endif
}

namespace {
LocalOperationResult owner_exchange(const OwnerEndpoint& endpoint, const std::string& command,
                                     const std::string& argument, std::chrono::milliseconds timeout) {
#ifdef _WIN32
    const bool starting = command == "start";
    const bool current = command == "current";
    if (!valid_timeout(timeout) || !valid_pipe_name(endpoint.pipe_name) ||
        !endpoint.process_id || !endpoint.process_creation_time ||
        (!starting && !current && !valid_id(argument)) || (current && !argument.empty())) return transport_unknown();
    const ULONGLONG deadline = GetTickCount64() + static_cast<ULONGLONG>(timeout.count());
    HANDLE pipe = INVALID_HANDLE_VALUE;
    do {
        pipe = CreateFileW(endpoint.pipe_name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        if (pipe != INVALID_HANDLE_VALUE) break;
        if (GetLastError() != ERROR_PIPE_BUSY || GetTickCount64() >= deadline) return transport_unknown();
        Sleep(10);
    } while (true);
    ULONG server_id = 0;
    HANDLE process = nullptr;
    bool verified = GetNamedPipeServerProcessId(pipe, &server_id) && server_id == endpoint.process_id;
    if (verified) process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, server_id);
    verified = process && creation_time(process) == endpoint.process_creation_time &&
               WaitForSingleObject(process, 0) == WAIT_TIMEOUT;
    if (process) CloseHandle(process);
    DWORD mode = PIPE_READMODE_MESSAGE | PIPE_NOWAIT;
    if (!verified || !SetNamedPipeHandleState(pipe, &mode, nullptr, nullptr)) { CloseHandle(pipe); return transport_unknown(); }
    json request = {{"version", 1}, {"command", command}};
    request[starting ? "request" : "run_id"] = argument;
    std::string response;
    const bool exchanged = write_message(pipe, request.dump()) && read_message(pipe, deadline, response);
    if (exchanged) (void)write_message(pipe, "ack");
    CloseHandle(pipe);
    if (!exchanged) return transport_unknown();
    try {
        const auto value = json::parse(response);
        if (!value.is_object() || value.size() != 5 || value.value("version", 0) != 1 ||
            !value.contains("code") || !value["code"].is_number_integer() ||
            !value.contains("message") || !value["message"].is_string() ||
            !value.contains("run_id") || !value["run_id"].is_string() ||
            !value.contains("job_stopped") || !value["job_stopped"].is_boolean()) return transport_unknown();
        const auto code = value["code"].get<int>();
        const auto returned_id = value["run_id"].get<std::string>();
        if ((code != 0 && code != 1 && code != 3 && code != 4) ||
            (!starting && code != 0 && code != 4) ||
            (code == 0 && (!valid_id(returned_id) || (!starting && !current && returned_id != argument))) ||
            (!starting && !current && !returned_id.empty() && returned_id != argument) ||
            (!returned_id.empty() && !valid_id(returned_id)) ||
            (value["job_stopped"].get<bool>() && (code != 0 || !valid_id(returned_id)))) return transport_unknown();
        return {code, value["message"].get<std::string>(), returned_id, value["job_stopped"].get<bool>()};
    } catch (...) { return transport_unknown(); }
#else
    (void)endpoint; (void)command; (void)argument; (void)timeout;
    return {4, "Owner transport unavailable or outcome unknown; reconcile the retained owner."};
#endif
}
}
LocalOperationResult owner_pipe_call(const OwnerEndpoint& endpoint, OwnerCommand command,
                                     const std::string& run_id, std::chrono::milliseconds timeout) {
    if (command != OwnerCommand::current && command != OwnerCommand::status && command != OwnerCommand::cancel)
        return {4, "Owner transport unavailable or outcome unknown; reconcile the retained owner."};
    const auto command_name = command == OwnerCommand::current ? "current" :
                              command == OwnerCommand::status ? "status" : "cancel";
    return owner_exchange(endpoint, command_name, run_id, timeout);
}
LocalOperationResult owner_pipe_start(const OwnerEndpoint& endpoint, const std::filesystem::path& request,
                                      std::chrono::milliseconds timeout) {
    if (!request.is_absolute()) return {4, "Owner request path is invalid."};
    const auto encoded = request.u8string();
    const std::string path(encoded.begin(), encoded.end());
    if (path.empty() || path.find('\0') != std::string::npos)
        return {4, "Owner request path is invalid."};
    return owner_exchange(endpoint, "start", path, timeout);
}

bool save_owner_endpoint(const std::filesystem::path& file, const OwnerEndpoint& endpoint, std::string& error) {
#ifdef _WIN32
    if (!file.is_absolute() || !valid_pipe_name(endpoint.pipe_name) || !endpoint.process_id || !endpoint.process_creation_time) {
        error = "invalid owner endpoint or private path"; return false;
    }
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    if (!current_user_descriptor(descriptor)) { error = "private endpoint security unavailable"; return false; }
    SECURITY_ATTRIBUTES attributes{sizeof(attributes), descriptor, FALSE};
    HANDLE handle = CreateFileW(file.c_str(), GENERIC_WRITE | READ_CONTROL, 0, &attributes, CREATE_NEW,
                                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    LocalFree(descriptor);
    if (handle == INVALID_HANDLE_VALUE) { error = "could not create private endpoint exclusively"; return false; }
    const std::string pipe_name(endpoint.pipe_name.begin(), endpoint.pipe_name.end());
    const json value = {{"version", 1}, {"pipe", pipe_name}, {"pid", endpoint.process_id},
                        {"creation_time", endpoint.process_creation_time}};
    const auto contents = value.dump();
    DWORD written = 0;
    const bool ok = private_file_security(handle) && contents.size() <= max_message &&
                    WriteFile(handle, contents.data(), static_cast<DWORD>(contents.size()), &written, nullptr) &&
                    written == contents.size() && FlushFileBuffers(handle);
    CloseHandle(handle);
    // On failure preserve any newly created partial file for inspection; never
    // delete or overwrite an existing endpoint or claim it is usable.
    if (!ok) { error = "private endpoint write or security validation failed"; return false; }
    error.clear(); return true;
#else
    (void)file; (void)endpoint; error = "private endpoint unavailable on this platform"; return false;
#endif
}

bool load_owner_endpoint(const std::filesystem::path& file, OwnerEndpoint& endpoint, std::string& error) {
    endpoint = {};
#ifdef _WIN32
    if (!file.is_absolute()) { error = "invalid private endpoint path"; return false; }
    HANDLE handle = CreateFileW(file.c_str(), GENERIC_READ | READ_CONTROL, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (handle == INVALID_HANDLE_VALUE) { error = "private endpoint unavailable"; return false; }
    BY_HANDLE_FILE_INFORMATION info{};
    LARGE_INTEGER size{};
    std::array<char, max_message + 1> buffer{};
    DWORD read = 0;
    const bool ok = GetFileType(handle) == FILE_TYPE_DISK && GetFileInformationByHandle(handle, &info) &&
                    !(info.dwFileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY)) &&
                    private_file_security(handle) && GetFileSizeEx(handle, &size) && size.QuadPart > 0 &&
                    size.QuadPart <= static_cast<LONGLONG>(max_message) &&
                    ReadFile(handle, buffer.data(), static_cast<DWORD>(size.QuadPart), &read, nullptr) &&
                    read == static_cast<DWORD>(size.QuadPart);
    CloseHandle(handle);
    if (!ok) { error = "private endpoint could not be verified"; return false; }
    try {
        const auto value = json::parse(buffer.data(), buffer.data() + read);
        if (!value.is_object() || value.size() != 4 || value.value("version", 0) != 1 ||
            !value.contains("pipe") || !value["pipe"].is_string() ||
            !value.contains("pid") || !value["pid"].is_number_unsigned() ||
            !value.contains("creation_time") || !value["creation_time"].is_number_unsigned()) throw 0;
        const auto name = value["pipe"].get<std::string>();
        const auto pid = value["pid"].get<std::uint64_t>();
        const auto birth = value["creation_time"].get<std::uint64_t>();
        const std::wstring wide(name.begin(), name.end());
        if (!valid_pipe_name(wide) || !pid || pid > std::numeric_limits<std::uint32_t>::max() || !birth) throw 0;
        endpoint = {wide, static_cast<std::uint32_t>(pid), birth};
        error.clear(); return true;
    } catch (...) { error = "private endpoint format invalid"; return false; }
#else
    (void)file; error = "private endpoint unavailable on this platform"; return false;
#endif
}
LocalOperationResult owner_file_call(const std::filesystem::path& file, OwnerCommand command,
                                     const std::string& run_id, std::chrono::milliseconds timeout) {
    OwnerEndpoint endpoint;
    std::string error;
    if (!load_owner_endpoint(file, endpoint, error))
        return {4, "Owner endpoint unavailable or unverified; reconcile the retained owner."};
    return owner_pipe_call(endpoint, command, run_id, timeout);
}
LocalOperationResult owner_file_start(const std::filesystem::path& file, const std::filesystem::path& request,
                                       std::chrono::milliseconds timeout) {
    OwnerEndpoint endpoint;
    std::string error;
    if (!load_owner_endpoint(file, endpoint, error))
        return {4, "Owner endpoint unavailable or unverified; reconcile the retained owner."};
    return owner_pipe_start(endpoint, request, timeout);
}
OwnerHost::OwnerHost(std::unique_ptr<OwnedRunBackend> backend) : owner_(std::move(backend)) {}
bool OwnerHost::open(const std::filesystem::path& endpoint_file, std::string& error) {
    if (ready_) { error = "owner host already open"; return false; }
    OwnerEndpoint endpoint;
    if (!pipe_.open(endpoint, error)) return false;
    if (!save_owner_endpoint(endpoint_file, endpoint, error)) return false;
    ready_ = true;
    return true;
}
OwnerServeResult OwnerHost::serve_once(std::chrono::milliseconds timeout, std::string& error) {
    if (!ready_) { error = "owner host not ready"; return OwnerServeResult::error; }
    return pipe_.serve_once(owner_, timeout, error);
}
LocalOperationResult OwnerHost::current() { return owner_.current(); }
bool OwnerHost::may_shutdown() const noexcept { return owner_.may_shutdown(); }
OwnerServeResult OwnerHost::serve_until_quiescent(bool (*stop_requested)(void*) noexcept, void* context,
                                                   std::chrono::milliseconds poll_interval, std::string& error) {
    if (!ready_ || !stop_requested || poll_interval < std::chrono::milliseconds(10) ||
        poll_interval > std::chrono::seconds(30)) {
        error = "owner loop configuration invalid"; return OwnerServeResult::error;
    }
    for (;;) {
        if (stop_requested(context) && owner_.may_shutdown()) { error.clear(); return OwnerServeResult::handled; }
        const auto served = pipe_.serve_once(owner_, poll_interval, error);
        // Observation is not a training receipt. It only allows a verified
        // stopped job to become quiescent; unknown stays non-quiescent.
        if (!owner_.may_shutdown()) (void)owner_.current();
        if (served == OwnerServeResult::error) {
            if (owner_.may_shutdown()) return served;
            // Do not spin or abandon ownership if the control transport fails.
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}
}
