#include "canopy_foundry/execution/local_backend.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <sddl.h>
#endif

namespace canopy_foundry {
#ifdef _WIN32
namespace {
using json = nlohmann::json;
[[noreturn]] void fail(const char* message) { throw std::runtime_error(message); }
struct Handle {
    HANDLE value{INVALID_HANDLE_VALUE};
    Handle() = default;
    explicit Handle(HANDLE handle) : value(handle) {}
    ~Handle() { reset(); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& other) noexcept : value(std::exchange(other.value, INVALID_HANDLE_VALUE)) {}
    Handle& operator=(Handle&& other) noexcept {
        if (this != &other) { reset(); value = std::exchange(other.value, INVALID_HANDLE_VALUE); }
        return *this;
    }
    bool valid() const { return value && value != INVALID_HANDLE_VALUE; }
    void reset(HANDLE handle = INVALID_HANDLE_VALUE) { if (valid()) CloseHandle(value); value = handle; }
};

std::string utf8(const std::filesystem::path& path) {
    const auto value = path.u8string(); return {value.begin(), value.end()};
}
bool digest_valid(const std::string& digest) {
    return digest.size() == 64 && std::all_of(digest.begin(), digest.end(), [](char ch) {
        return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
    });
}
std::string hex(const unsigned char* data, std::size_t length) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (std::size_t i = 0; i < length; ++i) { result += digits[data[i] >> 4]; result += digits[data[i] & 15]; }
    return result;
}
std::string nonce() {
    std::array<unsigned char, 16> bytes{};
    if (BCryptGenRandom(nullptr, bytes.data(), static_cast<ULONG>(bytes.size()), BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0)
        fail("local run identity unavailable");
    return hex(bytes.data(), bytes.size());
}
std::uint64_t file_size(HANDLE file) {
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(file, &size) || size.QuadPart < 0) fail("local file size unavailable");
    return static_cast<std::uint64_t>(size.QuadPart);
}
void rewind(HANDLE file) {
    LARGE_INTEGER zero{};
    if (!SetFilePointerEx(file, zero, nullptr, FILE_BEGIN)) fail("local file seek failed");
}
std::string read_json_bytes(HANDLE file) {
    const auto size = file_size(file);
    if (!size || size > (2u << 20)) fail("local request exceeds inspection bounds");
    rewind(file);
    std::string result(static_cast<std::size_t>(size), '\0');
    DWORD read = 0;
    if (!ReadFile(file, result.data(), static_cast<DWORD>(result.size()), &read, nullptr) || read != result.size())
        fail("local request could not be read completely");
    return result;
}
struct PreparationCancelled {};
void check_cancel(const std::atomic<bool>* cancelled) {
    if (cancelled && cancelled->load(std::memory_order_acquire)) throw PreparationCancelled{};
}
std::string hash_file(HANDLE file, const std::atomic<bool>* cancelled = nullptr) {
    struct Hash {
        std::vector<unsigned char> object;
        BCRYPT_ALG_HANDLE algorithm{};
        BCRYPT_HASH_HANDLE hash{};
        ~Hash() { if (hash) BCryptDestroyHash(hash); if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0); }
    } state;
    DWORD length = 0, received = 0;
    if (BCryptOpenAlgorithmProvider(&state.algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0 ||
        BCryptGetProperty(state.algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&length), sizeof(length), &received, 0) < 0)
        fail("SHA-256 unavailable");
    state.object.resize(length);
    if (BCryptCreateHash(state.algorithm, &state.hash, state.object.data(), length, nullptr, 0, 0) < 0) fail("SHA-256 initialization failed");
    rewind(file);
    std::array<unsigned char, 65536> buffer{};
    for (;;) {
        check_cancel(cancelled);
        DWORD count = 0;
        if (!ReadFile(file, buffer.data(), static_cast<DWORD>(buffer.size()), &count, nullptr)) fail("pinned file read failed");
        if (!count) break;
        if (BCryptHashData(state.hash, buffer.data(), count, 0) < 0) fail("SHA-256 input failed");
    }
    std::array<unsigned char, 32> digest{};
    if (BCryptFinishHash(state.hash, digest.data(), static_cast<ULONG>(digest.size()), 0) < 0) fail("SHA-256 completion failed");
    return hex(digest.data(), digest.size());
}

// Lock every ancestor against rename/delete and every input against writes.
// Reject remote, alternate-stream, junction and symbolic-link paths.
Handle pin_path(const std::filesystem::path& supplied, bool directory, std::vector<Handle>& pins) {
    const auto path = supplied.lexically_normal();
    const auto raw = path.native();
    if (!path.is_absolute() || raw.size() < 3 || raw[1] != L':' || raw.find(L':', 2) != std::wstring::npos || raw.find(L'\0') != std::wstring::npos)
        fail("an absolute local filesystem path is required");
    auto current = path.root_path();
    const auto components = path.relative_path();
    for (auto it = components.begin(); it != components.end(); ++it) {
        current /= *it;
        const bool last = std::next(it) == components.end();
        const bool want_directory = !last || directory;
        Handle handle(CreateFileW(current.c_str(), want_directory ? FILE_READ_ATTRIBUTES : GENERIC_READ,
            want_directory ? FILE_SHARE_READ | FILE_SHARE_WRITE : FILE_SHARE_READ, nullptr, OPEN_EXISTING,
            FILE_FLAG_OPEN_REPARSE_POINT | (want_directory ? FILE_FLAG_BACKUP_SEMANTICS : FILE_ATTRIBUTE_NORMAL), nullptr));
        BY_HANDLE_FILE_INFORMATION info{};
        if (!handle.valid() || !GetFileInformationByHandle(handle.value, &info) ||
            (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) ||
            bool(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != want_directory || GetFileType(handle.value) != FILE_TYPE_DISK)
            fail("local path is unavailable, linked, or not the required file type");
        if (last) return handle;
        pins.push_back(std::move(handle));
    }
    fail("a filesystem root cannot be used as an asset");
}
struct PrivateSecurity {
    PSECURITY_DESCRIPTOR descriptor{};
    SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, FALSE};
    PrivateSecurity() {
        Handle token;
        HANDLE raw = nullptr;
        if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw)) fail("owner identity unavailable");
        token.reset(raw);
        DWORD size = 0;
        GetTokenInformation(token.value, TokenUser, nullptr, 0, &size);
        std::vector<unsigned char> bytes(size);
        if (!size || !GetTokenInformation(token.value, TokenUser, bytes.data(), size, &size)) fail("owner identity unavailable");
        LPWSTR sid = nullptr;
        if (!ConvertSidToStringSidW(reinterpret_cast<TOKEN_USER*>(bytes.data())->User.Sid, &sid)) fail("owner identity unavailable");
        const auto sddl = std::wstring(L"D:P(A;OICI;FA;;;") + sid + L")";
        LocalFree(sid);
        if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.c_str(), SDDL_REVISION_1, &descriptor, nullptr))
            fail("private run security unavailable");
        attributes.lpSecurityDescriptor = descriptor;
    }
    ~PrivateSecurity() { if (descriptor) LocalFree(descriptor); }
};
void private_directory(const std::filesystem::path& path) {
    PrivateSecurity security;
    if (!CreateDirectoryW(path.c_str(), &security.attributes)) fail("fresh private run directory could not be created");
}
Handle new_file(const std::filesystem::path& path, bool inherit = false) {
    PrivateSecurity security;
    security.attributes.bInheritHandle = inherit;
    Handle file(CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, &security.attributes,
                            CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr));
    if (!file.valid()) fail("fresh local artifact could not be created");
    return file;
}
void write_new(const std::filesystem::path& path, const std::string& contents) {
    auto file = new_file(path);
    DWORD written = 0;
    if (contents.size() > std::numeric_limits<DWORD>::max() ||
        !WriteFile(file.value, contents.data(), static_cast<DWORD>(contents.size()), &written, nullptr) ||
        written != contents.size() || !FlushFileBuffers(file.value)) fail("local provenance write failed");
}
void publish_launch(const std::filesystem::path& directory, const std::string& contents,
                    const std::atomic<bool>& cancelled) {
    // A partial write remains visibly pending and is never a launch record.
    const auto pending = directory / "launch-provenance.pending";
    write_new(pending, contents);
    check_cancel(&cancelled);
    if (!MoveFileExW(pending.c_str(), (directory / "launch-provenance.json").c_str(), MOVEFILE_WRITE_THROUGH))
        fail("local provenance publication failed");
}
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

class NativeBackend final : public OwnedRunBackend {
public:
    explicit NativeBackend(LocalBackendConfig config) : config_(std::move(config)) {
        config_.binary = config_.binary.lexically_normal();
        config_.run_root = config_.run_root.lexically_normal();
        std::transform(config_.binary_sha256.begin(), config_.binary_sha256.end(), config_.binary_sha256.begin(),
            [](unsigned char ch) { return ch >= 'A' && ch <= 'F' ? static_cast<char>(ch + ('a' - 'A')) : static_cast<char>(ch); });
        if (!digest_valid(config_.binary_sha256) || config_.device < 0 || config_.device > 255 ||
            config_.maximum_runtime < std::chrono::seconds(1) || config_.maximum_runtime > std::chrono::hours(24) ||
            !config_.maximum_output_bytes || !config_.maximum_log_bytes || config_.binary.filename() != L"ida_native_train.exe")
            fail("invalid local owner configuration");
    }
    ~NativeBackend() override {
        cancel_preparation_.store(true, std::memory_order_release);
        if (preparation_.joinable()) preparation_.join();
    }
    OwnedRunObservation start(const ida_native::NativeRequest& request, const PlanResolution& resolution) override {
        if (preparation_.joinable()) {
            if (!preparation_done_.load(std::memory_order_acquire))
                return {OwnedRunState::unknown, id_, "Previous input preparation is still active."};
            preparation_.join();
        }
        if (created_any_ && !finished_) return {OwnedRunState::unknown, id_, "Previous containment is not reconciled."};
        reset();
        try { id_ = "local-" + nonce(); }
        catch (...) { return {OwnedRunState::rejected, {}, "Preparation identity unavailable; no process created."}; }
        cancel_preparation_.store(false, std::memory_order_relaxed);
        preparation_done_.store(false, std::memory_order_relaxed);
        try {
            preparation_ = std::thread([this, request, resolution] {
                try {
                    prepare(request, resolution);
                    check_cancel(&cancel_preparation_);
                    stage_ = Stage::probe;
                    started_ = GetTickCount64();
                    spawn(true);
                } catch (const PreparationCancelled&) {
                    finish_preparation_without_child("Preparation cancelled before process creation; training was not started.");
                } catch (const std::exception& error) {
                    if (created_any_) (void)unknown();
                    else finish_preparation_without_child(std::string("Preparation failed before process creation: ") + error.what());
                } catch (...) {
                    if (created_any_) (void)unknown();
                    else finish_preparation_without_child("Preparation failed before process creation; inspect local files.");
                }
                preparation_done_.store(true, std::memory_order_release);
            });
        } catch (...) {
            reset();
            return {OwnedRunState::rejected, {}, "Preparation worker could not start."};
        }
        return {OwnedRunState::running, id_, "Input preparation active; training has not started."};
    }
    OwnedRunObservation observe() override {
        if (preparation_.joinable()) {
            if (!preparation_done_.load(std::memory_order_acquire))
                return {OwnedRunState::running, id_, cancel_preparation_.load(std::memory_order_acquire)
                    ? "Cancellation requested; preparation is stopping. Training has not started."
                    : "Input preparation active; training has not started."};
            preparation_.join();
            if (cancel_preparation_.load(std::memory_order_acquire) && !finished_ && !cancelled_) {
                cancelled_ = operator_cancelled_ = true;
                stop_reason_ = "operator";
                final_message_ = "Cancellation requested during preparation; training outcome unverified.";
                if (!job_.valid() || !TerminateJobObject(job_.value, 1)) return unknown();
            }
        }
        if (finished_) return {OwnedRunState::stopped, id_, final_message_};
        if (!created_any_ || !job_.valid() || !process_.valid()) return {OwnedRunState::unknown, id_, "Containment is unavailable."};
        try {
            if (!cancelled_ && (GetTickCount64() - started_ >= static_cast<ULONGLONG>(config_.maximum_runtime.count()) * 1000 || over_quota())) {
                cancelled_ = true;
                stop_reason_ = "limit";
                final_message_ = "Owner runtime or output limit reached; training outcome unverified.";
                if (!TerminateJobObject(job_.value, 1)) return unknown();
            }
            JOBOBJECT_BASIC_ACCOUNTING_INFORMATION accounting{};
            if (!QueryInformationJobObject(job_.value, JobObjectBasicAccountingInformation, &accounting, sizeof(accounting), nullptr)) return unknown();
            if (accounting.ActiveProcesses) return running();
            DWORD code = 0;
            if (WaitForSingleObject(process_.value, 0) != WAIT_OBJECT_0 || !GetExitCodeProcess(process_.value, &code)) return unknown();
            if (!FlushFileBuffers(stdout_.value) || !FlushFileBuffers(stderr_.value)) return unknown();
            const auto log_prefix = stage_ == Stage::probe ? "probe" : "trainer";
            if (record_attempts_ >= 3) {
                unresolved_reason_ = "Process record retries exhausted; inspect the run directory; retained ownership remains fenced.";
                return unknown();
            }
            const auto record = std::string(log_prefix) + (record_attempts_ ? "-process-recovery-" + std::to_string(record_attempts_) : "-process") + ".json";
            ++record_attempts_;
            unresolved_reason_ = "Process record incomplete or unavailable; inspect the run directory, then use current to reconcile retained handles.";
            write_new(directory_ / record, json({
                {"schema", "canopy-foundry-local-process.v1"}, {"run_id", id_}, {"stage", log_prefix},
                {"exit_code", code}, {"job_empty", true}, {"cancel_requested", operator_cancelled_},
                {"termination_requested", cancelled_}, {"stop_reason", stop_reason_},
                {"stdout_sha256", hash_file(stdout_.value)}, {"stderr_sha256", hash_file(stderr_.value)},
                {"training_outcome", "unverified"}, {"promotion_eligible", false}}).dump(2));
            record_attempts_ = 0;
            unresolved_reason_.clear();
            if (stage_ == Stage::probe && !cancelled_ && code == 0) {
                process_.reset(); job_.reset(); stdout_.reset(); stderr_.reset();
                stage_ = Stage::trainer;
                try { spawn(false); }
                catch (...) {
                    if (process_.valid()) return unknown();
                    job_.reset(); stdout_.reset(); stderr_.reset();
                    finish_preparation_without_child("Trainer setup failed after a verified empty probe; trainer process was not created. Inspect local files.");
                    return {OwnedRunState::stopped, id_, final_message_};
                }
                return running();
            }
            finished_ = true;
            if (final_message_.empty()) final_message_ = stage_ == Stage::probe
                ? "Native self-check failed; training was not started. Inspect local probe diagnostics."
                : "Trainer exited with code " + std::to_string(code) + "; native checkpoint, metrics and evaluation require separate validation.";
            pins_.clear(); binary_.reset(); request_.reset();
            return {OwnedRunState::stopped, id_, final_message_};
        } catch (...) { return unknown(); }
    }
    OwnedRunObservation cancel(std::chrono::milliseconds timeout) override {
        if (preparation_.joinable() && !preparation_done_.load(std::memory_order_acquire)) {
            cancel_preparation_.store(true, std::memory_order_release);
            return {OwnedRunState::running, id_, "Cancellation requested; preparation is stopping. Refresh status."};
        }
        if (preparation_.joinable()) {
            cancel_preparation_.store(true, std::memory_order_release);
            return observe();
        }
        if (finished_) return observe();
        if (!created_any_ || !job_.valid() || timeout < std::chrono::milliseconds(0) || timeout > std::chrono::seconds(30)) return unknown();
        cancelled_ = true; // prevents an already-exited probe from starting training
        operator_cancelled_ = true;
        stop_reason_ = "operator";
        final_message_ = "Cancellation verified for the owned job; training outcome unverified.";
        if (!TerminateJobObject(job_.value, 1)) return unknown();
        const auto until = GetTickCount64() + static_cast<ULONGLONG>(timeout.count());
        for (;;) {
            const auto result = observe();
            if (result.state != OwnedRunState::running || GetTickCount64() >= until) return result;
            Sleep(10);
        }
    }
private:
    enum class Stage { probe, trainer };
    LocalBackendConfig config_;
    std::thread preparation_;
    std::atomic<bool> preparation_done_{true}, cancel_preparation_{false};
    std::vector<Handle> pins_;
    Handle binary_, request_, job_, process_, stdout_, stderr_;
    std::filesystem::path directory_, staged_request_;
    std::string id_, final_message_, stop_reason_;
    std::string unresolved_reason_;
    Stage stage_{Stage::probe};
    int minimum_arch_{0};
    ULONGLONG started_{0};
    bool created_any_{false}, finished_{false}, cancelled_{false}, assigned_{false}, operator_cancelled_{false};
    unsigned int record_attempts_{0};
    void finish_preparation_without_child(std::string message) {
        finished_ = true;
        final_message_ = std::move(message);
        pins_.clear(); binary_.reset(); request_.reset();
    }
    OwnedRunObservation unknown() {
        // Stop only handles owned here; failure remains unknown. An uncertain
        // result must not leave a trainer free to run after polling is fenced.
        cancelled_ = true; // a recovered probe must never advance after uncertainty
        stop_reason_ = "uncertain";
        final_message_ = "Execution stopped after an unverified observation; training outcome remains unverified.";
        if (job_.valid()) (void)TerminateJobObject(job_.value, 1);
        if (!assigned_ && process_.valid()) (void)TerminateProcess(process_.value, 1);
        return {OwnedRunState::unknown, id_, unresolved_reason_.empty()
            ? "Owned job or provenance could not be verified; retain and reconcile this owner."
            : unresolved_reason_};
    }
    OwnedRunObservation running() const {
        return {OwnedRunState::running, id_, cancelled_ ? "Cancellation pending; job is not yet verified empty."
            : stage_ == Stage::probe ? "Native self-check active; training has not started."
                                    : "Trainer process active; this is not a training-success receipt."};
    }
    void reset() {
        process_.reset(); job_.reset(); stdout_.reset(); stderr_.reset(); request_.reset(); binary_.reset(); pins_.clear();
        id_.clear(); final_message_.clear(); stop_reason_.clear(); unresolved_reason_.clear(); record_attempts_ = 0;
        created_any_ = finished_ = cancelled_ = assigned_ = operator_cancelled_ = false;
    }
    void prepare(const ida_native::NativeRequest& request, const PlanResolution& resolution) {
        auto root = pin_path(config_.run_root, true, pins_); pins_.push_back(std::move(root));
        binary_ = pin_path(config_.binary, false, pins_);
        if (hash_file(binary_.value, &cancel_preparation_) != config_.binary_sha256) fail("local binary SHA-256 does not match the operator pin");
        auto source = pin_path(std::filesystem::absolute(request.request_json_path), false, pins_);
        const auto original = read_json_bytes(source.value);
        if (resolution.local_source_request_bytes.empty() || original != resolution.local_source_request_bytes)
            fail("source request changed after admission or its parsed identity is unavailable");
        auto payload = json::parse(original);
        // Restrict external paths/side effects to the admission surface. Unknown
        // fields are rejected rather than silently translated to another engine.
        static const std::vector<std::string> allowed{
            "backend", "attention_backend", "precision_profile", "optimizer_type", "optimizer_state_precision",
            "gradient_buffer_precision", "gemm_accumulator_precision", "device", "model", "training", "input",
            "family", "seat", "version", "seed", "native_max_steps", "max_samples", "grad_accum_override",
            "output_dir", "status_file", "required_arch", "architecture_contract", "job_id", "engine_revision",
            "expected_terminal_phase", "init_from_model", "resume_from_checkpoint"};
        if (!payload.is_object()) fail("local request must be an object");
        for (const auto& [key, value] : payload.items())
            if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) fail("local owner does not support a supplied request field");
        if (!request.parent.init_from_model.empty() || !request.parent.resume_from_checkpoint.empty())
            fail("local owner requires native parent/checkpoint attestation before enabling fine-tune or resume; use the established governed path");
        if (request.backend != "native" || request.device.runtime != "cuda" || !request.device.model_parallel_devices.empty() ||
            request.attention_backend != "scalar_flash" ||
            (request.precision_profile != "legacy_bf16" && request.precision_profile != "legacy_fp8"))
            fail("local owner admits single-device public CUDA legacy scalar execution only");
        const auto& model = request.model;
        if ((model.architecture_contract != "ida_lattice_native_v1" && model.architecture_contract != "hf_gpt2_native_v1" &&
             model.architecture_contract != "generic_moe_native_v1") || model.generic_moe_shared_expert_width != 0 ||
            (model.architecture_contract != "generic_moe_native_v1" &&
             (model.generic_moe_num_experts || model.generic_moe_top_k || model.generic_moe_expert_width)))
            fail("local model layout does not match a supported public contract");
        const auto arch = request.device.required_arch;
        if (arch.size() < 5 || arch.size() > 7 || arch.rfind("sm_", 0) != 0 ||
            !std::all_of(arch.begin() + 3, arch.end(), [](char ch) { return ch >= '0' && ch <= '9'; }))
            fail("local owner requires an explicit numeric sm_ architecture floor");
        minimum_arch_ = std::stoi(arch.substr(3));
        if (minimum_arch_ < 10 || minimum_arch_ > 999) fail("invalid CUDA architecture floor");
        check_cancel(&cancel_preparation_);
        directory_ = config_.run_root / id_;
        private_directory(directory_);
        auto run_directory = pin_path(directory_, true, pins_); pins_.push_back(std::move(run_directory));
        private_directory(directory_ / "output"); private_directory(directory_ / "temp");
        for (const auto* name : {"output", "temp"}) {
            auto owned = pin_path(directory_ / name, true, pins_); pins_.push_back(std::move(owned));
        }
        json inputs = json::array();
        for (const char* field : {"token_blocks", "label_blocks", "seg_blocks"}) {
            auto& input = payload["input"];
            const auto found = input.find(field);
            if (found == input.end() || found->is_null() || *found == "") {
                if (std::string(field) != "seg_blocks") fail("token and label blocks are required");
                continue;
            }
            if (!found->is_string()) fail("native input path must be a string");
            const auto path = std::filesystem::absolute(std::filesystem::u8path(found->get<std::string>()));
            auto asset = pin_path(path, false, pins_);
            if (!file_size(asset.value)) fail("empty native input block");
            inputs.push_back({{"role", field}, {"sha256", hash_file(asset.value, &cancel_preparation_)}, {"bytes", file_size(asset.value)}});
            *found = utf8(path); pins_.push_back(std::move(asset));
        }
        const auto tokens = inputs[0]["bytes"].get<std::uint64_t>();
        if (tokens % 4 || inputs[1]["bytes"].get<std::uint64_t>() != tokens ||
            (inputs.size() == 3 && inputs[2]["bytes"].get<std::uint64_t>() != tokens / 2)) fail("native input block sizes do not match u32/i32/u16 roles");
        payload["output_dir"] = utf8(directory_ / "output");
        payload["status_file"] = utf8(directory_ / "output/status.json");
        payload["job_id"] = id_;
        write_new(directory_ / "source-request.json", original);
        staged_request_ = directory_ / "request.json";
        const auto staged = payload.dump(2);
        write_new(staged_request_, staged);
        request_ = pin_path(staged_request_, false, pins_);
        const auto frozen = ida_native::load_request(staged_request_);
        const auto mismatch = execution_plan_binding_error(frozen, resolution);
        if (!mismatch.empty()) fail("frozen request does not match the selected execution plan");
        // No path-free public receipt is synthesized from these local facts.
        publish_launch(directory_, json({
            {"schema", "canopy-foundry-local-launch.v1"}, {"run_id", id_}, {"binary_sha256", config_.binary_sha256},
            {"source_request_sha256", sha256_hex(original)}, {"request_sha256", sha256_hex(staged)},
            {"execution_plan_sha256", execution_plan_sha256(resolution.plan)}, {"device", config_.device},
            {"inputs", inputs}, {"training_outcome", "unverified"}, {"promotion_eligible", false}}).dump(2), cancel_preparation_);
    }
    bool over_quota() {
        if (file_size(stdout_.value) > config_.maximum_log_bytes || file_size(stderr_.value) > config_.maximum_log_bytes) return true;
        std::uint64_t bytes = 0, count = 0;
        for (const auto& item : std::filesystem::recursive_directory_iterator(directory_ / "output")) {
            const auto attributes = GetFileAttributesW(item.path().c_str());
            if (attributes == INVALID_FILE_ATTRIBUTES) {
                const auto error = GetLastError();
                if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) continue; // writer renamed an entry
                fail("unverifiable local output");
            }
            if (attributes & FILE_ATTRIBUTE_REPARSE_POINT) fail("linked local output is unsupported");
            if (!(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
                std::error_code error;
                const auto size = item.file_size(error);
                if (error == std::errc::no_such_file_or_directory) continue;
                if (error) fail("unverifiable local output size");
                if (size > config_.maximum_output_bytes - bytes || ++count > 10000) return true;
                bytes += size;
            }
        }
        return false;
    }
    void spawn(bool probe) {
        job_.reset(CreateJobObjectW(nullptr, nullptr));
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
        if (!job_.valid() || !SetInformationJobObject(job_.value, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) fail("owned containment unavailable");
        stdout_ = new_file(directory_ / (probe ? "probe-stdout.txt" : "trainer-stdout.txt"), true);
        stderr_ = new_file(directory_ / (probe ? "probe-stderr.txt" : "trainer-stderr.txt"), true);
        SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
        Handle input(CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &attributes, OPEN_EXISTING, 0, nullptr));
        if (!input.valid()) fail("native stdin unavailable");
        std::array<HANDLE, 3> inherited{input.value, stdout_.value, stderr_.value};
        SIZE_T size = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
        std::vector<unsigned char> storage(size);
        auto* list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
        if (!size || !InitializeProcThreadAttributeList(list, 1, 0, &size)) fail("native handle isolation unavailable");
        struct Attributes { LPPROC_THREAD_ATTRIBUTE_LIST value; ~Attributes() { DeleteProcThreadAttributeList(value); } } cleanup{list};
        if (!UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited.data(), sizeof(inherited), nullptr, nullptr)) fail("native handle whitelist unavailable");
        STARTUPINFOEXW startup{}; startup.StartupInfo.cb = sizeof(startup); startup.lpAttributeList = list;
        startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        startup.StartupInfo.hStdInput = input.value; startup.StartupInfo.hStdOutput = stdout_.value; startup.StartupInfo.hStdError = stderr_.value;
        auto command = quote(config_.binary.wstring()) + (probe ? L" --self-check --require-cublaslt --require-private-pool --require-cuda-arch " + std::to_wstring(minimum_arch_)
                                                              : L" --request-json " + quote(staged_request_.wstring()));
        command += L" --device " + std::to_wstring(config_.device);
        if (command.size() >= 32767) fail("native command exceeds Windows bounds");
        // Match the worker's minimal environment; do not inherit prior-body
        // precision, accelerator selection, loader paths or credentials.
        std::wstring environment;
        for (const wchar_t* key : {L"SystemRoot", L"TEMP", L"TMP", L"WINDIR"}) {
            std::wstring value;
            if (std::wstring(key) == L"TEMP" || std::wstring(key) == L"TMP") value = (directory_ / "temp").wstring();
            else {
                std::array<wchar_t, 32768> buffer{};
                const DWORD length = GetEnvironmentVariableW(key, buffer.data(), static_cast<DWORD>(buffer.size()));
                if (!length || length >= buffer.size()) fail("native system environment unavailable");
                value = buffer.data();
            }
            environment += std::wstring(key) + L"=" + value; environment.push_back(L'\0');
        }
        environment.push_back(L'\0');
        PROCESS_INFORMATION child{};
        if (!CreateProcessW(config_.binary.c_str(), command.data(), nullptr, nullptr, TRUE,
            CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
            environment.data(), directory_.c_str(), &startup.StartupInfo, &child)) fail("native process creation failed");
        created_any_ = true; assigned_ = false; process_.reset(child.hProcess); Handle thread(child.hThread);
        if (!AssignProcessToJobObject(job_.value, process_.value)) {
            TerminateProcess(process_.value, 1); WaitForSingleObject(process_.value, 3000);
            fail("suspended native child could not be assigned to containment");
        }
        assigned_ = true;
        if (ResumeThread(thread.value) == static_cast<DWORD>(-1)) {
            TerminateJobObject(job_.value, 1); fail("native child could not resume inside containment");
        }
    }
};
}
#endif

std::unique_ptr<OwnedRunBackend> make_local_native_backend(LocalBackendConfig config) {
#ifdef _WIN32
    return std::make_unique<NativeBackend>(std::move(config));
#else
    (void)config;
    throw std::runtime_error("the local native owner requires Windows Job Object containment");
#endif
}
}
