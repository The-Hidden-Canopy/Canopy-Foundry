#pragma once

// Public, explicitly opt-in kernel ontology telemetry. This is deliberately
// smaller than the deployment-owned observability overlay: it records launch
// geometry, bounded host enqueue timing, and step-level diagnostic records.
// It does not record paths, model identifiers, datasets, checkpoints, or
// hardware-counter/device-duration claims.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
#define IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY 0
#endif

namespace ida_native::ontology {

struct Context {
    int microbatch = -1;
    int grad_accum = -1;
    int num_experts = -1;
    int top_k_experts = -1;
    int seq_window = -1;
    std::string precision_profile;
    std::string attention_backend;
    std::string optimizer;
    int resident_bodies = -1;
    std::string body_key;
};

struct StepOutcome {
    long long opt_step = -1;
    double loss = 0.0;
    double wallclock_s = 0.0;
    long long tokens = 0;
    bool update_skipped = false;
    bool clip_fired = false;
    double grad_norm = 0.0;
};

inline std::filesystem::path& local_export_path_slot() {
    static std::filesystem::path value;
    return value;
}

inline bool& local_export_armed_slot() {
    static bool value = false;
    return value;
}

inline bool enabled() {
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
    thread_local const bool value = [] {
        const char* raw = std::getenv("IDA_NATIVE_ONTOLOGY");
        return raw && raw[0] == '1';
    }();
    return value;
#else
    return local_export_armed_slot();
#endif
}

inline std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch) << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

inline std::string json_number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

inline std::mutex& sink_mutex() {
    static std::mutex value;
    return value;
}

inline std::FILE*& sink_slot() {
    static std::FILE* value = nullptr;
    return value;
}

inline void enable_local_export(const std::filesystem::path& path) {
    if (path.empty()) throw std::invalid_argument("ontology export path is empty");
    std::lock_guard<std::mutex> lock(sink_mutex());
    if (local_export_armed_slot() || sink_slot() != nullptr) {
        throw std::logic_error("ontology export is already configured");
    }
    local_export_path_slot() = path;
    local_export_armed_slot() = true;
}

inline std::FILE* sink() {
    if (!enabled()) return stderr;
    auto*& value = sink_slot();
    if (value != nullptr) return value;
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
        const char* configured = std::getenv("IDA_NATIVE_ONTOLOGY_PATH");
        const std::filesystem::path path =
            (configured && *configured) ? std::filesystem::path(configured)
                                        : std::filesystem::path("artifacts/telemetry/kernel_ontology.jsonl");
#else
        const auto path = local_export_path_slot();
        if (path.empty()) return stderr;
#endif
        std::error_code error;
        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(path.parent_path(), error);
        }
        std::FILE* file = std::fopen(path.string().c_str(), "ab");
        if (!file) return stderr;
        static char buffer[1 << 20];
        std::setvbuf(file, buffer, _IOFBF, sizeof(buffer));
        value = file;
    return value;
}

inline bool sink_is_file() { return sink() != stderr; }

inline constexpr std::uint64_t kMaxTelemetryBytes = 32ull * 1024ull * 1024ull;
inline constexpr std::uint64_t kMaxTelemetryRecords = 250'000ull;

inline std::uint64_t& emitted_bytes_slot() {
    static std::uint64_t value = 0;
    return value;
}

inline std::uint64_t& emitted_records_slot() {
    static std::uint64_t value = 0;
    return value;
}

inline bool& telemetry_overflow_slot() {
    static bool value = false;
    return value;
}

inline bool telemetry_overflowed() {
    std::lock_guard<std::mutex> lock(sink_mutex());
    return telemetry_overflow_slot();
}

inline bool write_line_locked(std::string_view line) {
    if (telemetry_overflow_slot() || !sink_is_file()) return false;
    if (line.size() > kMaxTelemetryBytes - 1) {
        telemetry_overflow_slot() = true;
        return false;
    }
    const auto bytes = static_cast<std::uint64_t>(line.size()) + 1;
    if (emitted_bytes_slot() > kMaxTelemetryBytes - bytes ||
        emitted_records_slot() >= kMaxTelemetryRecords) {
        telemetry_overflow_slot() = true;
        return false;
    }
    auto* output = sink();
    if (std::fwrite(line.data(), 1, line.size(), output) != line.size() ||
        std::fputc('\n', output) == EOF) {
        telemetry_overflow_slot() = true;
        return false;
    }
    emitted_bytes_slot() += bytes;
    ++emitted_records_slot();
    return true;
}

inline int& epoch_slot() {
    static int value = -1;
    return value;
}

inline int context_epoch() { return epoch_slot(); }

inline void emit_line(const std::string& line) {
    if (!enabled()) return;
    std::lock_guard<std::mutex> lock(sink_mutex());
    write_line_locked(line);
}

inline void set_context(const Context& context) {
    if (!enabled()) return;
    const int epoch = ++epoch_slot();
    std::ostringstream row;
    row << "{\"type\":\"context\",\"telemetry_schema\":\"native_ontology_v2\","
        << "\"ctx\":" << epoch
        << ",\"microbatch\":" << context.microbatch
        << ",\"grad_accum\":" << context.grad_accum
        << ",\"num_experts\":" << context.num_experts
        << ",\"top_k_experts\":" << context.top_k_experts
        << ",\"seq_window\":" << context.seq_window
        << ",\"precision_profile\":\"" << json_escape(context.precision_profile) << "\""
        << ",\"attention_backend\":\"" << json_escape(context.attention_backend) << "\""
        << ",\"optimizer\":\"" << json_escape(context.optimizer) << "\""
        << ",\"kernel_collection\":\"public_bounded\"}";
    emit_line(row.str());
}

struct LaunchAggregate {
    std::string kernel;
    std::string role;
    dim3 grid{};
    dim3 block{};
    std::size_t dyn_smem = 0;
    std::uint64_t launches = 0;
    std::uint64_t enqueue_ns_sum = 0;
    std::uint64_t enqueue_ns_min = 0;
    std::uint64_t enqueue_ns_max = 0;
};

inline std::map<std::string, LaunchAggregate>& launch_aggregates() {
    static std::map<std::string, LaunchAggregate> value;
    return value;
}

inline bool timing_enabled() { return enabled(); }

inline std::string launch_key(
    const char* kernel, const char* role, dim3 grid, dim3 block, std::size_t dyn_smem
) {
    std::ostringstream key;
    key << (kernel ? kernel : "") << '\n' << (role ? role : "") << '\n'
        << grid.x << ',' << grid.y << ',' << grid.z << '/'
        << block.x << ',' << block.y << ',' << block.z << '/' << dyn_smem;
    return key.str();
}

inline void record_launch_enqueue(
    const char* kernel, const char* role, dim3 grid, dim3 block,
    std::size_t dyn_smem, std::uint64_t enqueue_ns
) {
    if (!timing_enabled()) return;
    std::lock_guard<std::mutex> lock(sink_mutex());
    if (telemetry_overflow_slot()) return;
    const auto key = launch_key(kernel, role, grid, block, dyn_smem);
    auto iterator = launch_aggregates().find(key);
    if (iterator == launch_aggregates().end() &&
        launch_aggregates().size() >= kMaxTelemetryRecords) {
        telemetry_overflow_slot() = true;
        return;
    }
    auto& aggregate = launch_aggregates()[key];
    if (aggregate.launches == 0) {
        aggregate.kernel = kernel ? kernel : "";
        aggregate.role = role ? role : "";
        aggregate.grid = grid;
        aggregate.block = block;
        aggregate.dyn_smem = dyn_smem;
        aggregate.enqueue_ns_min = enqueue_ns;
    }
    ++aggregate.launches;
    aggregate.enqueue_ns_sum += enqueue_ns;
    aggregate.enqueue_ns_min = std::min(aggregate.enqueue_ns_min, enqueue_ns);
    aggregate.enqueue_ns_max = std::max(aggregate.enqueue_ns_max, enqueue_ns);
}

inline void flush_launch_timing(long long opt_step) {
    if (!timing_enabled()) return;
    std::lock_guard<std::mutex> lock(sink_mutex());
    if (launch_aggregates().empty() || !sink_is_file()) return;
    for (const auto& [key, aggregate] : launch_aggregates()) {
        (void)key;
        const double sum_us = static_cast<double>(aggregate.enqueue_ns_sum) / 1000.0;
        const double mean_us = aggregate.launches
            ? sum_us / static_cast<double>(aggregate.launches) : 0.0;
        const double min_us = static_cast<double>(aggregate.enqueue_ns_min) / 1000.0;
        const double max_us = static_cast<double>(aggregate.enqueue_ns_max) / 1000.0;
        std::ostringstream row;
        row << "{\"type\":\"launch_timing\",\"telemetry_schema\":\"native_ontology_v3\","
            << "\"ctx\":" << context_epoch()
            << ",\"cuda_device\":-1,\"opt_step\":" << opt_step
            << ",\"kernel\":\"" << json_escape(aggregate.kernel) << "\""
            << ",\"role\":\"" << json_escape(aggregate.role) << "\""
            << ",\"grid\":[" << aggregate.grid.x << ',' << aggregate.grid.y << ',' << aggregate.grid.z << ']'
            << ",\"block\":[" << aggregate.block.x << ',' << aggregate.block.y << ',' << aggregate.block.z << ']'
            << ",\"dyn_smem\":" << aggregate.dyn_smem
            << ",\"launches\":" << aggregate.launches
            << ",\"enqueue_ns_sum\":" << aggregate.enqueue_ns_sum
            << ",\"enqueue_us_sum\":" << std::fixed << std::setprecision(3) << sum_us
            << ",\"enqueue_us_mean\":" << std::setprecision(6) << mean_us
            << ",\"enqueue_us_min\":" << min_us
            << ",\"enqueue_us_max\":" << max_us
            << ",\"timing_scope\":\"host_enqueue_only\",\"device_execution_timing\":false}";
        if (!write_line_locked(row.str())) break;
    }
    if (std::fflush(sink()) != 0) telemetry_overflow_slot() = true;
    launch_aggregates().clear();
}

inline void observe(
    const char* kernel, const void* function, dim3 grid, dim3 block,
    std::size_t dyn_smem, const char* role
) {
    if (!enabled()) return;
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp properties{};
    if (device >= 0) cudaGetDeviceProperties(&properties, device);
    cudaFuncAttributes attributes{};
    const bool attributes_valid =
        function && cudaFuncGetAttributes(&attributes, function) == cudaSuccess;
    const auto blocks = static_cast<unsigned long long>(grid.x) * grid.y * grid.z;
    const auto threads = static_cast<unsigned long long>(block.x) * block.y * block.z;
    std::vector<std::string> findings;
    if (threads > 0 && threads < static_cast<unsigned long long>(properties.warpSize)) {
        findings.emplace_back("SUB_WARP");
    } else if (threads > 0 && properties.warpSize > 0 &&
               threads % static_cast<unsigned long long>(properties.warpSize) != 0) {
        findings.emplace_back("WARP_WASTE");
    }
    if (properties.multiProcessorCount > 0 && blocks > 0 &&
        blocks < static_cast<unsigned long long>(properties.multiProcessorCount)) {
        findings.emplace_back("SM_STARVED");
    }
    if (attributes_valid && attributes.localSizeBytes > 0) findings.emplace_back("SPILL");
    std::ostringstream row;
    row << "{\"type\":\"launch\",\"telemetry_schema\":\"native_ontology_v2\","
        << "\"ctx\":" << context_epoch()
        << ",\"cuda_device\":" << device
        << ",\"kernel\":\"" << json_escape(kernel ? kernel : "") << "\""
        << ",\"role\":\"" << json_escape(role ? role : "") << "\""
        << ",\"grid\":[" << grid.x << ',' << grid.y << ',' << grid.z << ']'
        << ",\"block\":[" << block.x << ',' << block.y << ',' << block.z << ']'
        << ",\"blocks\":" << blocks
        << ",\"threads_per_block\":" << threads
        << ",\"warp_fill\":" << json_number(
            properties.warpSize > 0 && threads > 0
                ? static_cast<double>(threads) /
                    (static_cast<double>((threads + properties.warpSize - 1) / properties.warpSize) * properties.warpSize)
                : 0.0)
        << ",\"dyn_smem\":" << dyn_smem
        << ",\"static_smem\":" << (attributes_valid ? attributes.sharedSizeBytes : 0)
        << ",\"regs_per_thread\":" << (attributes_valid ? attributes.numRegs : 0)
        << ",\"local_bytes\":" << (attributes_valid ? attributes.localSizeBytes : 0)
        << ",\"max_threads_per_block\":" << properties.maxThreadsPerBlock
        << ",\"sm_count\":" << properties.multiProcessorCount
        << ",\"findings\":[";
    for (std::size_t i = 0; i < findings.size(); ++i) {
        if (i) row << ',';
        row << "\"" << findings[i] << "\"";
    }
    row << "]}";
    emit_line(row.str());
}

inline void observe_step(const StepOutcome& outcome) {
    if (!enabled()) return;
    flush_launch_timing(outcome.opt_step);
    std::ostringstream row;
    row << "{\"type\":\"objective\",\"telemetry_schema\":\"native_ontology_v2\","
        << "\"ctx\":" << context_epoch()
        << ",\"opt_step\":" << outcome.opt_step
        << ",\"loss\":" << json_number(outcome.loss)
        << ",\"wallclock_s\":" << json_number(outcome.wallclock_s)
        << ",\"tokens\":" << outcome.tokens
        << ",\"update_skipped\":" << (outcome.update_skipped ? "true" : "false")
        << ",\"clip_fired\":" << (outcome.clip_fired ? "true" : "false")
        << ",\"grad_norm\":" << json_number(outcome.grad_norm)
        << ",\"timing_scope\":\"host_observation\",\"device_execution_timing\":false}";
    emit_line(row.str());
}

}  // namespace ida_native::ontology

#define IDA_LAUNCH(KERNEL, ROLE, GRID, BLOCK, SMEM, STREAM, ...)               \
    do {                                                                       \
        ::ida_native::ontology::observe(#KERNEL, (const void*)(KERNEL),       \
                                        (GRID), (BLOCK), (SMEM), (ROLE));      \
        if (::ida_native::ontology::timing_enabled()) {                       \
            const auto _ida_launch_t0 =                                        \
                std::chrono::steady_clock::now();                              \
            KERNEL<<<(GRID), (BLOCK), (SMEM), (STREAM)>>>(__VA_ARGS__);         \
            const auto _ida_launch_t1 =                                        \
                std::chrono::steady_clock::now();                              \
            ::ida_native::ontology::record_launch_enqueue(                     \
                #KERNEL, (ROLE), (GRID), (BLOCK), (SMEM),                      \
                static_cast<std::uint64_t>(                                    \
                    std::chrono::duration_cast<std::chrono::nanoseconds>(     \
                        _ida_launch_t1 - _ida_launch_t0).count()));            \
        } else {                                                                \
            KERNEL<<<(GRID), (BLOCK), (SMEM), (STREAM)>>>(__VA_ARGS__);         \
        }                                                                       \
    } while (0)
