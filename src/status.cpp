#include "ida_native/status.hpp"

#include "ida_native/device_info.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#ifndef IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
#define IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY 0
#endif

namespace ida_native {

std::string json_number(float value) {
    return std::isfinite(value) ? std::to_string(value) : "null";
}

std::string json_number(double value) {
    return std::isfinite(value) ? std::to_string(value) : "null";
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += ch; break;
        }
    }
    return escaped;
}

std::string now_utc_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const auto now_c = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now_c);
#else
    gmtime_r(&now_c, &utc);
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

void write_status(
    const std::filesystem::path& status_file,
    const NativeRequest& request,
    const std::string& phase,
    const std::vector<std::string>& extra_entries
) {
    std::ostringstream payload;
    const RuntimeDeviceInfo device = runtime_device_info();
    payload << "{\n";
    payload << "  \"backend\": \"native\",\n";
    payload << "  \"cuda_device\": " << device.device << ",\n";
    payload << "  \"compute_capability\": \""
            << compute_capability(device) << "\",\n";
    payload << "  \"phase\": \"" << json_escape(phase) << "\",\n";
    payload << "  \"seat\": \"" << json_escape(request.seat) << "\",\n";
    payload << "  \"family\": \"" << json_escape(request.family) << "\",\n";
    payload << "  \"version\": \"" << json_escape(request.version) << "\",\n";
    payload << "  \"engine_revision\": \"" << json_escape(request.engine_revision) << "\",\n";
    payload << "  \"attention_backend\": \"" << json_escape(request.attention_backend) << "\",\n";
    payload << "  \"precision_profile\": \"" << json_escape(request.precision_profile) << "\",\n";
    payload << "  \"hardware_profile\": \"" << json_escape(request.hardware_profile) << "\",\n";
    payload << "  \"memory_profile\": \"" << json_escape(request.memory_profile) << "\",\n";
    payload << "  \"peer_transport\": \"" << json_escape(
        request.device.peer_transport.empty() ? "auto" : request.device.peer_transport) << "\",\n";
    payload << "  \"fp8_storage\": \"" << json_escape(
        request.fp8_storage_format.empty() ? "none" : request.fp8_storage_format) << "\",\n";
    payload << "  \"fp8_compute\": \"" << json_escape(
        request.fp8_compute_path.empty() ? "none" : request.fp8_compute_path) << "\",\n";
    payload << "  \"native_fp8_tensorcore\": "
            << (request.precision_profile == "ampere_fp8_packed" ? "false" : "null") << ",\n";
#if IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
    payload << "  \"ontology_required\": " << (request.ontology_required ? "true" : "false") << ",\n";
    payload << "  \"analytics_required\": " << (request.analytics_required ? "true" : "false") << ",\n";
    payload << "  \"ontology_path\": \"" << json_escape(request.ontology_path.string()) << "\",\n";
    payload << "  \"analytics_path\": \"" << json_escape(request.analytics_path.string()) << "\",\n";
    payload << "  \"analytics_contract\": \"" << json_escape(request.analytics_contract) << "\",\n";
#endif
    payload << "  \"nvfp4_storage\": \"" << json_escape(
        request.nvfp4_storage_format.empty() ? "none" : request.nvfp4_storage_format) << "\",\n";
    payload << "  \"nvfp4_compute\": \"" << json_escape(
        request.nvfp4_compute_path.empty() ? "none" : request.nvfp4_compute_path) << "\",\n";
    payload << "  \"native_nvfp4_tensorcore\": null,\n";
    payload << "  \"optimizer_state_precision\": \"" << json_escape(request.optimizer_state_precision) << "\",\n";
    payload << "  \"gradient_buffer_precision\": \"" << json_escape(request.gradient_buffer_precision) << "\",\n";
    payload << "  \"gemm_accumulator_precision\": \"" << json_escape(request.gemm_accumulator_precision) << "\",\n";
    payload << "  \"architecture_compatibility\": \"" << json_escape(request.architecture_compatibility) << "\",\n";
    payload << "  \"spec_hash\": \"" << json_escape(request.spec_hash) << "\",\n";
    // optimizer_type existed on NativeRequest but was never echoed here --
    // the live .training_status.json never said whether a burn was running
    // Lion or AdamW. lion_policy echoes the REQUEST's own override fields
    // (not the worker's resolved-with-env-fallback values, which this file
    // has no access to) -- -1/-1.0f means "not overridden by this request,
    // the worker's own cached env value applies," which is itself useful
    // diagnostic signal for the exact multi-tenancy staleness this phase
    // exists to make visible (Phase 2, 2026-07-22).
    payload << "  \"optimizer_type\": \"" << json_escape(request.optimizer_type) << "\",\n";
    payload << "  \"lion_policy\": {\n";
    payload << "    \"lr_scale_override\": " << request.lion_lr_scale_override << ",\n";
    payload << "    \"wd_scale_override\": " << request.lion_wd_scale_override << ",\n";
    payload << "    \"beta1_override\": " << request.lion_beta1_override << ",\n";
    payload << "    \"beta2_override\": " << request.lion_beta2_override << ",\n";
    payload << "    \"trust_ratio_enabled_override\": " << request.lion_trust_ratio_enabled_override << ",\n";
    payload << "    \"trust_ratio_lo_override\": " << request.lion_trust_ratio_lo_override << ",\n";
    payload << "    \"trust_ratio_hi_override\": " << request.lion_trust_ratio_hi_override << "\n";
    payload << "  },\n";
    payload << "  \"promotion_eligible\": false,\n";
    payload << "  \"updated_at\": \"" << now_utc_iso8601() << "\"";
    for (const auto& entry : extra_entries) {
        payload << ",\n  " << entry;
    }
    payload << "\n}\n";

    std::filesystem::create_directories(status_file.parent_path());
    const auto tmp = status_file.string() + ".tmp";
    {
        std::ofstream output(tmp, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("unable to write temporary status file");
        }
        output << payload.str();
    }
    std::filesystem::rename(tmp, status_file);
}

}  // namespace ida_native
