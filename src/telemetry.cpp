#include "ida_native/telemetry.hpp"

#include "ida_native/device_info.hpp"

#include <fstream>
#include <sstream>

namespace ida_native {

namespace {

void write_text(const std::filesystem::path& path, const std::string& contents) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << contents;
}

}  // namespace

void write_training_contract_files(
    const NativeRequest& request,
    const SmokeStepResult& result
) {
    const std::string compute_cap = compute_capability();
    std::ostringstream manifest;
    manifest
        << "{\n"
        << "  \"backend\": \"native\",\n"
        << "  \"compute_capability\": \"" << compute_cap << "\",\n"
        << "  \"attention_backend\": \"" << request.attention_backend << "\",\n"
        << "  \"precision_profile\": \"" << request.precision_profile << "\",\n"
        << "  \"hardware_profile\": \"" << request.hardware_profile << "\",\n"
        << "  \"memory_profile\": \"" << request.memory_profile << "\",\n"
        << "  \"fp8_storage\": \"" << (request.fp8_storage_format.empty() ? "none" : request.fp8_storage_format) << "\",\n"
        << "  \"fp8_compute\": \"" << (request.fp8_compute_path.empty() ? "none" : request.fp8_compute_path) << "\",\n"
        << "  \"native_fp8_tensorcore\": " << (request.precision_profile == "ampere_fp8_packed" ? "false" : "null") << ",\n"
        << "  \"nvfp4_storage\": \"" << (request.nvfp4_storage_format.empty() ? "none" : request.nvfp4_storage_format) << "\",\n"
        << "  \"nvfp4_compute\": \"" << (request.nvfp4_compute_path.empty() ? "none" : request.nvfp4_compute_path) << "\",\n"
        << "  \"native_nvfp4_tensorcore\": null,\n"
        << "  \"optimizer_state_precision\": \"" << request.optimizer_state_precision << "\",\n"
        << "  \"gradient_buffer_precision\": \"" << request.gradient_buffer_precision << "\",\n"
        << "  \"gemm_accumulator_precision\": \"" << request.gemm_accumulator_precision << "\",\n"
        << "  \"engine_revision\": \"" << request.engine_revision << "\",\n"
        << "  \"architecture_compatibility\": \"" << request.architecture_compatibility << "\",\n"
        << "  \"promotion_eligible\": false,\n"
        << "  \"seat\": \"" << request.seat << "\",\n"
        << "  \"family\": \"" << request.family << "\",\n"
        << "  \"version\": \"" << request.version << "\"\n"
        << "}\n";
    write_text(request.output_dir / "training_manifest.json", manifest.str());

    std::ostringstream metrics;
    metrics
        << "{"
        << "\"backend\":\"native\","
        << "\"compute_capability\":\"" << compute_cap << "\","
        << "\"attention_backend\":\"" << request.attention_backend << "\","
        << "\"precision_profile\":\"" << request.precision_profile << "\","
        << "\"nvfp4_storage\":\"" << (request.nvfp4_storage_format.empty() ? "none" : request.nvfp4_storage_format) << "\","
        << "\"nvfp4_compute\":\"" << (request.nvfp4_compute_path.empty() ? "none" : request.nvfp4_compute_path) << "\","
        << "\"native_nvfp4_tensorcore\":null,"
        << "\"optimizer_state_precision\":\"" << request.optimizer_state_precision << "\","
        << "\"gradient_buffer_precision\":\"" << request.gradient_buffer_precision << "\","
        << "\"gemm_accumulator_precision\":\"" << request.gemm_accumulator_precision << "\","
        << "\"metric_kind\":\"native_smoke\","
        << "\"global_step\":" << result.global_step << ","
        << "\"tokens_processed\":" << result.tokens_processed << ","
        << "\"elapsed_seconds\":" << result.elapsed_seconds << ","
        << "\"tokens_per_second\":" << result.tokens_per_second << ","
        << "\"promotion_eligible\":false"
        << "}\n";
    write_text(request.output_dir / "native_smoke_metrics.jsonl", metrics.str());

    write_text(
        request.output_dir / "training_log_history.json",
        ("[{\"phase\":\"native_smoke_complete\",\"backend\":\"native\","
         "\"attention_backend\":\"" + request.attention_backend + "\","
         "\"precision_profile\":\"" + request.precision_profile + "\","
         "\"optimizer_state_precision\":\"" + request.optimizer_state_precision + "\","
         "\"gradient_buffer_precision\":\"" + request.gradient_buffer_precision + "\","
         "\"gemm_accumulator_precision\":\"" + request.gemm_accumulator_precision + "\"}]\n")
    );

    std::ostringstream loss_summary;
    loss_summary
        << "{\n"
        << "  \"backend\": \"native\",\n"
        << "  \"attention_backend\": \"" << request.attention_backend << "\",\n"
        << "  \"precision_profile\": \"" << request.precision_profile << "\",\n"
        << "  \"optimizer_state_precision\": \"" << request.optimizer_state_precision << "\",\n"
        << "  \"gradient_buffer_precision\": \"" << request.gradient_buffer_precision << "\",\n"
        << "  \"gemm_accumulator_precision\": \"" << request.gemm_accumulator_precision << "\",\n"
        << "  \"loss_mode\": \"not_available_for_smoke_vertical_slice\",\n"
        << "  \"promotion_eligible\": false\n"
        << "}\n";
    write_text(request.output_dir / "training_loss_summary.json", loss_summary.str());

    std::ostringstream optimizer_summary;
    optimizer_summary
        << "{\n"
        << "  \"backend\": \"native\",\n"
        << "  \"attention_backend\": \"" << request.attention_backend << "\",\n"
        << "  \"precision_profile\": \"" << request.precision_profile << "\",\n"
        << "  \"optimizer_state_precision\": \"" << request.optimizer_state_precision << "\",\n"
        << "  \"gradient_buffer_precision\": \"" << request.gradient_buffer_precision << "\",\n"
        << "  \"gemm_accumulator_precision\": \"" << request.gemm_accumulator_precision << "\",\n"
        // Was hardcoded "adamw_smoke_placeholder" unconditionally -- a Lion
        // burn's own telemetry falsely claimed AdamW (Phase 2, 2026-07-22).
        << "  \"optimizer\": \"" << request.optimizer_type << "\",\n"
        << "  \"global_step\": " << result.global_step << ",\n"
        << "  \"promotion_eligible\": false\n"
        << "}\n";
    write_text(
        request.output_dir / "native_smoke_optimizer_summary.json",
        optimizer_summary.str()
    );
}

}  // namespace ida_native
