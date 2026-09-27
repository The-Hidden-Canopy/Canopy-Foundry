#include "ida_native/telemetry.hpp"

#include "ida_native/device_info.hpp"
#include "ida_native/ontology.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <system_error>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <nlohmann/json.hpp>

namespace ida_native {

namespace {

using json = nlohmann::json;

void reject_linked_components(const std::filesystem::path& path, const char* label) {
    const auto candidate = std::filesystem::absolute(path).lexically_normal();
    for (auto current = candidate; !current.empty();) {
        std::error_code component_error;
        if (std::filesystem::is_symlink(current, component_error)) {
            throw std::runtime_error(std::string(label) + " must not pass through a link");
        }
        if (component_error &&
            component_error != std::make_error_code(std::errc::no_such_file_or_directory)) {
            throw std::runtime_error(std::string("unable to inspect ") + label);
        }
#if defined(_WIN32)
        const auto attributes = GetFileAttributesW(current.c_str());
        if (attributes != INVALID_FILE_ATTRIBUTES &&
            (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
            throw std::runtime_error(std::string(label) + " must not pass through a reparse point");
        }
        if (attributes == INVALID_FILE_ATTRIBUTES) {
            const auto win_error = GetLastError();
            if (win_error != ERROR_FILE_NOT_FOUND && win_error != ERROR_PATH_NOT_FOUND) {
                throw std::runtime_error(std::string("unable to inspect ") + label);
            }
        }
#endif
        const auto parent = current.parent_path();
        if (parent == current) break;
        current = parent;
    }
}

void write_text(const std::filesystem::path& path, const std::string& contents) {
    reject_linked_components(path, "telemetry contract path");
    std::error_code error;
    if (std::filesystem::is_symlink(path, error) || error) {
        throw std::runtime_error("telemetry contract path must not be a link");
    }
    error.clear();
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) throw std::runtime_error("unable to create telemetry contract directory");
    }
    reject_linked_components(path, "telemetry contract path");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("unable to open telemetry contract file");
    output << contents;
    output.flush();
    if (!output) throw std::runtime_error("unable to write telemetry contract file");
}

}  // namespace

struct Sha256 {
    std::array<std::uint8_t, 64> block{};
    std::size_t block_size{0};
    std::uint64_t bit_length{0};
    std::array<std::uint32_t, 8> state{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };

    static constexpr std::array<std::uint32_t, 64> k = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };

    static std::uint32_t rotr(std::uint32_t value, std::uint32_t amount) {
        return (value >> amount) | (value << (32u - amount));
    }

    void transform() {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t i = 0; i < 16; ++i) {
            words[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
                (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
                (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
                static_cast<std::uint32_t>(block[i * 4 + 3]);
        }
        for (std::size_t i = 16; i < words.size(); ++i) {
            const auto s0 = rotr(words[i - 15], 7) ^ rotr(words[i - 15], 18) ^ (words[i - 15] >> 3);
            const auto s1 = rotr(words[i - 2], 17) ^ rotr(words[i - 2], 19) ^ (words[i - 2] >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }
        std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        std::uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
        for (std::size_t i = 0; i < words.size(); ++i) {
            const auto s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const auto choose = (e & f) ^ ((~e) & g);
            const auto temp1 = h + s1 + choose + k[i] + words[i];
            const auto s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temp2 = s0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    void update(const std::uint8_t* bytes, std::size_t length) {
        for (std::size_t i = 0; i < length; ++i) {
            block[block_size++] = bytes[i];
            if (block_size == block.size()) {
                transform();
                bit_length += 512;
                block_size = 0;
            }
        }
    }

    std::string finish() {
        const auto original_bits = bit_length + static_cast<std::uint64_t>(block_size) * 8;
        block[block_size++] = 0x80;
        while (block_size != 56) {
            if (block_size == block.size()) {
                transform();
                block_size = 0;
            }
            block[block_size++] = 0;
        }
        for (int i = 7; i >= 0; --i) {
            block[block_size++] = static_cast<std::uint8_t>(original_bits >> (i * 8));
        }
        transform();
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (const auto word : state) out << std::setw(8) << word;
        return out.str();
    }
};

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to read kernel telemetry file");
    Sha256 digest;
    std::array<std::uint8_t, 1 << 16> buffer{};
    while (input) {
        input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) digest.update(buffer.data(), static_cast<std::size_t>(count));
    }
    return digest.finish();
}

std::uint64_t line_count(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return 0;
    std::uint64_t count = 0;
    std::string line;
    while (std::getline(input, line)) ++count;
    return count;
}

void write_shareable_kernel_telemetry(
    const std::filesystem::path& path,
    const NativeRequest& request,
    const SmokeStepResult& result
) {
    (void)request;
    reject_linked_components(path, "kernel telemetry output");
    std::error_code path_error;
    if (path.empty() || std::filesystem::is_symlink(path, path_error) || path_error ||
        (std::filesystem::exists(path) && !std::filesystem::is_directory(path))) {
        throw std::runtime_error("kernel telemetry output must be a directory");
    }
    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (error) throw std::runtime_error("unable to create kernel telemetry output directory");
    reject_linked_components(path, "kernel telemetry output");
    if (::ida_native::ontology::sink_is_file()) {
        if (std::fflush(::ida_native::ontology::sink()) != 0) {
            throw std::runtime_error("unable to flush kernel telemetry");
        }
    }
    if (::ida_native::ontology::telemetry_overflowed()) {
        throw std::runtime_error("kernel telemetry exceeded the bounded public export limit");
    }
    const auto ontology_path = path / "kernel_ontology.jsonl";
    reject_linked_components(ontology_path, "kernel telemetry ontology path");
    std::error_code ontology_error;
    if (std::filesystem::is_symlink(ontology_path, ontology_error) || ontology_error) {
        throw std::runtime_error("kernel telemetry ontology path must not be a link");
    }
    ontology_error.clear();
    if (!std::filesystem::is_regular_file(ontology_path, ontology_error) || ontology_error) {
        throw std::runtime_error("kernel telemetry was requested but no ontology records were written");
    }
    ontology_error.clear();
    const auto ontology_bytes = std::filesystem::file_size(ontology_path, ontology_error);
    if (ontology_error || ontology_bytes == 0 ||
        ontology_bytes > ::ida_native::ontology::kMaxTelemetryBytes) {
        throw std::runtime_error("kernel telemetry ontology file exceeds the bounded public export limit");
    }
    const auto metrics_path = path / "training_metrics.jsonl";
    const json metrics = {
        {"record_type", "native_kernel_telemetry_v2"},
        {"telemetry_schema", "native_kernel_telemetry_v2"},
        {"backend", "native"},
        {"global_step", result.global_step},
        {"optimizer_steps", result.optimizer_steps},
        {"tokens_processed", result.tokens_processed},
        {"elapsed_seconds", result.elapsed_seconds},
        {"tokens_per_second", result.tokens_per_second},
        {"promotion_eligible", false},
        {"profile_claim", "native_ontology_telemetry_only"},
        {"coverage_scope", "wrapped_public_launch_sites_only"},
        {"hardware_counter_profile", false},
        {"device_elapsed_kernel_time", false},
    };
    write_text(metrics_path, metrics.dump() + "\n");
    const json receipt = {
        {"schema_version", "native-kernel-telemetry.v2"},
        {"source", "canopy-foundry"},
        {"claims", {
            {"profile_claim", "native_ontology_telemetry_only"},
            {"coverage_scope", "wrapped_public_launch_sites_only"},
            {"external_profiler_used", false},
            {"hardware_counter_profile", false},
            {"device_elapsed_kernel_time", false},
            {"promotion_eligible", false},
        }},
        {"counts", {
            {"kernel_ontology_records", line_count(ontology_path)},
            {"training_metric_records", line_count(metrics_path)},
        }},
        {"files", {
            {"kernel_ontology.jsonl", {
                {"bytes", std::filesystem::file_size(ontology_path)},
                {"sha256", sha256_file(ontology_path)},
            }},
            {"training_metrics.jsonl", {
                {"bytes", std::filesystem::file_size(metrics_path)},
                {"sha256", sha256_file(metrics_path)},
            }},
        }},
    };
    write_text(path / "receipt.json", receipt.dump(2) + "\n");
}

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
