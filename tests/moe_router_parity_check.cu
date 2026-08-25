// Native-side half of the sparse-MoE router parity check
// (scripts/native_parity_check.py is the PyTorch-side half). Runs native's
// exact router-pipeline kernel sequence (mirrors moe_router_forward in
// trainer.cu, which is file-static and not directly callable from here)
// against the same fixed weights/input the PyTorch reference used, on the
// PACKED case (one row, two samples via segs) -- the packing-aware pooling
// is the single highest-risk part of the port, so this is the case that
// matters most to prove.
//
// Not part of the main CMake build (no CMakeLists entry) -- it's a
// standalone verification tool, built and run manually against an already-
// built ida_native_train tree's object files:
//
//   python scripts/native_parity_check.py --export
//   nvcc -I native/include -I <build>/_deps/nlohmann_json-src/include \
//        -x cu -c native/tests/moe_router_parity_check.cu -o /tmp/parity.o -std=c++20
//   nvcc /tmp/parity.o <build>/CMakeFiles/ida_native_train.dir/**/*.o \
//        -o /tmp/parity_bin -lcublasLt -lcublas -lcudart -lcuda
//   /tmp/parity_bin native/tests/fixtures/moe_router_parity/weights.json \
//                   native/tests/fixtures/moe_router_parity/packed_case.json \
//                   /tmp/native_out.json
//   python scripts/native_parity_check.py --diff /tmp/native_out.json
//
// If moe_router_forward's kernel sequence in trainer.cu changes, update the
// sequence below to match -- this file intentionally duplicates it rather
// than calling it directly, since that function is file-static (internal
// linkage) and not part of the native/ public API surface.
//
// Step 4 addition: also reproduces moe_expert_bank_forward's sequence
// (trunk fc_in->GELU->fc_out unconditional, then each expert fc_in->GELU->
// fc_out accumulated by its route weight) -- gemm_bf16_nt itself is also
// file-static in trainer.cu, so its exact cublasGemmEx call is duplicated
// here too (row-major A[M,K] @ B^T, B stored [N,K]).
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include "ida_native/kernels.hpp"

using namespace ida_native;
using json = nlohmann::json;

static void gemm_bf16_nt(
    cublasHandle_t handle, int M, int N, int K, float alpha,
    const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb,
    float beta, __nv_bfloat16* C, int ldc
) {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                 B, CUDA_R_16BF, ldb, A, CUDA_R_16BF, lda, &beta,
                 C, CUDA_R_16BF, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

static __nv_bfloat16* to_device_bf16(const std::vector<float>& host) {
    __nv_bfloat16* d;
    std::vector<__nv_bfloat16> bf(host.size());
    for (std::size_t i = 0; i < host.size(); ++i) bf[i] = __float2bfloat16(host[i]);
    cudaMalloc(&d, bf.size() * sizeof(__nv_bfloat16));
    cudaMemcpy(d, bf.data(), bf.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    return d;
}

static std::vector<float> flatten2d(const json& j) {
    std::vector<float> out;
    for (const auto& row : j) for (const auto& v : row) out.push_back(v.get<float>());
    return out;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s weights.json packed_case.json out.json\n", argv[0]);
        return 1;
    }
    cudaStream_t s;
    cudaStreamCreate(&s);

    json weights = json::parse(std::ifstream(argv[1]));
    json packed = json::parse(std::ifstream(argv[2]));

    const int H = weights["H"].get<int>();
    const int num_routes = weights["num_routes"].get<int>();
    const int num_experts = weights["num_experts"].get<int>();
    const int top_k = weights["top_k"].get<int>();
    const float dominance_cap = weights["dominance_cap"].get<float>();
    const float minority_floor = weights["minority_floor"].get<float>();

    __nv_bfloat16* d_pressure_proj_w = to_device_bf16(flatten2d(weights["pressure_proj_w"]));   // [num_routes, H]
    __nv_bfloat16* d_pressure_mod_w  = to_device_bf16(flatten2d(weights["pressure_mod_w"]));     // [H, num_routes]
    __nv_bfloat16* d_router_score_w  = to_device_bf16(flatten2d(weights["router_score_w"]));     // [num_experts, H]
    __nv_bfloat16* d_pressure_to_routes_w = to_device_bf16(flatten2d(weights["pressure_to_routes_w"])); // [num_experts, num_routes]

    // Step 4: expert bank weights.
    const int trunk_I = weights["TRUNK_I"].get<int>();
    const int expert_I = weights["EXPERT_I"].get<int>();
    __nv_bfloat16* d_trunk_fc_in_w  = to_device_bf16(flatten2d(weights["trunk_fc_in_w"]));   // [trunk_I, H]
    __nv_bfloat16* d_trunk_fc_out_w = to_device_bf16(flatten2d(weights["trunk_fc_out_w"]));  // [H, trunk_I]
    std::vector<float> expert_fc_in_flat, expert_fc_out_flat;
    for (const auto& e : weights["expert_fc_in_w"]) for (const auto& row : e) for (const auto& v : row) expert_fc_in_flat.push_back(v.get<float>());
    for (const auto& e : weights["expert_fc_out_w"]) for (const auto& row : e) for (const auto& v : row) expert_fc_out_flat.push_back(v.get<float>());
    __nv_bfloat16* d_expert_fc_in_w  = to_device_bf16(expert_fc_in_flat);   // [num_experts, expert_I, H] contiguous
    __nv_bfloat16* d_expert_fc_out_w = to_device_bf16(expert_fc_out_flat);  // [num_experts, H, expert_I] contiguous

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, s);

    std::vector<float> hidden_flat;
    for (const auto& row : packed["hidden"][0]) for (const auto& v : row) hidden_flat.push_back(v.get<float>());
    const int S = static_cast<int>(packed["hidden"][0].size());
    const int B = 1;
    const std::size_t BS = static_cast<std::size_t>(B) * S;

    __nv_bfloat16* d_hidden = to_device_bf16(hidden_flat);
    std::vector<uint16_t> segs_host;
    for (const auto& v : packed["segs"]) segs_host.push_back(static_cast<uint16_t>(v.get<int>()));
    uint16_t* d_segs;
    cudaMalloc(&d_segs, segs_host.size() * sizeof(uint16_t));
    cudaMemcpy(d_segs, segs_host.data(), segs_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    // Scratch, mirroring StepBuffers' moe_* fields exactly.
    float *d_pool_scratch, *d_count_scratch, *d_pooled, *d_pressure, *d_modulation, *d_pooled2, *d_routed_pressure, *d_logits;
    __nv_bfloat16* d_hidden_gated;
    cudaMalloc(&d_pool_scratch, BS * H * sizeof(float));
    cudaMalloc(&d_count_scratch, BS * sizeof(float));
    cudaMalloc(&d_pooled, BS * H * sizeof(float));
    cudaMalloc(&d_pressure, BS * num_routes * sizeof(float));
    cudaMalloc(&d_modulation, BS * H * sizeof(float));
    cudaMalloc(&d_hidden_gated, BS * H * sizeof(__nv_bfloat16));
    cudaMalloc(&d_pooled2, BS * H * sizeof(float));
    cudaMalloc(&d_routed_pressure, BS * num_experts * sizeof(float));
    cudaMalloc(&d_logits, BS * num_experts * sizeof(float));

    // ── Exact same sequence as moe_router_forward (trainer.cu) ──────────────
    moe_pool_by_sample(d_hidden, d_segs, d_pool_scratch, d_count_scratch, B, S, H, s);
    moe_gather_by_sample(d_pool_scratch, d_segs, d_pooled, B, S, H, s);

    moe_small_proj_f32(d_pooled, d_pressure_proj_w, d_pressure, static_cast<int>(BS), H, num_routes, s);
    tanh_inplace_f32(d_pressure, BS * num_routes, s);

    moe_small_proj_f32(d_pressure, d_pressure_mod_w, d_modulation, static_cast<int>(BS), num_routes, H, s);
    sigmoid_inplace_f32(d_modulation, BS * H, s);

    moe_modulate_hidden_f32(d_hidden, d_modulation, d_hidden_gated, BS * H, s);

    moe_pool_by_sample(d_hidden_gated, d_segs, d_pool_scratch, d_count_scratch, B, S, H, s);
    moe_gather_by_sample(d_pool_scratch, d_segs, d_pooled2, B, S, H, s);

    moe_small_proj_f32(d_pooled2, d_router_score_w, d_logits, static_cast<int>(BS), H, num_experts, s);
    moe_small_proj_f32(d_pressure, d_pressure_to_routes_w, d_routed_pressure, static_cast<int>(BS), num_routes, num_experts, s);
    add_inplace_f32(d_logits, d_routed_pressure, BS * num_experts, s);

    moe_topk_route_f32(d_logits, static_cast<int>(BS), num_experts, top_k, s);
    moe_lateral_inhibition_f32(d_logits, static_cast<int>(BS), num_experts, dominance_cap, minority_floor, s);

    // ── Step 4: exact same sequence as moe_expert_bank_forward (trainer.cu) ─
    const int scratch_w = trunk_I > expert_I ? trunk_I : expert_I;
    __nv_bfloat16 *d_expert_scratch, *d_expert_out, *d_mixed;
    cudaMalloc(&d_expert_scratch, BS * scratch_w * sizeof(__nv_bfloat16));
    cudaMalloc(&d_expert_out, BS * H * sizeof(__nv_bfloat16));
    cudaMalloc(&d_mixed, BS * H * sizeof(__nv_bfloat16));

    gemm_bf16_nt(handle, static_cast<int>(BS), trunk_I, H, 1.f,
                 d_hidden_gated, H, d_trunk_fc_in_w, H, 0.f, d_expert_scratch, trunk_I);
    gelu_forward(d_expert_scratch, d_expert_scratch, BS * static_cast<std::size_t>(trunk_I), s);
    gemm_bf16_nt(handle, static_cast<int>(BS), H, trunk_I, 1.f,
                 d_expert_scratch, trunk_I, d_trunk_fc_out_w, trunk_I, 0.f, d_mixed, H);

    for (int e = 0; e < num_experts; ++e) {
        const __nv_bfloat16* fc_in_w  = d_expert_fc_in_w  + static_cast<std::size_t>(e) * expert_I * H;
        const __nv_bfloat16* fc_out_w = d_expert_fc_out_w + static_cast<std::size_t>(e) * H * expert_I;
        gemm_bf16_nt(handle, static_cast<int>(BS), expert_I, H, 1.f,
                     d_hidden_gated, H, fc_in_w, H, 0.f, d_expert_scratch, expert_I);
        gelu_forward(d_expert_scratch, d_expert_scratch, BS * static_cast<std::size_t>(expert_I), s);
        gemm_bf16_nt(handle, static_cast<int>(BS), H, expert_I, 1.f,
                     d_expert_scratch, expert_I, fc_out_w, expert_I, 0.f, d_expert_out, H);
        moe_scale_accumulate_bf16(d_mixed, d_expert_out, d_logits, static_cast<int>(BS), e, num_experts, H, s);
    }

    cudaStreamSynchronize(s);

    std::vector<float> route_weights(BS * num_experts);
    cudaMemcpy(route_weights.data(), d_logits, route_weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> pooled(BS * H);
    cudaMemcpy(pooled.data(), d_pooled, pooled.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<__nv_bfloat16> mixed_bf16(BS * H);
    cudaMemcpy(mixed_bf16.data(), d_mixed, mixed_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    std::vector<float> mixed(BS * H);
    for (std::size_t i = 0; i < mixed.size(); ++i) mixed[i] = __bfloat162float(mixed_bf16[i]);

    json out;
    for (int pos = 0; pos < static_cast<int>(BS); ++pos) {
        json row = json::array();
        for (int e = 0; e < num_experts; ++e) row.push_back(route_weights[pos * num_experts + e]);
        out["route_weights"].push_back(row);
    }
    for (int pos = 0; pos < static_cast<int>(BS); ++pos) {
        json row = json::array();
        for (int h = 0; h < H; ++h) row.push_back(pooled[pos * H + h]);
        out["pooled"].push_back(row);
    }
    for (int pos = 0; pos < static_cast<int>(BS); ++pos) {
        json row = json::array();
        for (int h = 0; h < H; ++h) row.push_back(mixed[pos * H + h]);
        out["mixed"].push_back(row);
    }
    std::ofstream(argv[3]) << out.dump(2);
    std::printf("wrote %s\n", argv[3]);
    return 0;
}
