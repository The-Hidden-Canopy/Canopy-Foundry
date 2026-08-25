// Ampere-only evidence probe: raw E4M3 weights are decoded into a BF16 tile
// immediately before an Ampere BF16 WMMA GEMM.  This is deliberately not a
// production dispatch.  It answers one question first: can the 1-byte weight
// read plus tile-local decode beat the current full-BF16 staging path on the
// actual Swift GEMM shapes?

#include "ida_native/fp8_e4m3.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

using BFloat16 = __nv_bfloat16;
using namespace nvcuda;

[[noreturn]] void fail(const char* what, cudaError_t status) {
    std::fprintf(stderr, "ampere_fp8_fused_gemm_bench: %s: %s\n",
                 what, cudaGetErrorString(status));
    std::exit(2);
}

[[noreturn]] void fail_cublas(const char* what, cublasStatus_t status) {
    std::fprintf(stderr, "ampere_fp8_fused_gemm_bench: %s: cuBLAS status %d\n",
                 what, static_cast<int>(status));
    std::exit(2);
}

#define CUDA_CHECK(call) do { \
    const cudaError_t _status = (call); \
    if (_status != cudaSuccess) fail(#call, _status); \
} while (0)

#define CUBLAS_CHECK(call) do { \
    const cublasStatus_t _status = (call); \
    if (_status != CUBLAS_STATUS_SUCCESS) fail_cublas(#call, _status); \
} while (0)

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &expanded, sizeof(value));
    return value;
}

struct DeviceBuffer {
    void* ptr{};
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t bytes) { CUDA_CHECK(cudaMalloc(&ptr, bytes)); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaFree(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    template <typename T> T* as() { return static_cast<T*>(ptr); }
};

__global__ void pack_scaled_e4m3_kernel(
    const BFloat16* input, std::uint8_t* packed, float scale, std::size_t n
) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        packed[i] = ida_native::fp8_e4m3::pack(__bfloat162float(input[i]) * scale).bits;
    }
}

__global__ void decode_e4m3_bf16_kernel(
    const std::uint8_t* packed, BFloat16* output, float descale, std::size_t n
) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        output[i] = __float2bfloat16(ida_native::fp8_e4m3::unpack(packed[i]) * descale);
    }
}

// B is stored row-major as [N,K].  A is row-major [M,K], so this computes
// C[M,N] = A[M,K] @ B[N,K]^T.  A single warp owns one 16x16 output tile.  The
// packed weight bytes are decoded into shared BF16 only for the current K tile;
// no full BF16 copy of B is created on this path.  A 32x64 block has eight
// warps (one 16x16 output tile each), so the packed B tile is decoded once and
// reused by two row tiles instead of being reloaded once per row tile.
__global__ __launch_bounds__(256) void fused_e4m3_bf16_nt_wmma_kernel(
    const BFloat16* a,
    const std::uint8_t* packed_b,
    float descale,
    float* c,
    int m,
    int n,
    int k
) {
    const int tile_m = static_cast<int>(blockIdx.y) * 32;
    const int tile_n = static_cast<int>(blockIdx.x) * 64;
    if (tile_m >= m || tile_n >= n) return;

    __shared__ BFloat16 a_tile[32 * 16];
    __shared__ BFloat16 b_tile[64 * 16];

    const int warp_id = static_cast<int>(threadIdx.x) / 32;
    const int warp_m = warp_id / 4;
    const int warp_n = warp_id % 4;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < k; k0 += 16) {
        for (int index = static_cast<int>(threadIdx.x); index < 64 * 16; index += 256) {
            const int row = index / 16;
            const int col = index % 16;
            const int a_row = tile_m + row;
            const int b_row = tile_n + row;
            const int a_col = k0 + col;
            const int b_col = k0 + col;
            if (row < 32) {
                a_tile[row * 16 + col] = (a_row < m && a_col < k)
                    ? a[a_row * k + a_col]
                    : __float2bfloat16(0.0f);
            }
            b_tile[index] = (b_row < n && b_col < k)
                ? __float2bfloat16(
                    ida_native::fp8_e4m3::unpack(packed_b[b_row * k + b_col]) * descale)
                : __float2bfloat16(0.0f);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, BFloat16, wmma::row_major> a_frag;
        // b_tile is physically [N_tile,K_tile] row-major.  Viewed as a
        // column-major [K_tile,N_tile] matrix, it is exactly B^T.
        wmma::fragment<wmma::matrix_b, 16, 16, 16, BFloat16, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, a_tile + warp_m * 16 * 16, 16);
        wmma::load_matrix_sync(b_frag, b_tile + warp_n * 16 * 16, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(
        c + (tile_m + warp_m * 16) * n + tile_n + warp_n * 16,
        c_frag, n, wmma::mem_row_major);
}

void check_device(int device) {
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    if (props.major != 8 || props.minor != 6) {
        std::fprintf(stderr,
            "ampere_fp8_fused_gemm_bench: requires sm_86, got sm_%d%d (%s)\n",
            props.major, props.minor, props.name);
        std::exit(3);
    }
    CUDA_CHECK(cudaSetDevice(device));
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return milliseconds;
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    float result = 0.0f;
    for (std::size_t i = 0; i < lhs.size(); ++i)
        result = std::max(result, std::fabs(lhs[i] - rhs[i]));
    return result;
}

struct Case {
    const char* name;
    int m;
    int n;
    int k;
};

void run_case(const Case& test_case, int iterations, int warmup) {
    const int m = test_case.m;
    const int n = test_case.n;
    const int k = test_case.k;
    if ((m % 32) != 0 || (n % 64) != 0 || (k % 16) != 0) {
        std::fprintf(stderr, "case %s must have M divisible by 32, N by 64, and K by 16\n", test_case.name);
        std::exit(4);
    }

    std::vector<std::uint16_t> host_a(static_cast<std::size_t>(m) * k);
    std::vector<std::uint16_t> host_b(static_cast<std::size_t>(n) * k);
    float amax = 0.0f;
    for (std::size_t i = 0; i < host_a.size(); ++i) {
        const float value = 0.125f * std::sin(static_cast<float>(i % 997) * 0.071f)
            + 0.03125f * std::cos(static_cast<float>(i % 193) * 0.113f);
        host_a[i] = float_to_bf16(value);
    }
    for (std::size_t i = 0; i < host_b.size(); ++i) {
        const float value = 0.25f * std::sin(static_cast<float>(i % 1009) * 0.037f)
            + 0.0625f * std::cos(static_cast<float>(i % 257) * 0.097f);
        host_b[i] = float_to_bf16(value);
        amax = std::max(amax, std::fabs(bf16_to_float(host_b[i])));
    }
    const float scale = 448.0f / std::max(amax, 1.0e-12f);
    const float descale = 1.0f / scale;
    const std::size_t a_bytes = host_a.size() * sizeof(BFloat16);
    const std::size_t b_bytes = host_b.size() * sizeof(BFloat16);
    const std::size_t packed_bytes = host_b.size() * sizeof(std::uint8_t);
    const std::size_t c_bytes = static_cast<std::size_t>(m) * n * sizeof(float);

    DeviceBuffer d_a(a_bytes);
    DeviceBuffer d_b_master(b_bytes);
    DeviceBuffer d_b_dequant(b_bytes);
    DeviceBuffer d_b_packed(packed_bytes);
    DeviceBuffer d_c_baseline(c_bytes);
    DeviceBuffer d_c_fused(c_bytes);
    CUDA_CHECK(cudaMemcpy(d_a.ptr, host_a.data(), a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_master.ptr, host_b.data(), b_bytes, cudaMemcpyHostToDevice));
    const int pack_blocks = std::max(1, static_cast<int>((host_b.size() + 255) / 256));
    pack_scaled_e4m3_kernel<<<pack_blocks, 256>>>(
        d_b_master.as<BFloat16>(), d_b_packed.as<std::uint8_t>(), scale, host_b.size());
    decode_e4m3_bf16_kernel<<<pack_blocks, 256>>>(
        d_b_packed.as<std::uint8_t>(), d_b_dequant.as<BFloat16>(), descale, host_b.size());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto baseline = [&] {
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            n, m, k, &alpha,
            d_b_dequant.as<BFloat16>(), CUDA_R_16BF, k,
            d_a.as<BFloat16>(), CUDA_R_16BF, k,
            &beta, d_c_baseline.as<float>(), CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    auto fused = [&] {
        fused_e4m3_bf16_nt_wmma_kernel<<<dim3((n + 63) / 64, (m + 31) / 32), 256, 0, stream>>>(
            d_a.as<BFloat16>(), d_b_packed.as<std::uint8_t>(), descale,
            d_c_fused.as<float>(), m, n, k);
        CUDA_CHECK(cudaGetLastError());
    };

    for (int i = 0; i < warmup; ++i) { baseline(); fused(); }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) baseline();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const float baseline_total_ms = elapsed_ms(start, stop);

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) fused();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const float fused_total_ms = elapsed_ms(start, stop);

    std::vector<float> baseline_host(static_cast<std::size_t>(m) * n);
    std::vector<float> fused_host(static_cast<std::size_t>(m) * n);
    CUDA_CHECK(cudaMemcpy(baseline_host.data(), d_c_baseline.ptr, c_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fused_host.data(), d_c_fused.ptr, c_bytes, cudaMemcpyDeviceToHost));
    const float error = max_abs_diff(baseline_host, fused_host);
    const double operations = 2.0 * static_cast<double>(m) * n * k;
    const double baseline_tflops = operations * iterations /
        (static_cast<double>(baseline_total_ms) * 1.0e9);
    const double fused_tflops = operations * iterations /
        (static_cast<double>(fused_total_ms) * 1.0e9);
    const double speedup_pct = (static_cast<double>(baseline_total_ms) /
        static_cast<double>(fused_total_ms) - 1.0) * 100.0;
    std::printf(
        "{\"case\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,"
        "\"iterations\":%d,\"baseline_ms\":%.6f,\"fused_ms\":%.6f,"
        "\"baseline_tflops\":%.6f,\"fused_tflops\":%.6f,"
        "\"fused_speedup_pct\":%.4f,\"max_abs_error\":%.7g,"
        "\"fp8_storage\":\"weights_e4m3\",\"compute\":\"bf16_wmma_fp32_accum\"}\n",
        test_case.name, m, n, k, iterations,
        baseline_total_ms / iterations, fused_total_ms / iterations,
        baseline_tflops, fused_tflops, speedup_pct, error);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(handle));
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    int iterations = 200;
    int warmup = 30;
    bool all_swift = true;
    Case single{"custom", 64, 512, 128};
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto next_int = [&](const char* name) -> int {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(1);
            }
            return std::atoi(argv[++i]);
        };
        if (arg == "--device") device = next_int("--device");
        else if (arg == "--iterations") iterations = next_int("--iterations");
        else if (arg == "--warmup") warmup = next_int("--warmup");
        else if (arg == "--m") { single.m = next_int("--m"); all_swift = false; }
        else if (arg == "--n") { single.n = next_int("--n"); all_swift = false; }
        else if (arg == "--k") { single.k = next_int("--k"); all_swift = false; }
        else if (arg == "--single") all_swift = false;
        else if (arg == "--help") {
            std::puts("usage: ampere_fp8_fused_gemm_bench [--device N] [--iterations N] [--warmup N] [--single --m M --n N --k K]");
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 1;
        }
    }
    if (iterations <= 0 || warmup < 0) return 1;
    check_device(device);
    if (all_swift) {
        run_case({"swift_qkv_o", 64, 128, 128}, iterations, warmup);
        run_case({"swift_ffn_up_gate", 64, 512, 128}, iterations, warmup);
        run_case({"swift_ffn_down", 64, 128, 512}, iterations, warmup);
    } else {
        run_case(single, iterations, warmup);
    }
    std::puts("ampere_fp8_fused_gemm_bench: PASS");
    return 0;
}
