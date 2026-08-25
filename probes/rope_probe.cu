// Standalone correctness probe for k_rope_apply_bf16 (native/src/trainer.cu).
//
// Two independent checks, neither of which trusts the other:
//   1. Round-trip: forward rotation (angle_sign=+1) followed by the
//      backward/adjoint rotation (angle_sign=-1) must return the exact
//      original values, to bf16 precision. This holds for ANY correct
//      implementation of an orthogonal rotation's forward+transpose pair,
//      independent of whether the "rotate half" formula itself was copied
//      correctly from anywhere -- it is a property, not a memorized answer.
//   2. Host reference: an independently-written host implementation of the
//      same Llama-style "rotate half" RoPE formula, compared element-by-
//      element against the forward kernel's output.
//   3. Packed-position check: two samples packed into one row (segs != 0
//      for the second) must rotate by DIFFERENT effective positions than
//      an unpacked row of the same absolute index, proving the
//      seg-relative position reset actually fires.
//
// The kernel below is a deliberate copy of trainer.cu's k_rope_apply_bf16,
// not an include -- keeps this probe buildable standalone the same way
// mxf4_gemm_probe.cu/nvfp4_gemm_probe.cu are. That means it can drift out
// of sync with the real kernel silently; if trainer.cu's RoPE math changes,
// re-copy it here before trusting a PASS from this file again.
//
// Build:
//   nvcc -std=c++20 -gencode arch=compute_120a,code=sm_120a rope_probe.cu -o rope_probe
// Or via CMake: cmake --build native/build --target rope_probe
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

// Exact copy of native/src/trainer.cu's k_rope_apply_bf16 -- kept identical
// on purpose so this probe tests the real kernel logic, not a paraphrase.
__global__ void k_rope_apply_bf16(
    __nv_bfloat16* __restrict__ x,
    const std::uint16_t* __restrict__ segs,
    int B, int nH, int S, int Hd,
    float rope_theta, float angle_sign
) {
    const int half = Hd / 2;
    const long total = static_cast<long>(B) * nH * S * half;
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int i  = static_cast<int>(idx % half); idx /= half;
    const int s  = static_cast<int>(idx % S);    idx /= S;
    const int nh = static_cast<int>(idx % nH);   idx /= nH;
    const int b  = static_cast<int>(idx);

    const int seg_start = segs
        ? static_cast<int>(segs[static_cast<std::size_t>(b) * S + s])
        : 0;
    const int pos = s - seg_start;

    const float theta_i = powf(rope_theta, -2.0f * static_cast<float>(i) / static_cast<float>(Hd));
    const float angle = angle_sign * static_cast<float>(pos) * theta_i;
    const float c = cosf(angle);
    const float sn = sinf(angle);

    __nv_bfloat16* row = x +
        ((static_cast<std::size_t>(b) * nH + nh) * S + s) * Hd;
    const float x1 = __bfloat162float(row[i]);
    const float x2 = __bfloat162float(row[i + half]);
    row[i]        = __float2bfloat16(x1 * c - x2 * sn);
    row[i + half] = __float2bfloat16(x2 * c + x1 * sn);
}

bool cuda_ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA FAIL %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

void launch(
    __nv_bfloat16* d_x, const std::uint16_t* d_segs,
    int B, int nH, int S, int Hd, float theta, float sign
) {
    const long total = static_cast<long>(B) * nH * S * (Hd / 2);
    const int threads = 256;
    const long blocks = (total + threads - 1) / threads;
    k_rope_apply_bf16<<<static_cast<unsigned>(blocks), threads>>>(
        d_x, d_segs, B, nH, S, Hd, theta, sign);
}

// Independent host reference: same formula, written separately from the
// device kernel (not copy-pasted), to catch a transcription bug the
// round-trip test alone couldn't (round-trip passes even if forward and
// backward share the SAME wrong formula).
void host_rope_reference(
    std::vector<float>& x,  // [B, nH, S, Hd], row-major, modified in place
    const std::vector<std::uint16_t>& segs, bool has_segs,
    int B, int nH, int S, int Hd, float theta
) {
    const int half = Hd / 2;
    for (int b = 0; b < B; ++b) {
        for (int nh = 0; nh < nH; ++nh) {
            for (int s = 0; s < S; ++s) {
                const int seg_start = has_segs ? static_cast<int>(segs[b * S + s]) : 0;
                const int pos = s - seg_start;
                float* row = &x[((static_cast<std::size_t>(b) * nH + nh) * S + s) * Hd];
                for (int i = 0; i < half; ++i) {
                    const double theta_i = std::pow(static_cast<double>(theta),
                        -2.0 * i / static_cast<double>(Hd));
                    const double angle = pos * theta_i;
                    const double c = std::cos(angle);
                    const double sn = std::sin(angle);
                    const double x1 = row[i];
                    const double x2 = row[i + half];
                    row[i]        = static_cast<float>(x1 * c - x2 * sn);
                    row[i + half] = static_cast<float>(x2 * c + x1 * sn);
                }
            }
        }
    }
}

float bf16_round(float v) { return __bfloat162float(__float2bfloat16(v)); }

}  // namespace

int main() {
    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    std::printf("device %d %s\n", device, prop.name);

    const int B = 2, nH = 4, S = 16, Hd = 64;
    const float theta = 130000.0f;  // SmolLM2's real rope_theta
    const std::size_t n = static_cast<std::size_t>(B) * nH * S * Hd;

    // Deterministic pseudo-random input, bf16-rounded so the round-trip
    // comparison isn't polluted by input values the format can't represent.
    // Deliberately a function of (nh, s, hd) only, NOT b -- check 3 below
    // compares the same (nh, s=10) row across two different batch items and
    // needs their input data to be provably identical, so any difference in
    // the rotated output is attributable to position alone.
    std::vector<float> h_orig(n);
    for (int b = 0; b < B; ++b) {
        for (int nh = 0; nh < nH; ++nh) {
            for (int s = 0; s < S; ++s) {
                for (int hd = 0; hd < Hd; ++hd) {
                    const long k_shape_indep =
                        (static_cast<long>(nh) * S + s) * Hd + hd;
                    const float v = std::sin(static_cast<float>(k_shape_indep) * 0.017f) * 3.0f
                                   + std::cos(static_cast<float>(k_shape_indep) * 0.101f);
                    h_orig[((static_cast<std::size_t>(b) * nH + nh) * S + s) * Hd + hd] = bf16_round(v);
                }
            }
        }
    }

    // segs: sample 0 occupies rows [0,10), sample 1 occupies rows [10,16)
    // for batch item 0 only; batch item 1 is fully unpacked (segs all 0).
    // Row 10 in batch 0 and row 10 in batch 1 therefore must rotate by
    // DIFFERENT effective positions (0 vs 10) -- the check below verifies
    // that actually happens.
    std::vector<std::uint16_t> h_segs(static_cast<std::size_t>(B) * S, 0);
    for (int s = 10; s < S; ++s) h_segs[0 * S + s] = 10;

    __nv_bfloat16* d_x = nullptr;
    std::uint16_t* d_segs = nullptr;
    cuda_ok(cudaMalloc(&d_x, n * sizeof(__nv_bfloat16)), "cudaMalloc(d_x)");
    cuda_ok(cudaMalloc(&d_segs, h_segs.size() * sizeof(std::uint16_t)), "cudaMalloc(d_segs)");

    std::vector<__nv_bfloat16> h_x_bf16(n);
    for (std::size_t k = 0; k < n; ++k) h_x_bf16[k] = __float2bfloat16(h_orig[k]);
    cuda_ok(cudaMemcpy(d_x, h_x_bf16.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "H2D x");
    cuda_ok(cudaMemcpy(d_segs, h_segs.data(), h_segs.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice), "H2D segs");

    // ---- Check 1: forward vs independent host reference ----
    launch(d_x, d_segs, B, nH, S, Hd, theta, +1.0f);
    cuda_ok(cudaDeviceSynchronize(), "sync after forward");
    std::vector<__nv_bfloat16> h_fwd(n);
    cuda_ok(cudaMemcpy(h_fwd.data(), d_x, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "D2H fwd");

    std::vector<float> h_ref(h_orig);
    host_rope_reference(h_ref, h_segs, true, B, nH, S, Hd, theta);

    double max_abs_diff = 0.0;
    int bad_forward = 0;
    for (std::size_t k = 0; k < n; ++k) {
        const double got = __bfloat162float(h_fwd[k]);
        const double ref = bf16_round(h_ref[k]);
        const double diff = std::fabs(got - ref);
        max_abs_diff = std::max(max_abs_diff, diff);
        if (diff > 0.02) ++bad_forward;  // bf16 has ~2-3 decimal digits
    }
    std::printf("[forward vs host reference] max_abs_diff=%.6f bad=%d/%zu %s\n",
        max_abs_diff, bad_forward, n, bad_forward == 0 ? "PASS" : "FAIL");

    // ---- Check 2: round-trip forward then backward returns the original ----
    launch(d_x, d_segs, B, nH, S, Hd, theta, -1.0f);
    cuda_ok(cudaDeviceSynchronize(), "sync after backward");
    std::vector<__nv_bfloat16> h_roundtrip(n);
    cuda_ok(cudaMemcpy(h_roundtrip.data(), d_x, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "D2H roundtrip");

    double max_roundtrip_diff = 0.0;
    int bad_roundtrip = 0;
    for (std::size_t k = 0; k < n; ++k) {
        const double got = __bfloat162float(h_roundtrip[k]);
        const double orig = h_orig[k];
        const double diff = std::fabs(got - orig);
        max_roundtrip_diff = std::max(max_roundtrip_diff, diff);
        if (diff > 0.02) ++bad_roundtrip;
    }
    std::printf("[round-trip fwd+bwd == identity] max_abs_diff=%.6f bad=%d/%zu %s\n",
        max_roundtrip_diff, bad_roundtrip, n, bad_roundtrip == 0 ? "PASS" : "FAIL");

    // ---- Check 3: packed-position reset actually changes the rotation ----
    // Re-run forward fresh (state was already un-rotated back to original
    // by check 2) and compare row s=10 of batch 0 (pos should reset to 0)
    // against row s=10 of batch 1 (pos stays 10, unpacked). If seg-relative
    // positioning is wired correctly these must differ; if segs were
    // silently ignored they would be identical.
    cuda_ok(cudaMemcpy(d_x, h_x_bf16.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "H2D x (re-seed)");
    launch(d_x, d_segs, B, nH, S, Hd, theta, +1.0f);
    cuda_ok(cudaDeviceSynchronize(), "sync after forward (check 3)");
    std::vector<__nv_bfloat16> h_fwd2(n);
    cuda_ok(cudaMemcpy(h_fwd2.data(), d_x, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "D2H fwd2");

    const int nh_probe = 0;
    const std::size_t row_b0_s10 = ((static_cast<std::size_t>(0) * nH + nh_probe) * S + 10) * Hd;
    const std::size_t row_b1_s10 = ((static_cast<std::size_t>(1) * nH + nh_probe) * S + 10) * Hd;
    double packed_vs_unpacked_diff = 0.0;
    for (int i = 0; i < Hd; ++i) {
        const double d = std::fabs(static_cast<double>(__bfloat162float(h_fwd2[row_b0_s10 + i]))
            - static_cast<double>(__bfloat162float(h_fwd2[row_b1_s10 + i])));
        packed_vs_unpacked_diff = std::max(packed_vs_unpacked_diff, d);
    }
    // Inputs at these two rows are identical (same h_orig pattern indexed
    // by absolute row), so ANY difference here is entirely attributable to
    // the position reset -- not to different input data.
    const bool packed_reset_fires = packed_vs_unpacked_diff > 0.05;
    std::printf("[packed-position reset differs from unpacked] diff=%.6f %s\n",
        packed_vs_unpacked_diff, packed_reset_fires ? "PASS (positions differ as expected)" : "FAIL (segs ignored)");

    cudaFree(d_x);
    cudaFree(d_segs);

    const bool all_pass = bad_forward == 0 && bad_roundtrip == 0 && packed_reset_fires;
    std::printf("%s\n", all_pass ? "PROBE PASS" : "PROBE FAIL");
    return all_pass ? 0 : 1;
}
