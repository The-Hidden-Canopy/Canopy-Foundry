#include "ida_native/cpu_kernels.hpp"

#include <cmath>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#endif

namespace ida_native {

namespace {

bool detect_avx2_fma() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int regs[4]{};
    __cpuid(regs, 0);
    if (regs[0] < 7) return false;
    __cpuidex(regs, 1, 0);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    const bool fma = (regs[2] & (1 << 12)) != 0;
    if (!osxsave || !avx || !fma) return false;
    const unsigned __int64 xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) return false;
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;  // EBX.AVX2
#else
    return false;
#endif
}

bool avx2_fma_available() {
    static const bool value = detect_avx2_fma();
    return value;
}

float scalar_dot(const float* left, const float* right, std::size_t count) {
    float result = 0.0f;
    for (std::size_t i = 0; i < count; ++i) result += left[i] * right[i];
    return result;
}

#if defined(_MSC_VER)
__declspec(noinline) float avx2_dot(const float* left, const float* right, std::size_t count) {
    std::size_t i = 0;
    __m256 accumulator = _mm256_setzero_ps();
    for (; i + 8 <= count; i += 8) {
        const __m256 a = _mm256_loadu_ps(left + i);
        const __m256 b = _mm256_loadu_ps(right + i);
        accumulator = _mm256_fmadd_ps(a, b, accumulator);
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, accumulator);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                   lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < count; ++i) result += left[i] * right[i];
    return result;
}
#endif

}  // namespace

const char* cpu_kernel_variant() {
    return avx2_fma_available() ? "amd_avx2_fma" : "scalar";
}

float cpu_dot(const float* left, const float* right, std::size_t count) {
#if defined(_MSC_VER)
    if (avx2_fma_available()) return avx2_dot(left, right, count);
#endif
    return scalar_dot(left, right, count);
}

}  // namespace ida_native
