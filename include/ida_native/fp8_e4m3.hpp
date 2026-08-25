#pragma once

#include <cstdint>
#include <cmath>

#if defined(__CUDACC__)
#define IDA_FP8_HD __host__ __device__
#else
#define IDA_FP8_HD
#endif

namespace ida_native::fp8_e4m3 {

struct Packed {
    std::uint8_t bits{0};
    bool finite{true};
};

IDA_FP8_HD inline int round_nearest_even(float value) {
    const float lower_f = floorf(value);
    const int lower = static_cast<int>(lower_f);
    const float fraction = value - lower_f;
    if (fraction > 0.5f || (fraction == 0.5f && (lower & 1))) return lower + 1;
    return lower;
}

// NVIDIA's finite E4M3 encoding: exponent 0 is subnormal, exponent 15 is
// finite for mantissas 0..6, and 0x7f is NaN. The largest finite value is
// 448 (0x7e). This is intentionally implemented from the byte contract rather
// than through __nv_fp8_e4m3 so it remains usable on sm_86 as storage only.
IDA_FP8_HD inline Packed pack(float value) {
    const bool negative = signbit(value) != 0;
    const std::uint8_t sign = negative ? 0x80u : 0u;
    if (!isfinite(value)) return {static_cast<std::uint8_t>(sign | 0x7fu), false};
    const float magnitude = fabsf(value);
    if (magnitude == 0.0f) return {sign, true};
    if (magnitude >= 448.0f) return {static_cast<std::uint8_t>(sign | 0x7eu), true};

    // E4M3 subnormals are mantissa * 2^-9. Round-to-nearest-even is shared
    // with the normalized path so boundary behavior is deterministic on host
    // and device.
    if (magnitude < 0.015625f) { // 2^-6, the smallest normal
        int mantissa = round_nearest_even(magnitude * 512.0f);
        if (mantissa <= 0) return {sign, true};
        if (mantissa >= 8) return {static_cast<std::uint8_t>(sign | 0x08u), true};
        return {static_cast<std::uint8_t>(sign | static_cast<std::uint8_t>(mantissa)), true};
    }

    const int exponent = static_cast<int>(floorf(log2f(magnitude)));
    const float unit = scalbnf(1.0f, exponent);
    int mantissa = round_nearest_even((magnitude / unit - 1.0f) * 8.0f);
    int encoded_exponent = exponent + 7;
    if (mantissa >= 8) {
        mantissa = 0;
        ++encoded_exponent;
    }
    if (encoded_exponent >= 15) {
        // exp=15/mant=6 is the finite ceiling; mant=7 is reserved for NaN.
        if (encoded_exponent > 15 || mantissa > 6) {
            return {static_cast<std::uint8_t>(sign | 0x7eu), true};
        }
    }
    if (encoded_exponent <= 0) return {sign, true};
    return {static_cast<std::uint8_t>(
                sign | (static_cast<std::uint8_t>(encoded_exponent) << 3) |
                static_cast<std::uint8_t>(mantissa)), true};
}

IDA_FP8_HD inline float unpack(std::uint8_t bits) {
    const float sign = (bits & 0x80u) ? -1.0f : 1.0f;
    const int exponent = (bits >> 3) & 0x0f;
    const int mantissa = bits & 0x07;
    if (exponent == 0) return sign * scalbnf(static_cast<float>(mantissa), -9);
    if (exponent == 15 && mantissa == 7) return NAN;
    if (exponent == 15) return sign * (1.0f + static_cast<float>(mantissa) / 8.0f) * 256.0f;
    return sign * (1.0f + static_cast<float>(mantissa) / 8.0f) *
        scalbnf(1.0f, exponent - 7);
}

}  // namespace ida_native::fp8_e4m3

#undef IDA_FP8_HD
