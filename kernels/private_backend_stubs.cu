#include "ida_native/kernels.hpp"

#include <stdexcept>

namespace ida_native {
namespace {

[[noreturn]] void reject_private_backend() {
    throw std::runtime_error(
        "private advanced runtime is not included in the public Neural Foundry binary");
}

}  // namespace

// These symbols keep the public trainer linkable while making accidental
// direct selection fail closed. The implementation package that owns the
// instruction-level kernels is selected by an opaque deployment binding, not
// by a public request, environment variable, or source include.
void nvfp4_pack_rows_bf16(
    const __nv_bfloat16*, std::uint8_t*, std::uint8_t*, int, int,
    std::uint32_t*, cudaStream_t
) { reject_private_backend(); }

void nvfp4_pack_transpose_bf16(
    const __nv_bfloat16*, std::uint8_t*, std::uint8_t*, int, int,
    std::uint32_t*, cudaStream_t
) { reject_private_backend(); }

void nvfp4_unpack_rows_bf16(
    const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*, int, int,
    cudaStream_t
) { reject_private_backend(); }

void wgmma_flash_forward_hd64(
    const std::uint8_t*, const float*, const float*, float*, float*, float*,
    float*, float*, int, const __nv_bfloat16*, __nv_bfloat16*, float*, int,
    int, int, float, cudaStream_t, const std::uint8_t*, const std::uint8_t*,
    const float*, const float*, const std::uint16_t*, int
) { reject_private_backend(); }

void wgmma_flash_bwd_dkv_hd64(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const float*, const float*, const std::uint16_t*,
    int, __nv_bfloat16*, __nv_bfloat16*, int, int, int, float, cudaStream_t
) { reject_private_backend(); }

void wgmma_flash_bwd_dq(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const float*, const float*, const std::uint16_t*,
    int, __nv_bfloat16*, int, int, int, float, cudaStream_t
) { reject_private_backend(); }

void wgmma_flash_forward_bf16src(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, float*,
    float*, __nv_bfloat16*, float*, int, int, int, float,
    const std::uint16_t*, int, cudaStream_t
) { reject_private_backend(); }

void roundtrip_f32_through_e4m3(float*, std::size_t, cudaStream_t) {
    reject_private_backend();
}

void roundtrip_bf16_through_e4m3(__nv_bfloat16*, std::size_t, cudaStream_t) {
    reject_private_backend();
}

void roundtrip_bf16_through_scaled_e4m3(
    __nv_bfloat16*, std::size_t, cudaStream_t
) { reject_private_backend(); }

void fp8_e4m3_pair_to_bf16(
    const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*,
    std::size_t, cudaStream_t
) { reject_private_backend(); }

void fp4_scale_from_amax(
    const float*, float*, float*, cudaStream_t
) { reject_private_backend(); }

void fp4_pack_pair_e2m1_record(
    const __nv_bfloat16*, const __nv_bfloat16*, std::uint8_t*, const float*,
    const float*, float*, float*, std::size_t, cudaStream_t
) { reject_private_backend(); }

void fp4_scale_from_amax_e2m1(
    const float*, float*, float*, cudaStream_t
) { reject_private_backend(); }

void fp4_pack_pair_e2m1_true_record(
    const __nv_bfloat16*, const __nv_bfloat16*, std::uint8_t*, const float*,
    const float*, float*, float*, std::size_t, cudaStream_t
) { reject_private_backend(); }

void fp4_scale_from_amax_int2(
    const float*, float*, float*, cudaStream_t
) { reject_private_backend(); }

void fp4_pack_pair_int2_record(
    const __nv_bfloat16*, const __nv_bfloat16*, std::uint8_t*, const float*,
    const float*, float*, float*, std::size_t, cudaStream_t
) { reject_private_backend(); }

void fp4_scale_from_amax_gauss(
    const float*, float*, float*, cudaStream_t
) { reject_private_backend(); }

void fp4_pack_pair_gauss_record(
    const __nv_bfloat16*, const __nv_bfloat16*, std::uint8_t*, const float*,
    const float*, float*, float*, std::size_t, cudaStream_t
) { reject_private_backend(); }

void fp4_amax_pair(
    const __nv_bfloat16*, const __nv_bfloat16*, float*, float*, std::size_t,
    cudaStream_t
) { reject_private_backend(); }

void fp4_scale_from_stats_centered(
    const float*, const float*, float, float*, float*, float*, cudaStream_t
) { reject_private_backend(); }

void fp4_scale_from_stats_centered_smooth(
    const float*, const float*, float, float*, float*, float*, cudaStream_t
) { reject_private_backend(); }

void fp4_pack_pair_centered_record(
    const __nv_bfloat16*, const __nv_bfloat16*, std::uint8_t*, const float*,
    const float*, const float*, const float*, float*, float*, float*, float*,
    std::size_t, cudaStream_t
) { reject_private_backend(); }

void fp4_scale_from_stats_centered_int2(
    const float*, const float*, float, float*, float*, float*, cudaStream_t
) { reject_private_backend(); }

void fp4_pack_pair_centered_int2_record(
    const __nv_bfloat16*, const __nv_bfloat16*, std::uint8_t*, const float*,
    const float*, const float*, const float*, float*, float*, float*, float*,
    std::size_t, cudaStream_t
) { reject_private_backend(); }

void fp4_unpack_pair_to_bf16(
    const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*, const float*,
    const float*, std::size_t, cudaStream_t, const float*, const float*
) { reject_private_backend(); }

void fp4_unpack_pair_to_bf16_via_e4m3(
    const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*, const float*,
    const float*, std::size_t, cudaStream_t, const float*, const float*
) { reject_private_backend(); }

void fp4_unpack_pair_to_f32(
    const std::uint8_t*, float*, float*, const float*, const float*,
    std::size_t, cudaStream_t
) { reject_private_backend(); }

// The public scalar attention path owns the bounded health surface. The
// advanced implementation is not allowed to contribute private counters.
void attn_bwd_health_reset(cudaStream_t stream) {
    attn_bwd_health_reset_scalar(stream);
}

void attn_bwd_health_read(
    double* p_zero_frac, float* drift_max, cudaStream_t
) {
    if (p_zero_frac) *p_zero_frac = 0.0;
    if (drift_max) *drift_max = 0.0f;
}

}  // namespace ida_native
