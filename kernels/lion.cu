#include "ida_native/kernels.hpp"

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ida_native {

// Duplicated from adamw.cu rather than shared: it's declared __forceinline__
// there, and __forceinline__ device functions aren't reliably linkable
// across separate .cu translation units. Keep both copies byte-identical.
//
// Stochastic rounding FP32 -> BF16: add uniform noise below the truncation
// boundary before dropping the low 16 mantissa bits. Removes the
// deterministic round-to-nearest bias that compounds over thousands of
// direct-BF16 weight updates (no FP32 master copy in this engine).
__device__ __forceinline__ __nv_bfloat16 f32_to_bf16_sr(float x, unsigned rnd) {
    unsigned bits = __float_as_uint(x);
    if ((bits & 0x7f800000u) == 0x7f800000u)   // NaN/Inf: no rounding games
        return __float2bfloat16(x);
    bits += (rnd & 0xffffu);
    __nv_bfloat16_raw r;
    r.x = static_cast<unsigned short>(bits >> 16);
    return __nv_bfloat16(r);
}

// Lion (EvoLved Sign Momentum, Chen et al. 2023): single momentum buffer,
// sign-based update, no second moment, no bias correction (scale-invariant
// by construction). Two betas: beta1 blends the update DIRECTION, beta2
// updates the STORED momentum for next step -- these are deliberately
// different from AdamW's single-beta-pair role and must not be aliased to
// adamw_step's beta1/beta2 by a caller.
//   update = sign(beta1 * m + (1 - beta1) * g)
//   m_new  = beta2 * m + (1 - beta2) * g
//   w_new  = w - lr * (update + wd * w)          [decoupled weight decay]
//
// Fused stochastic-rounding BF16 weight write + optional FP8 amax pass,
// same as k_adamw_*_state (adamw.cu) -- neither is AdamW-specific.
__global__ void k_lion_fp32_state(
    __nv_bfloat16* w,
    float*         m,
    const float*   grad,
    std::size_t    n,
    float lr, float beta1, float beta2, float wd,
    float* amax_out,
    unsigned sr_seed
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool ok = (i < n);

    float w_new = 0.0f;
    if (ok) {
        const float g = grad[i];
        const float m_prev = m[i];
        const float blend = beta1 * m_prev + (1.0f - beta1) * g;
        const float update = (blend > 0.0f) - (blend < 0.0f);   // sign(), 0 at 0
        m[i] = beta2 * m_prev + (1.0f - beta2) * g;

        const float w_f32 = __bfloat162float(w[i]);
        w_new = w_f32 - lr * (update + wd * w_f32);

        unsigned h = static_cast<unsigned>(i) * 2654435761u ^ sr_seed;
        h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
        w[i] = f32_to_bf16_sr(w_new, h);
    }

    if (amax_out) {
        float local = ok ? fabsf(w_new) : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, off));
        if ((threadIdx.x & 31) == 0)
            atomicMax(reinterpret_cast<int*>(amax_out), __float_as_int(local));
    }
}

__global__ void k_lion_bf16_state(
    __nv_bfloat16* w,
    __nv_bfloat16* m,
    const float*   grad,
    std::size_t    n,
    float lr, float beta1, float beta2, float wd,
    float* amax_out,
    unsigned sr_seed
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool ok = (i < n);

    float w_new = 0.0f;
    if (ok) {
        const float g = grad[i];
        const float m_prev = __bfloat162float(m[i]);
        const float blend = beta1 * m_prev + (1.0f - beta1) * g;
        const float update = (blend > 0.0f) - (blend < 0.0f);
        const float m_new = beta2 * m_prev + (1.0f - beta2) * g;

        unsigned h = static_cast<unsigned>(i) * 2654435761u ^ sr_seed;
        h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
        m[i] = f32_to_bf16_sr(m_new, h ^ 0x9e3779b9u);

        const float w_f32 = __bfloat162float(w[i]);
        w_new = w_f32 - lr * (update + wd * w_f32);
        w[i] = f32_to_bf16_sr(w_new, h);
    }

    if (amax_out) {
        float local = ok ? fabsf(w_new) : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, off));
        if ((threadIdx.x & 31) == 0)
            atomicMax(reinterpret_cast<int*>(amax_out), __float_as_int(local));
    }
}

void lion_step(
    __nv_bfloat16* d_w,
    float*         d_m,
    const float*   d_grad,
    std::size_t    n,
    float lr, float beta1, float beta2, float wd,
    cudaStream_t stream,
    float*        d_weight_amax,
    unsigned      sr_seed
) {
    const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
    k_lion_fp32_state<<<blocks, 256, 0, stream>>>(
        d_w, d_m, d_grad, n, lr, beta1, beta2, wd, d_weight_amax, sr_seed
    );
}

void lion_step(
    __nv_bfloat16* d_w,
    __nv_bfloat16* d_m,
    const float*   d_grad,
    std::size_t    n,
    float lr, float beta1, float beta2, float wd,
    cudaStream_t stream,
    float*        d_weight_amax,
    unsigned      sr_seed
) {
    const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
    k_lion_bf16_state<<<blocks, 256, 0, stream>>>(
        d_w, d_m, d_grad, n, lr, beta1, beta2, wd, d_weight_amax, sr_seed
    );
}

}  // namespace ida_native
