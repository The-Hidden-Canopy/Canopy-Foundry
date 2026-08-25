#pragma once

// Public builds expose no private GEMM/pack telemetry implementation.  These
// no-op contracts keep the public trainer buildable without copying the
// deployment-owned observability headers into the release tree.

#include <cstdio>

namespace ida_native::gemm_trace {

enum Role : int {
    R_ATTN_QKV = 0,
    R_ATTN_OUT,
    R_MOE_ROUTER,
    R_MOE_TRUNK_IN,
    R_MOE_TRUNK_OUT,
    R_MOE_EXPERT_IN,
    R_MOE_EXPERT_OUT,
    R_MOE_EXPERT_DPOST,
    R_MOE_EXPERT_DW_IN,
    R_MOE_EXPERT_DW_OUT,
    R_FFN_DENSE,
    R_LM_HEAD,
    R_PSS_PRED,
    R_LRSS,
    R_REGION_LAYER_FWD,
    R_REGION_MOE_FWD,
    R_REGION_MOE_BWD,
    R_REGION_BWD_ACCUM,
    R_OTHER,
    R_ATTN_WGMMA_QK,
    R_ATTN_WGMMA_PV,
    R_COUNT
};

inline bool enabled() { return false; }
inline void reset() {}
inline void record_current(long long, long long, long long) {}
inline void record_current_batched(long long, long long, long long, long long) {}
inline void dump(std::FILE*, double, int) {}

struct ScopedRole {
    explicit ScopedRole(int) {}
    ScopedRole(const ScopedRole&) = delete;
    ScopedRole& operator=(const ScopedRole&) = delete;
};

}  // namespace ida_native::gemm_trace

#define IDA_GEMM_ROLE(R) ::ida_native::gemm_trace::ScopedRole _ida_gemm_role_##__LINE__(R)
