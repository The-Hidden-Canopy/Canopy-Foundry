#pragma once

// Public builds expose no private pack telemetry implementation.  The native
// trainer keeps the call sites source-compatible, but all observability stays
// in the deployment-owned private overlay.

#include <cstdio>
#include <cstddef>

namespace ida_native::pack_trace {

enum Kind : int {
    K_FP4_PACK = 0,
    K_FP4_UNPACK,
    K_FP8_PACK,
    K_FP8_PACK_TRANSPOSE,
    K_FP8_PACK_FUSED_TRANSPOSE,
    K_FP8_UNPACK,
    K_FP4_PACK_DEQUANT,
    K_FP8_PACK_DEQUANT,
    K_COUNT
};

inline bool enabled() { return false; }
inline void reset() {}
inline void record(Kind, std::size_t, std::size_t, std::size_t) {}
inline void dump(std::FILE*, int) {}

}  // namespace ida_native::pack_trace
