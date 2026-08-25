#pragma once

// Public builds expose a no-op observability boundary only.  The implementation
// is supplied by the deployment-owned private include overlay when explicitly
// enabled; no private paths, files, or telemetry records are emitted here.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <string>

namespace ida_native::ontology {

struct Context {
    int microbatch = -1;
    int grad_accum = -1;
    int num_experts = -1;
    int top_k_experts = -1;
    int seq_window = -1;
    std::string precision_profile;
    std::string attention_backend;
    std::string optimizer;
    int resident_bodies = -1;
    std::string body_key;
};

struct StepOutcome {
    long long opt_step = -1;
    double loss = 0.0;
    double wallclock_s = 0.0;
    long long tokens = 0;
    bool update_skipped = false;
    bool clip_fired = false;
    double grad_norm = 0.0;
};

inline bool enabled() { return false; }
inline std::FILE* sink() { return nullptr; }
inline bool sink_is_file() { return false; }
inline int context_epoch() { return -1; }
inline void set_context(const Context&) {}
inline void observe_step(const StepOutcome&) {}
inline void observe(const char*, const void*, dim3, dim3, std::size_t, const char*) {}

}  // namespace ida_native::ontology

#define IDA_LAUNCH(KERNEL, ROLE, GRID, BLOCK, SMEM, STREAM, ...)              \
    do {                                                                       \
        ::ida_native::ontology::observe(#KERNEL, (const void*)(KERNEL),       \
                                        (GRID), (BLOCK), (SMEM), (ROLE));      \
        KERNEL<<<(GRID), (BLOCK), (SMEM), (STREAM)>>>(__VA_ARGS__);            \
    } while (0)
