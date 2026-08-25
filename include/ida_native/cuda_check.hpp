#pragma once

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace ida_native {

inline void cuda_check(cudaError_t code, const char* expr, const char* file, int line) {
    if (code == cudaSuccess) {
        return;
    }
    throw std::runtime_error(
        std::string("CUDA failure at ") + file + ":" + std::to_string(line) +
        " for " + expr + ": " + cudaGetErrorString(code)
    );
}

// A context poisoned by a fatal/sticky CUDA error (illegal address, ECC
// failure, launch failure, ...) returns that SAME error on every subsequent
// call for the rest of the process's life -- there is no in-process
// recovery, only destroying the process and starting a fresh context.
// 2026-07-23, found live: an Xid 31 GPU MMU fault silently poisoned the
// shared multi-body worker process; nothing detected it for over 6 hours
// (every per-job exception was treated as "this one job failed", never as
// "the whole context is dead"), during which every OTHER concurrently
// hosted body kept being served from an officially undefined-behavior
// context with no warning. cudaFree(nullptr) is documented as a safe
// no-op that still round-trips through the live context/driver, making it
// a cheap, real probe for "can this context still talk to the device at
// all" -- not just bookkeeping state.
inline bool cuda_context_is_healthy() {
    cudaGetLastError();  // clear any stale non-fatal error so it can't mask the probe
    return cudaFree(nullptr) == cudaSuccess;
}

}  // namespace ida_native

#define IDA_CUDA_CHECK(expr) ::ida_native::cuda_check((expr), #expr, __FILE__, __LINE__)
