#pragma once

// Runtime device facts are lineage, not a build-time guess.  Cache properties
// per device so status/telemetry/trace writers share one probe while remaining
// correct when model-parallel workers change the current CUDA device.

#include <cuda_runtime.h>

#include <map>
#include <mutex>
#include <string>

namespace ida_native {

struct RuntimeDeviceInfo {
    int device{-1};
    int major{-1};
    int minor{-1};
    bool valid{false};
};

inline RuntimeDeviceInfo runtime_device_info(int requested_device = -1) {
    int device = requested_device;
    if (device < 0 && cudaGetDevice(&device) != cudaSuccess) {
        return {};
    }

    static std::mutex cache_mutex;
    static std::map<int, RuntimeDeviceInfo> cache;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto it = cache.find(device);
        if (it != cache.end()) return it->second;
    }

    RuntimeDeviceInfo info;
    info.device = device;
    int major = -1;
    int minor = -1;
    if (cudaDeviceGetAttribute(
            &major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
        cudaDeviceGetAttribute(
            &minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess) {
        info.major = major;
        info.minor = minor;
        info.valid = true;
    }

    // Do not cache an unavailable probe as if it were a device fact. A later
    // status write may run after CUDA initialization and should be allowed to
    // retry instead of inheriting a stale "unknown" result.
    if (!info.valid) return info;

    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto [it, inserted] = cache.emplace(device, info);
    return inserted ? info : it->second;
}

inline std::string compute_capability(const RuntimeDeviceInfo& info) {
    if (!info.valid) return "unknown";
    return std::to_string(info.major) + "." + std::to_string(info.minor);
}

inline std::string compute_capability(int requested_device = -1) {
    return compute_capability(runtime_device_info(requested_device));
}

}  // namespace ida_native
