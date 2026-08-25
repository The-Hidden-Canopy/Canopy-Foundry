#include "ida_native/arena.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#include "ida_native/cuda_check.hpp"

namespace ida_native {

NativeArena create_arena(int device_id, std::int64_t requested_reserve_bytes) {
    NativeArena arena{};
    arena.device_id = device_id;
    arena.release_threshold = std::numeric_limits<std::size_t>::max();

    IDA_CUDA_CHECK(cudaSetDevice(device_id));
    IDA_CUDA_CHECK(cudaStreamCreateWithFlags(&arena.stream, cudaStreamNonBlocking));
    cudaMemPoolProps pool_props{};
    pool_props.allocType = cudaMemAllocationTypePinned;
    pool_props.handleTypes = cudaMemHandleTypeNone;
    pool_props.location.type = cudaMemLocationTypeDevice;
    pool_props.location.id = device_id;
    IDA_CUDA_CHECK(cudaMemPoolCreate(&arena.pool, &pool_props));
    IDA_CUDA_CHECK(cudaMemPoolSetAttribute(
        arena.pool,
        cudaMemPoolAttrReleaseThreshold,
        &arena.release_threshold
    ));
    // Deliberately NOT calling cudaDeviceSetMemPool: that sets the
    // device-WIDE default pool, which a second concurrent arena in the same
    // process (MPS-replacement design) would silently steal or fight over.
    // Every allocation site now takes arena.pool explicitly instead.

    // Prime the pool with one contiguous slab (PyTorch-style reservation).
    // The release threshold is max, so the freed slab stays in the pool and
    // every later cudaMallocAsync carves from it instead of growing the pool
    // through the driver — incremental growth is where fragmentation spikes
    // come from.  The slab only needs to cover the working set (Edge runs at
    // ~6 GiB), not the whole card.  IDA_ARENA_RESERVE_GB sets the slab size
    // (default 8; 0 disables); IDA_ARENA_RESERVE_PCT overrides as a fraction
    // of free VRAM for the AI family or other large bodies.
    std::size_t want = 8ull << 30;
    if (requested_reserve_bytes >= 0) {
        want = static_cast<std::size_t>(requested_reserve_bytes);
    } else if (const char* env = std::getenv("IDA_ARENA_RESERVE_GB")) {
        try { want = static_cast<std::size_t>(std::stod(env) * (1ull << 30)); } catch (...) {}
    }
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        if (requested_reserve_bytes < 0) {
            if (const char* env = std::getenv("IDA_ARENA_RESERVE_PCT")) {
            try {
                const double pct = std::stod(env);
                if (pct > 0.0 && pct <= 0.98)
                    want = static_cast<std::size_t>(static_cast<double>(free_b) * pct);
            } catch (...) {}
            }
        }
        // Never prime beyond what's actually free (leave 5% slack).
        const std::size_t cap = static_cast<std::size_t>(static_cast<double>(free_b) * 0.95);
        if (want > cap) want = cap;
    }
    if (want > 0) {
        void* slab = nullptr;
        if (cudaMallocFromPoolAsync(&slab, want, arena.pool, arena.stream) == cudaSuccess) {
            IDA_CUDA_CHECK(cudaFreeAsync(slab, arena.stream));
            IDA_CUDA_CHECK(cudaStreamSynchronize(arena.stream));
            arena.reserved_bytes = want;
            std::fprintf(stderr,
                "[ida_native_train] arena primed: %.2f GiB reserved\n",
                static_cast<double>(want) / (1024.0 * 1024.0 * 1024.0));
        } else {
            cudaGetLastError();  // clear; run unprimed
        }
    }
    return arena;
}

bool cuda_peer_access_supported(
    int source_device,
    int destination_device,
    std::string* error
) {
    if (source_device == destination_device) {
        if (error) *error = "peer devices must be distinct";
        return false;
    }
    int can_access = 0;
    const cudaError_t status = cudaDeviceCanAccessPeer(
        &can_access, destination_device, source_device);
    if (status != cudaSuccess) {
        if (error) {
            *error = "cudaDeviceCanAccessPeer(" + std::to_string(destination_device) +
                ", " + std::to_string(source_device) + ") failed: " +
                cudaGetErrorString(status);
        }
        return false;
    }
    if (!can_access && error) {
        *error = "CUDA peer access is unavailable from device " +
            std::to_string(destination_device) + " to device " +
            std::to_string(source_device);
    }
    return can_access != 0;
}

void enable_cuda_peer_access(int source_device, int destination_device) {
    std::string error;
    if (!cuda_peer_access_supported(source_device, destination_device, &error)) {
        throw std::runtime_error(error);
    }
    IDA_CUDA_CHECK(cudaSetDevice(destination_device));
    const cudaError_t status = cudaDeviceEnablePeerAccess(source_device, 0);
    if (status == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return;
    }
    IDA_CUDA_CHECK(status);
}

void enable_bidirectional_cuda_peer_access(int first_device, int second_device) {
    enable_cuda_peer_access(first_device, second_device);
    enable_cuda_peer_access(second_device, first_device);
}

// Two-hop copy through a reusable pinned host buffer, for hardware without
// CUDA P2P (consumer Blackwell/GeForce drivers do not expose P2P between
// cards even over PCIe with both GPUs on the same NUMA node -- verified via
// nvidia-smi topo -m + cudaDeviceCanAccessPeer on a real 2x RTX 5090 box,
// 2026-08-13; NODE-class link, no NV# entry). Fully synchronous by design:
// this is a correctness-first fallback for hardware the fast P2P path
// cannot run on at all, not a performance path, so it pays for a full
// destination_stream drain up front rather than risk a subtle
// stream-ordering bug from trying to keep it async without a
// source-device-native stream to enqueue the D2H leg on.
static void cuda_peer_copy_host_staged(
    void* destination,
    int destination_device,
    const void* source,
    int source_device,
    std::size_t bytes,
    cudaStream_t destination_stream
) {
    static void* staging = nullptr;
    static std::size_t staging_bytes = 0;
    if (bytes > staging_bytes) {
        if (staging) IDA_CUDA_CHECK(cudaFreeHost(staging));
        IDA_CUDA_CHECK(cudaHostAlloc(&staging, bytes, cudaHostAllocDefault));
        staging_bytes = bytes;
    }
    // Anything already enqueued on destination_stream that the source data
    // depends on (e.g. a cudaStreamWaitEvent bridging the producing stage's
    // completion) must finish before the D2H leg reads `source`.
    IDA_CUDA_CHECK(cudaSetDevice(destination_device));
    IDA_CUDA_CHECK(cudaStreamSynchronize(destination_stream));
    IDA_CUDA_CHECK(cudaSetDevice(source_device));
    IDA_CUDA_CHECK(cudaMemcpy(staging, source, bytes, cudaMemcpyDeviceToHost));
    IDA_CUDA_CHECK(cudaSetDevice(destination_device));
    IDA_CUDA_CHECK(cudaMemcpy(destination, staging, bytes, cudaMemcpyHostToDevice));
}

void cuda_peer_copy_async(
    void* destination,
    int destination_device,
    const void* source,
    int source_device,
    std::size_t bytes,
    cudaStream_t destination_stream,
    bool force_host_staged
) {
    if (bytes == 0) return;
    if (destination == nullptr || source == nullptr) {
        throw std::runtime_error("cuda peer copy received a null buffer");
    }
    if (force_host_staged) {
        cuda_peer_copy_host_staged(
            destination, destination_device, source, source_device, bytes,
            destination_stream);
        return;
    }
    int can_access = 0;
    const cudaError_t peer_status = cudaDeviceCanAccessPeer(
        &can_access, destination_device, source_device);
    if (peer_status == cudaSuccess && can_access) {
        IDA_CUDA_CHECK(cudaSetDevice(destination_device));
        IDA_CUDA_CHECK(cudaMemcpyPeerAsync(
            destination,
            destination_device,
            source,
            source_device,
            bytes,
            destination_stream
        ));
        return;
    }
    cudaGetLastError();  // clear the sticky error from the capability probe
    cuda_peer_copy_host_staged(
        destination, destination_device, source, source_device, bytes,
        destination_stream);
}

cudaError_t ida_malloc_async(void** ptr, std::size_t size, cudaMemPool_t pool, cudaStream_t stream) {
    cudaError_t err = cudaMallocFromPoolAsync(ptr, size, pool, stream);
    if (err != cudaErrorMemoryAllocation) return err;
    // PyTorch caching-allocator fallback: drain in-flight frees, release
    // THIS pool's cached-but-unused memory back to the driver, retry once.
    //
    // 2026-07-30: the old version called a blind cudaGetLastError() here to
    // "clear the sticky OOM" -- but cudaGetLastError() clears whatever the
    // CURRENT sticky error is, not specifically confirming it really was a
    // benign, recoverable OOM. If an earlier, unrelated kernel had already
    // poisoned the context with a genuinely fatal error (illegal address,
    // ECC failure -- see cuda_context_is_healthy()'s own comment on this
    // exact class of bug), blindly clearing here would silently launder
    // that fatal state into what looks like a normal, recovered OOM retry
    // -- masking the real fault and pushing its eventual surfacing to
    // whatever unrelated call happens to hit the next context sync, which
    // is exactly why "the reported crash site" kept moving between runs
    // that otherwise looked identical. Probe real context health before
    // treating this as recoverable; if the context is actually poisoned,
    // fail loudly here instead of retrying into a corrupted allocation.
    if (!cuda_context_is_healthy()) {
        std::fprintf(stderr,
            "[ida_malloc_async] requested %.2f MiB, retry skipped -- CUDA "
            "context is unhealthy (likely poisoned by an earlier fault, not "
            "a real capacity shortfall)\n",
            static_cast<double>(size) / (1024.0 * 1024.0));
        return err;
    }
    cudaStreamSynchronize(stream);
    if (pool != nullptr) {
        cudaMemPoolTrimTo(pool, 0);
    }
    const cudaError_t retry_err = cudaMallocFromPoolAsync(ptr, size, pool, stream);
    if (retry_err == cudaErrorMemoryAllocation) {
        // The generic alloc_f32/alloc_bf16/alloc_u8 helpers all funnel through
        // this one choke point with an IDA_CUDA_CHECK that reports ITS OWN
        // call site, not the caller's -- so a bare "out of memory" here is
        // genuinely ambiguous about which of dozens of allocation sites
        // failed, and whether the driver actually agrees memory is low
        // (real capacity shortfall) versus something else entirely
        // (fragmentation, or a poisoned-but-still-"healthy"-per-cudaFree
        // context). Answer both questions here instead of leaving the next
        // person to re-derive them from scratch.
        std::size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        std::size_t used_b = 0, reserved_b = 0;
        if (pool != nullptr) {
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used_b);
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved_b);
        }
        std::fprintf(stderr,
            "[ida_malloc_async] FAILED requested=%.2f MiB after pool trim + "
            "retry. Driver: free=%.2f MiB / total=%.2f MiB. This pool: "
            "used=%.2f MiB reserved=%.2f MiB.\n",
            static_cast<double>(size) / (1024.0 * 1024.0),
            static_cast<double>(free_b) / (1024.0 * 1024.0),
            static_cast<double>(total_b) / (1024.0 * 1024.0),
            static_cast<double>(used_b) / (1024.0 * 1024.0),
            static_cast<double>(reserved_b) / (1024.0 * 1024.0));
    }
    return retry_err;
}

void destroy_arena(NativeArena& arena) {
    // CUDA stream and pool handles are owned by the arena's device. Direct
    // two-stage execution destroys device 1 first and then device 0, so the
    // caller's current device cannot be assumed to match either handle.
    IDA_CUDA_CHECK(cudaSetDevice(arena.device_id));
    if (arena.stream != nullptr) {
        cudaStreamDestroy(arena.stream);
        arena.stream = nullptr;
    }
    if (arena.pool != nullptr) {
        cudaMemPoolDestroy(arena.pool);
        arena.pool = nullptr;
    }
}

}  // namespace ida_native
