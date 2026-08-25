#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

namespace ida_native {

struct NativeArena {
    int device_id{0};
    cudaStream_t stream{nullptr};
    cudaMemPool_t pool{nullptr};
    std::size_t release_threshold{0};
    std::size_t reserved_bytes{0};   // pool primed with this much at startup
};

// requested_reserve_bytes: -1 keeps the legacy env/default policy; >=0 is a
// per-request override, including 0 to disable pre-reservation.
NativeArena create_arena(int device_id, std::int64_t requested_reserve_bytes = -1);
void destroy_arena(NativeArena& arena);

// Native two-stage model parallel transport. These helpers deliberately use
// CUDA peer access and cudaMemcpyPeerAsync directly: the engine has no NCCL
// dependency and does not create replicated data-parallel model copies.
bool cuda_peer_access_supported(int source_device, int destination_device,
                                std::string* error = nullptr);
void enable_cuda_peer_access(int source_device, int destination_device);
void enable_bidirectional_cuda_peer_access(int first_device, int second_device);
void cuda_peer_copy_async(
    void* destination,
    int destination_device,
    const void* source,
    int source_device,
    std::size_t bytes,
    cudaStream_t destination_stream,
    bool force_host_staged = false
);

// cudaMallocFromPoolAsync with the PyTorch caching-allocator fallback: on
// OOM, sync the stream, trim THIS pool (empty_cache), and retry once before
// failing. `pool` is always the owning arena's explicit pool -- MPS
// replacement (single process, multiple concurrent bodies) requires every
// allocation to be bound to its own body's pool rather than routed through
// cudaMallocAsync's implicit per-device default, which is shared/global and
// would let two bodies' arenas silently fight over one pool.
cudaError_t ida_malloc_async(void** ptr, std::size_t size, cudaMemPool_t pool, cudaStream_t stream);

// Typed-pointer shim mirroring cudaMallocAsync's template overload.
template <typename T>
cudaError_t ida_malloc_async(T** ptr, std::size_t size, cudaMemPool_t pool, cudaStream_t stream) {
    return ida_malloc_async(reinterpret_cast<void**>(ptr), size, pool, stream);
}

}  // namespace ida_native
