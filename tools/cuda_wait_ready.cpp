// cuda_wait_ready — polls the CUDA runtime until a context can be created,
// or a timeout elapses. Replaces the old `import torch; torch.cuda.is_available()`
// check in start_mps_from_hardware_envelope.sh: torch's own CUDA init is heavy
// enough that a single 10s-bounded attempt can lose the race against the MPS
// control daemon finishing startup, causing the setup script to declare CUDA
// dead and tear MPS back down when it would have come up fine a moment later.
//
// Usage: cuda_wait_ready [--timeout-seconds N] [--poll-ms N]
// Exit 0 + "CUDA_OK" on stdout once a context is confirmed live.
// Exit 1 + "CUDA_TIMEOUT: <last cuda error>" on stderr if the deadline passes.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

int main(int argc, char** argv) {
    int timeout_seconds = 10;
    int poll_ms = 250;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--timeout-seconds") == 0 && i + 1 < argc) {
            timeout_seconds = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--poll-ms") == 0 && i + 1 < argc) {
            poll_ms = std::atoi(argv[++i]);
        }
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
    cudaError_t last_err = cudaErrorUnknown;

    while (true) {
        int device_count = 0;
        last_err = cudaGetDeviceCount(&device_count);
        if (last_err == cudaSuccess && device_count > 0) {
            last_err = cudaSetDevice(0);
            if (last_err == cudaSuccess) {
                // Force actual context creation (cudaSetDevice alone is lazy).
                last_err = cudaFree(0);
                if (last_err == cudaSuccess) {
                    std::printf("CUDA_OK\n");
                    return 0;
                }
            }
        }
        // Clear sticky error state before the next attempt.
        cudaGetLastError();

        if (std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
    }

    std::fprintf(stderr, "CUDA_TIMEOUT: %s\n", cudaGetErrorString(last_err));
    return 1;
}
