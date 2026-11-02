#pragma once

#include "canopy_foundry/execution/service.hpp"
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>

namespace canopy_foundry {
// Operator-owned settings, never supplied by a request or Hub manifest.
// The first local owner admits the established single-device public CUDA
// from-scratch path only. It does not adapt private engine/graph schemas.
struct LocalBackendConfig {
    std::filesystem::path binary;
    std::string binary_sha256;
    std::filesystem::path run_root;
    int device{0};
    std::chrono::seconds maximum_runtime{3600};
    std::uint64_t maximum_output_bytes{8ull << 30};
    std::uint64_t maximum_log_bytes{32ull << 20};
};

// Windows Job Object containment; other platforms fail closed. Construction
// launches nothing. start() returns a preparation identity and asynchronously
// freezes inputs; only then may it launch a self-check. A running preparation
// is not a training or process-creation success;
// observe() launches the trainer only after the pinned probe's whole job is
// empty with exit 0. stopped is a process fact, never training success.
std::unique_ptr<OwnedRunBackend> make_local_native_backend(LocalBackendConfig config);
}
