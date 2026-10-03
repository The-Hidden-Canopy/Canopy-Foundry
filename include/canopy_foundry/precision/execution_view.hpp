#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string_view>

namespace canopy_foundry {

using TensorId = std::uint64_t;

// Generation zero is reserved for an uninitialized tensor.  Optimizer or
// checkpoint publication code must advance the authoritative generation
// before a packed view can be admitted for reuse.
struct TensorGeneration {
    std::uint64_t value{0};

    [[nodiscard]] bool valid() const noexcept { return value != 0; }
    friend bool operator==(const TensorGeneration&, const TensorGeneration&) = default;
};

enum class ExecutionRole : std::uint8_t {
    Forward,
    BackwardDx,
    BackwardDwOperand,
    ExpertForward,
    ExpertDx,
    ExpertDw,
};

std::string_view execution_role_name(ExecutionRole role) noexcept;

// These are representation identities, not claims that a device can execute
// the format.  Hardware/build admission remains the planner's responsibility.
enum class NumericFormat : std::uint8_t {
    Unknown,
    F32,
    BF16,
    FP8E4M3,
    MXFP8E4M3UE8M0K32,
    NVFP4E2M1UE4M3,
};

std::string_view numeric_format_name(NumericFormat format) noexcept;
NumericFormat numeric_format_from_name(std::string_view name) noexcept;

enum class PackedOrientation : std::uint8_t {
    RowMajor,
    ColumnMajor,
    Transposed,
};

struct PackedLayout {
    std::uint32_t rows{0};
    std::uint32_t columns{0};
    std::uint16_t tile_rows{0};
    std::uint16_t tile_columns{0};
    PackedOrientation orientation{PackedOrientation::RowMajor};

    friend bool operator==(const PackedLayout&, const PackedLayout&) = default;
    friend auto operator<=>(const PackedLayout&, const PackedLayout&) = default;
};

struct ExecutionViewKey {
    TensorId tensor{0};
    ExecutionRole role{ExecutionRole::Forward};
    NumericFormat format{NumericFormat::Unknown};
    PackedLayout layout{};

    friend bool operator==(const ExecutionViewKey&, const ExecutionViewKey&) = default;
    friend auto operator<=>(const ExecutionViewKey&, const ExecutionViewKey&) = default;
};

// The execution core owns the actual backend allocation.  The contract keeps
// only an opaque identity here so CPU planning/tests cannot accidentally
// retain a raw CUDA pointer or assume a particular allocator.
struct PackedExecutionView {
    ExecutionViewKey key{};
    TensorGeneration source_generation{};
    std::uint64_t storage_token{0};
    std::size_t payload_bytes{0};
    std::size_t scale_bytes{0};
    bool ready{false};

    [[nodiscard]] bool valid_for(
        const ExecutionViewKey& expected_key,
        TensorGeneration expected_generation
    ) const noexcept;
};

struct ExecutionViewCacheStats {
    std::uint64_t builds{0};
    std::uint64_t reuses{0};
    std::uint64_t stale_rejections{0};
    std::uint64_t unavailable_rejections{0};
    std::uint64_t invalidations{0};
};

class ExecutionViewError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

using ExecutionViewBuilder = std::function<PackedExecutionView(
    const ExecutionViewKey&, TensorGeneration)>;

// Generation-aware cache contract extracted from the V2 execution model.
// It is deliberately backend-neutral: CUDA/OpenCL/CPU adapters provide the
// builder and own the storage represented by storage_token.
class ExecutionViewCache {
public:
    [[nodiscard]] const PackedExecutionView* lookup(
        const ExecutionViewKey& key,
        TensorGeneration generation
    ) const noexcept;

    const PackedExecutionView& require(
        const ExecutionViewKey& key,
        TensorGeneration generation,
        const ExecutionViewBuilder& builder
    );

    // Optimizer mutation or a newly published checkpoint calls this before
    // any role may reuse the previous generation's packed representation.
    void invalidate(TensorId tensor, TensorGeneration new_generation);
    void invalidate_all();

    [[nodiscard]] std::size_t size() const noexcept { return views_.size(); }
    [[nodiscard]] bool empty() const noexcept { return views_.empty(); }
    [[nodiscard]] const ExecutionViewCacheStats& stats() const noexcept {
        return stats_;
    }

private:
    std::map<ExecutionViewKey, PackedExecutionView> views_;
    ExecutionViewCacheStats stats_{};
};

}  // namespace canopy_foundry
