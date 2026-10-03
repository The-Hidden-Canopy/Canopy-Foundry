#pragma once

#include "canopy_foundry/moe/route_plan.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string_view>

namespace canopy_foundry {

enum class ExpertDispatchMode : std::uint8_t {
    DevicePaddedPersistent,
    DeviceCompact,
    DiagnosticFixed,
};

std::string_view expert_dispatch_mode_name(ExpertDispatchMode mode) noexcept;

struct SchedulerKey {
    RouteGeneration route_generation{};
    int experts{0};
    int max_rows{0};
    int n_tiles{0};
    int k_chunks{0};
    int orientation{0};

    friend bool operator==(const SchedulerKey&, const SchedulerKey&) = default;
    friend auto operator<=>(const SchedulerKey&, const SchedulerKey&) = default;

    [[nodiscard]] bool valid() const noexcept {
        return route_generation.valid() && experts > 0 && max_rows > 0 &&
            n_tiles > 0 && k_chunks > 0 && orientation >= 0;
    }
};

// Backend-owned schedule/workspace identity. The token fields keep the public
// contract independent of CUDA allocation types while preserving the V2 rule
// that a schedule can only be reused for the exact route-generation key.
struct SchedulerWorkspace {
    SchedulerKey key{};
    std::uint64_t workspace_token{0};
    std::uint64_t schedule_token{0};
    std::size_t tile_capacity{0};
    std::size_t expert_capacity{0};
    bool schedule_valid{false};

    [[nodiscard]] bool valid_for(const SchedulerKey& expected) const noexcept;
};

struct SchedulerWorkspaceStats {
    std::uint64_t builds{0};
    std::uint64_t reuses{0};
    std::uint64_t stale_rejections{0};
    std::uint64_t unavailable_rejections{0};
    std::uint64_t invalidations{0};
};

class SchedulerWorkspaceError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

using SchedulerWorkspaceBuilder = std::function<SchedulerWorkspace(
    const SchedulerKey&)>;

class SchedulerWorkspaceCache {
public:
    const SchedulerWorkspace& require(
        const SchedulerKey& key,
        const SchedulerWorkspaceBuilder& builder
    );

    void invalidate_route(RouteGeneration new_generation);
    void invalidate_all();

    [[nodiscard]] std::size_t size() const noexcept { return workspaces_.size(); }
    [[nodiscard]] const SchedulerWorkspaceStats& stats() const noexcept {
        return stats_;
    }

private:
    std::map<SchedulerKey, SchedulerWorkspace> workspaces_;
    SchedulerWorkspaceStats stats_{};
};

}  // namespace canopy_foundry
