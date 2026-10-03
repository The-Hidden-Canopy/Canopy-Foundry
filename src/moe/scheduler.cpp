#include "canopy_foundry/moe/scheduler.hpp"

#include <utility>

namespace canopy_foundry {

std::string_view expert_dispatch_mode_name(ExpertDispatchMode mode) noexcept {
    switch (mode) {
    case ExpertDispatchMode::DevicePaddedPersistent:
        return "device_padded_persistent";
    case ExpertDispatchMode::DeviceCompact:
        return "device_compact";
    case ExpertDispatchMode::DiagnosticFixed:
        return "diagnostic_fixed";
    }
    return "unknown";
}

bool SchedulerWorkspace::valid_for(const SchedulerKey& expected) const noexcept {
    return schedule_valid && key == expected && expected.valid() &&
        workspace_token != 0 && schedule_token != 0 && tile_capacity > 0 &&
        expert_capacity >= static_cast<std::size_t>(expected.experts);
}

const SchedulerWorkspace& SchedulerWorkspaceCache::require(
    const SchedulerKey& key,
    const SchedulerWorkspaceBuilder& builder
) {
    if (!key.valid()) {
        throw SchedulerWorkspaceError("invalid scheduler workspace key");
    }

    const auto found = workspaces_.find(key);
    if (found != workspaces_.end()) {
        if (found->second.valid_for(key)) {
            ++stats_.reuses;
            return found->second;
        }
        ++stats_.stale_rejections;
        workspaces_.erase(found);
    }

    if (!builder) {
        ++stats_.unavailable_rejections;
        throw SchedulerWorkspaceError(
            "scheduler workspace is missing or stale and no rebuild is available");
    }

    SchedulerWorkspace rebuilt = builder(key);
    if (!rebuilt.valid_for(key)) {
        ++stats_.stale_rejections;
        throw SchedulerWorkspaceError(
            "scheduler workspace builder returned mismatched metadata");
    }

    const auto [inserted, ok] = workspaces_.emplace(key, std::move(rebuilt));
    if (!ok) {
        throw SchedulerWorkspaceError("scheduler workspace insertion failed");
    }
    ++stats_.builds;
    return inserted->second;
}

void SchedulerWorkspaceCache::invalidate_route(RouteGeneration new_generation) {
    if (!new_generation.valid()) {
        throw SchedulerWorkspaceError(
            "scheduler invalidation requires a non-zero route generation");
    }

    for (auto it = workspaces_.begin(); it != workspaces_.end();) {
        if (it->first.route_generation != new_generation) {
            it = workspaces_.erase(it);
            ++stats_.invalidations;
        } else {
            ++it;
        }
    }
}

void SchedulerWorkspaceCache::invalidate_all() {
    stats_.invalidations += workspaces_.size();
    workspaces_.clear();
}

}  // namespace canopy_foundry
