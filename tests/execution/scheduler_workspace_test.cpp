#include "canopy_foundry/moe/scheduler.hpp"

#include <iostream>

namespace {

using namespace canopy_foundry;

bool require(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

SchedulerKey key(RouteGeneration generation, int orientation = 0) {
    return {generation, 4, 64, 8, 2, orientation};
}

}  // namespace

int main() {
    if (!require(expert_dispatch_mode_name(ExpertDispatchMode::DeviceCompact) ==
                     "device_compact",
                 "dispatch mode name changed")) return 1;

    SchedulerWorkspaceCache cache;
    int builder_calls = 0;
    const auto builder = [&builder_calls](const SchedulerKey& requested) {
        ++builder_calls;
        return SchedulerWorkspace{
            requested,
            static_cast<std::uint64_t>(builder_calls),
            static_cast<std::uint64_t>(builder_calls + 10),
            128,
            static_cast<std::size_t>(requested.experts),
            true,
        };
    };

    const auto first_key = key({1});
    const auto& first = cache.require(first_key, builder);
    if (!require(first.workspace_token == 1, "workspace was not built")) return 1;
    const auto& reused = cache.require(first_key, builder);
    if (!require(&reused == &first, "matching workspace was not reused")) return 1;
    if (!require(cache.stats().reuses == 1, "workspace reuse was not counted")) return 1;

    const auto& next_route = cache.require(key({2}), builder);
    if (!require(next_route.workspace_token == 2,
                 "route generation did not rebuild workspace")) return 1;
    const auto& next_orientation = cache.require(key({2}, 1), builder);
    if (!require(next_orientation.workspace_token == 3,
                 "orientation did not rebuild workspace")) return 1;

    cache.invalidate_route({2});
    if (!require(cache.size() == 2,
                 "route invalidation removed matching generation")) return 1;
    if (!require(cache.stats().invalidations == 1,
                 "route invalidation was not counted")) return 1;

    bool unavailable_rejected = false;
    try {
        cache.invalidate_all();
        (void)cache.require(key({3}), {});
    } catch (const SchedulerWorkspaceError&) {
        unavailable_rejected = true;
    }
    if (!require(unavailable_rejected,
                 "missing scheduler rebuild was not rejected")) return 1;

    bool invalid_builder_rejected = false;
    try {
        (void)cache.require(key({4}), [](const SchedulerKey& requested) {
            return SchedulerWorkspace{
                requested,
                1,
                2,
                0,
                static_cast<std::size_t>(requested.experts),
                true,
            };
        });
    } catch (const SchedulerWorkspaceError&) {
        invalid_builder_rejected = true;
    }
    if (!require(invalid_builder_rejected,
                 "invalid scheduler workspace was accepted")) return 1;
    if (!require(cache.stats().stale_rejections == 1,
                 "invalid scheduler workspace evidence changed unexpectedly")) return 1;
    return 0;
}
