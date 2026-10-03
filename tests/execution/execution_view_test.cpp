#include "canopy_foundry/precision/execution_view.hpp"

#include <iostream>
#include <string>

namespace {

using namespace canopy_foundry;

bool require(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

ExecutionViewKey key() {
    return {
        7,
        ExecutionRole::Forward,
        NumericFormat::FP8E4M3,
        PackedLayout{128, 256, 16, 16, PackedOrientation::RowMajor},
    };
}

}  // namespace

int main() {
    if (!require(numeric_format_from_name("fp32") == NumericFormat::F32,
                 "fp32 alias was not normalized")) return 1;
    if (!require(numeric_format_name(NumericFormat::NVFP4E2M1UE4M3) ==
                     "nvfp4_e2m1_ue4m3",
                 "numeric format name changed")) return 1;
    if (!require(numeric_format_from_name("future_format") == NumericFormat::Unknown,
                 "unknown numeric format was promoted")) return 1;

    ExecutionViewCache cache;
    const auto view_key = key();
    int builder_calls = 0;
    const auto builder = [&builder_calls](
        const ExecutionViewKey& requested,
        TensorGeneration generation
    ) {
        ++builder_calls;
        return PackedExecutionView{
            requested,
            generation,
            static_cast<std::uint64_t>(builder_calls),
            4096,
            16,
            true,
        };
    };

    const auto& first = cache.require(view_key, TensorGeneration{1}, builder);
    if (!require(first.storage_token == 1, "first view was not built")) return 1;
    if (!require(cache.size() == 1, "first view was not cached")) return 1;
    if (!require(cache.lookup(view_key, TensorGeneration{1}) != nullptr,
                 "matching generation was not reusable")) return 1;

    const auto& reused = cache.require(view_key, TensorGeneration{1}, builder);
    if (!require(&reused == &first, "matching view was rebuilt instead of reused")) return 1;
    if (!require(cache.stats().reuses == 1, "reuse was not counted")) return 1;

    if (!require(cache.lookup(view_key, TensorGeneration{2}) == nullptr,
                 "stale generation was returned by lookup")) return 1;
    const auto& rebuilt = cache.require(view_key, TensorGeneration{2}, builder);
    if (!require(rebuilt.storage_token == 2, "stale view was not rebuilt")) return 1;
    if (!require(cache.stats().stale_rejections == 1,
                 "stale view rejection was not recorded")) return 1;

    bool unavailable_rejected = false;
    try {
        cache.invalidate(7, TensorGeneration{3});
        (void)cache.require(view_key, TensorGeneration{3}, {});
    } catch (const ExecutionViewError&) {
        unavailable_rejected = true;
    }
    if (!require(unavailable_rejected,
                 "missing rebuild path was not rejected")) return 1;
    if (!require(cache.stats().unavailable_rejections == 1,
                 "unavailable rebuild was not counted")) return 1;

    bool invalid_builder_rejected = false;
    try {
        (void)cache.require(view_key, TensorGeneration{4},
            [](const ExecutionViewKey& requested, TensorGeneration) {
                return PackedExecutionView{
                    requested,
                    TensorGeneration{99},
                    99,
                    1,
                    1,
                    true,
                };
            });
    } catch (const ExecutionViewError&) {
        invalid_builder_rejected = true;
    }
    if (!require(invalid_builder_rejected,
                 "builder generation mismatch was accepted")) return 1;
    if (!require(cache.lookup(view_key, TensorGeneration{4}) == nullptr,
                 "failed rebuild left a consumable view")) return 1;

    bool zero_generation_rejected = false;
    try {
        (void)cache.require(view_key, TensorGeneration{}, builder);
    } catch (const ExecutionViewError&) {
        zero_generation_rejected = true;
    }
    if (!require(zero_generation_rejected,
                 "zero generation was admitted")) return 1;

    cache.invalidate_all();
    if (!require(cache.empty(), "invalidate_all left cached views")) return 1;
    if (!require(cache.stats().invalidations == 1,
                 "invalidation count changed unexpectedly")) return 1;
    return 0;
}
