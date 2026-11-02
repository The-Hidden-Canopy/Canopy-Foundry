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

    // A mutation invalidates every representation for a tensor, and a stale
    // caller cannot reconstruct an older generation even after cache clear.
    ExecutionViewCache multi;
    auto backward_key = view_key;
    backward_key.role = ExecutionRole::BackwardDx;
    auto other_tensor = view_key;
    other_tensor.tensor = 8;
    (void)multi.require(view_key, TensorGeneration{1}, builder);
    (void)multi.require(backward_key, TensorGeneration{1}, builder);
    (void)multi.require(other_tensor, TensorGeneration{1}, builder);
    (void)multi.require(view_key, TensorGeneration{2}, builder);
    if (!require(multi.lookup(backward_key, TensorGeneration{1}) == nullptr && multi.size() == 2 &&
                     multi.lookup(other_tensor, TensorGeneration{1}) != nullptr,
                 "new generation left an old role reusable")) return 1;
    const int before_stale = builder_calls;
    bool old_generation_rejected = false;
    try { (void)multi.require(backward_key, TensorGeneration{1}, builder); }
    catch (const ExecutionViewError&) { old_generation_rejected = true; }
    if (!require(old_generation_rejected && builder_calls == before_stale,
                 "stale generation was rebuilt")) return 1;
    multi.invalidate(7, TensorGeneration{3});
    multi.invalidate_all();
    bool stale_after_clear = false;
    try { (void)multi.require(view_key, TensorGeneration{2}, builder); }
    catch (const ExecutionViewError&) { stale_after_clear = true; }
    if (!require(stale_after_clear, "cache clear forgot the generation floor")) return 1;
    bool rollback_rejected = false;
    try { multi.invalidate(7, TensorGeneration{2}); }
    catch (const ExecutionViewError&) { rollback_rejected = true; }
    if (!require(rollback_rejected, "invalidation rolled generation backwards")) return 1;
    (void)multi.require(view_key, TensorGeneration{3}, builder);
    if (!require(multi.lookup(view_key, TensorGeneration{3}) != nullptr,
                 "current generation was not admitted after invalidation")) return 1;
    multi.reset_for_new_model();
    (void)multi.require(view_key, TensorGeneration{1}, builder);
    if (!require(multi.lookup(view_key, TensorGeneration{1}) != nullptr,
                 "explicit new-model reset did not release the generation floor")) return 1;

    // A builder can call back into the cache. A newer generation published
    // inside that callback must fence the outer, now-stale rebuild.
    ExecutionViewCache reentrant;
    bool superseded_rejected = false;
    try {
        (void)reentrant.require(view_key, TensorGeneration{1},
            [&reentrant, &builder](const ExecutionViewKey& requested, TensorGeneration generation) {
                (void)reentrant.require(requested, TensorGeneration{2}, builder);
                return PackedExecutionView{requested, generation, 999, 4096, 16, true};
            });
    } catch (const ExecutionViewError&) { superseded_rejected = true; }
    if (!require(superseded_rejected && reentrant.lookup(view_key, TensorGeneration{1}) == nullptr &&
                     reentrant.lookup(view_key, TensorGeneration{2}) != nullptr,
                 "superseded reentrant build was published")) return 1;
    return 0;
}
