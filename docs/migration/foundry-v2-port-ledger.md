# Canopy Foundry V2 port ledger

This ledger records bounded execution mechanisms adapted from a private donor
system into the standalone Foundry execution core. The donor remains a
read-only proving ground for this migration; Foundry does not import, link, or
require it. Donor paths, source listings, private recipes, and deployment
details intentionally stay out of this public ledger.

## Ported in the first slice

The first slice ports the generation rule behind V2's packed/execution and
activation caches into a backend-neutral contract:

- `include/canopy_foundry/precision/execution_view.hpp`
- `src/precision/execution_view.cpp`
- `tests/execution/execution_view_test.cpp`

Each cached view is keyed by tensor, execution role, numeric format, and packed
layout. It carries the source tensor generation and cannot be returned when that
generation is stale. A backend builder must return matching metadata before a
view becomes consumable. The cache stores only an opaque storage identity; a
backend continues to own device allocation and synchronization.

The planner now reports explicit master, forward, backward-dX, gradient,
accumulator, optimizer-state, and normalization format identities while keeping
legacy profile strings for compatibility. Current Foundry FP8 profiles resolve
to BF16 master storage plus an FP8 E4M3 representation. This describes the
existing storage contract; for the current Ampere adapter the actual GEMM
operands remain BF16 after dequant staging. It does not promote native FP8
Tensor Core or NVFP4 execution.

The planner revision is advanced to `planner-foundation-v2-port-v1` because
these explicit fields participate in the canonical plan hash.

## Donor evidence and adaptation

The donor execution contract established that forward owns the route schedule
and backward consumes the forward-owned buffers. The donor also uses
generation tags for activation/cache admission and route-plan reuse.

Foundry intentionally adapts the invariant rather than copying donor structs:
V2 device pointers, strict-family expert caches, private telemetry, and
deployment-specific allocation state do not cross the repository boundary.

## Explicitly not promoted

- No V2 source path or build dependency was added.
- No NVFP4 hardware capability was added to the Foundry planner.
- No Blackwell-only kernel was copied into the public tree.
- No strict Omni/family-specific cache was exposed as a generic public model.
- No new execution feature was added to `src/trainer.cu`.

## Ported in the second slice

The second slice adds a generic forward-owned `RoutePlan` and `RoutePlanStore`:

- `include/canopy_foundry/moe/route_plan.hpp`
- `src/moe/route_plan.cpp`
- `tests/execution/route_plan_test.cpp`

The store requires a monotonically advancing route generation, refuses duplicate
microstep/layer publication, and makes backward provide the exact generation
published by forward. Missing plans and mismatches are explicit errors with
bounded counters. The route plan carries opaque storage tokens and shape/
schedule identity only; it does not copy V2 device pointers or router-private
surfaces.

The execution planner now marks MoE requests as `forward_owned` with a required
route generation while leaving scheduler selection unresolved. This is an
admission statement, not a claim that a persistent or compact CUDA scheduler is
already present.

## Ported in the third slice

The third slice adds a route-generation-keyed scheduler workspace contract:

- `include/canopy_foundry/moe/scheduler.hpp`
- `src/moe/scheduler.cpp`
- `tests/execution/scheduler_workspace_test.cpp`

It distinguishes padded-persistent, compact, and diagnostic-fixed modes, but
does not select a production default. A workspace can be reused only for the
same route generation, shape, and orientation key. Changed route generations,
changed orientations, invalid builder metadata, and unavailable rebuilds are
all explicit and counted.

## Next bounded port

The next compatible slice is a backend-neutral forward tape that binds
activation policy and route-plan identity before backward. No donor-specific
selected-row kernel should be considered until that ownership boundary is in
place.

## Ported in the fourth slice

The fourth slice adds `ForwardTape` and backward validation:

- `include/canopy_foundry/train/forward_tape.hpp`
- `src/train/forward_tape.cpp`
- `tests/execution/forward_tape_test.cpp`

The tape seals model generation, microstep, activation policy, layer order, and
route generation. Backward validation rejects an unsealed tape, mismatched
model/microstep identity, or a missing/mismatched forward-owned route plan.
This is still a contract and admission layer; activation allocation and CUDA
copy/recompute implementations remain future backend work.

## Ported in the fifth slice

The fifth slice makes the plan boundary carry the execution choices that were
previously implicit or represented only as legacy strings:

- `include/canopy_foundry/memory/allocation_plan.hpp`
- `include/canopy_foundry/parallel/plan.hpp`
- `include/canopy_foundry/checkpoint/generation.hpp`
- `include/canopy_foundry/execution/receipt.hpp`

Memory planning now records persistent, per-microstep, and checkpoint-peak
estimates, explicit residency, activation policy, and whether runtime capacity
has actually been verified. Shape arithmetic uses checked operations and
overflow rejects the plan. A shape estimate is never reported as device
capacity evidence.

Parallel planning now distinguishes single-device, layer-pipeline, and future
expert-sharded modes. Transport is explicit as none, unresolved, CUDA peer, or
host-staged; automatic transport remains unresolved until a runtime topology
probe. Existing compatibility strings remain in the device section while the
typed parallel section is the forward contract.

Checkpoint planning now separates model export from exact resume. Generation
validation requires the execution-plan and model hashes for export and adds
optimizer/data-cursor hashes for exact resume. Publication is marked atomic,
but exact resume is not marked supported until runtime validation proves a
complete generation.

The bounded execution receipt keeps requested, resolved, accepted, observed,
and persisted states separate and carries only aggregate counters. Evidence
tier is explicit, so source/build presence cannot become hardware, burn, or
promotion evidence by serialization alone.

The fifth slice does not implement a CUDA allocator, a topology probe, a
checkpoint filesystem publisher, a two-device executor, or a private adapter.
Those remain backend/runtime responsibilities and are intentionally not
claimed by these contracts.

## Ported in the sixth slice

The sixth slice adds an optimizer ownership contract:

- `include/canopy_foundry/optimizer/plan.hpp`
- `src/optimizer/plan.cpp`
- `tests/execution/optimizer_plan_test.cpp`

Optimizer groups now carry an owner class and explicit tensor ranges. Cross-
group overlap is rejected before any overlap schedule can be admitted. The
execution plan reports the owner, whether overlap is enabled, and a bounded
group count; the current planner keeps overlap disabled and uses device
ownership for compatibility. No optimizer math, host thread, or CUDA event
schedule was promoted by this slice.
