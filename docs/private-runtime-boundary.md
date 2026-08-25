# Private advanced runtime boundary

Neural Foundry publishes the local capability contract, model-contract IDs,
request admission, and bounded worker behavior. Advanced kernel
implementations and their mathematical details are deployment-owned.

The Hub may retain a private runtime package behind its private artifact
storage, but the worker contract carries only:

- an opaque package ID;
- a SHA-256 package digest; and
- the opaque tier `private_advanced`.

There is no public package URL, source reference, download route, source
listing, kernel name, formula, raw trace, or private telemetry field. A local
deployment map must contain the same opaque package reference before a worker
can use it. A different package ID or digest fails closed. The Hub coordinates
the reference; the local worker resolves the actual private package through
ignored machine-local bindings.

## Model contracts

Model contract IDs are compatibility claims with an evidence tier, not model
downloads or quality guarantees:

- `hf_gpt2_native_v1` — observed GPT-2-compatible layout;
- `ida_lattice_native_v1` — observed dense SmolLM2-shaped and Qwen2.5-shaped
  portability surface;
- `generic_moe_native_v1` — experimental Mixtral-shaped portability surface.

The generic MoE contract does not claim Qwen-MoE always-on shared-expert
support. A local model or checkpoint binding may include its contract ID and
digest, but model bytes remain on the machine running the trainer.

## Retry and idempotency

The Hub uses an opaque worker operation ID and compare-and-swap receipt. The
local worker additionally writes an ignored, path-free receipt under the local
run root. Replaying the same run and manifest does not launch a second native
process; changing the manifest under an existing run ID is rejected. Receipts
contain only an opaque run ID, a manifest fingerprint, terminal state, and a
bounded metrics flag.

## What is intentionally not public

Do not copy private kernel source, private binaries, deployment maps, package
URLs, private observability headers, profiler output, model weights, datasets,
checkpoints, or credentials into this repository. The public boundary checker
and ignored local-runtime paths are release gates, not storage for private
implementation details.
