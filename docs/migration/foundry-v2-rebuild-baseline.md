# Canopy Foundry V2 rebuild baseline

Baseline date: 2026-09-29  
Repository: `The-Hidden-Canopy/Canopy-Foundry`  
Baseline revision: `53442be` (`Native Trainer Updates, Model Types - ETC`)  
Working tree at capture: clean

This is a migration baseline, not a capability or promotion claim. It records
what was observed before the planner foundation was added and keeps missing
hardware evidence explicit.

## Structural snapshot

| Area | Observed size | Boundary concern |
| --- | ---: | --- |
| `src/trainer.cu` | 702,656 bytes | Model state, precision, forward/backward, MoE, optimizer, memory, pipeline, checkpoint, telemetry, and loop orchestration remain co-located. |
| `include/ida_native/trainer.hpp` | 22,002 bytes | Trainer-facing types remain under the legacy `ida_native` namespace. |
| `scripts/neural_forge_worker.py` | 117,716 bytes | Worker still contains substantial manifest translation and execution-policy compatibility logic. |
| `api/native_api.py` | 8,868 bytes at baseline | In-process admission validator; the current port moves implementation to `foundry/admission/request_admission.py` and retains this path as a compatibility import. |

The first implementation batch does not rename or delete these surfaces. It
adds an independent planner contract and gates the CUDA entry point on plan
resolution before telemetry-directory creation and arena/model allocation.
The old trainer remains the execution implementation while extraction is
parity-tested.

## Test and evidence baseline

- Python suite: `119 passed, 2 skipped, 35 subtests passed` via `python -m pytest -q`.
- Public-boundary scan: `PASS` via `python scripts/check_public_boundary.py`.
- History-aware public-boundary scan: `PASS` via `python scripts/check_public_boundary.py --history`.
- CUDA compiler: unavailable on the capture host (`nvcc` was not found).
- CUDA smoke, target-hardware acceptance, binary hash, and representative
  checkpoint receipts: not captured on this host.

The planner foundation was compiled through a CPU-only CMake configuration.
The temporary CMake dependency fetch used the repository's existing pinned
`nlohmann_json` SHA-256 after this host's CA trust store prevented normal
certificate verification. The dependency remains in the ignored build
directory and was not added to the repository.

Native planner validation:

- `foundry_execution_plan_test`: passed.
- `ida_native_request_governance_test`: passed.
- CTest registered set: `2/2` passed.
- Known SHA-256 vector and deterministic plan-hash checks: passed.

## First-batch deliverables

The planner foundation now provides:

- `RuntimeCapabilities` with `AVAILABLE`, `UNAVAILABLE`, `EXPERIMENTAL`,
  `BUILD_DISABLED`, `HARDWARE_UNSUPPORTED`, and `RUNTIME_UNVERIFIED` states;
- a first-class `ExecutionPlan` containing device, model, precision, and
  optimizer projections;
- deterministic canonical plan JSON and `execution_plan_sha256`;
- `foundry plan REQUEST.json` and `foundry capabilities` targets;
- CUDA entry-point admission before telemetry and arena/model allocation,
  with the plan schema/hash carried into setup status;
- fail-closed rejection for unsupported precision profiles, invalid pipeline
  topology, unsupported attention backends, unavailable build targets, and
  the currently disabled CPU/OpenCL Lion paths;
- tests proving deterministic hashes, known-vector hashing, experimental
  precision rejection, duplicate-device rejection, and no false CPU support.

The plan command is non-allocating and does not launch a backend. CUDA
architecture and topology remain explicitly runtime-unverified until a
runtime probe is supplied.

## Deliberately not included in this baseline batch

- no V2 runtime dependency or build link;
- no CUDA kernel port;
- no NVFP4, MXFP8, MXFP2, or NVINT2 promotion;
- no MoE route/scheduler replacement;
- no trainer monolith deletion;
- no Hub HTTP/API changes;
- no worker-policy removal;
- no public capability claim based only on source or build presence.

## Next acceptance gate

The next bounded gate is to promote the parity fixtures into the native CUDA
build and begin extracting model allocation behind the already-resolved plan
without changing execution math. CUDA hardware evidence must be captured
separately before any architecture-specific capability moves beyond
`RUNTIME_UNVERIFIED`.
