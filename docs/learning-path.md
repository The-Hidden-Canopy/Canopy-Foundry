# Canopy Foundry Learning Path

Canopy Foundry is a local/native runtime repository. It contains the public
request contract, native execution code, portability paths, and bounded local
receipts. It is not the organization-authority layer, an HTTP service, a
promotion system, or proof of training quality on a particular machine.

This path is for someone who can read Python and C++ but is still building a
mental model of the repository. It is deliberately safe to follow without
credentials, private datasets, a remote service, or a production GPU.

## Read in this order

1. [README](../README.md) — the product boundary, supported public surface,
   and current limitations.
2. [Starter guide](../STARTER_GUIDE.md) — how to build and invoke the native
   one-shot executable.
3. [Neural Foundry guide](neural-foundry-guide.md) — the operational contract,
   input rules, and expected failures.
4. [Request-admission README](../api/README.md) — why `api/` is an in-process
   validator and not a server.
5. [Python wrapper contract](python-wrapper-contract.md) — the boundary between
   the governed external wrapper and native execution.
6. [Public/local boundary](public-boundary.md) — what may be public and what
   must remain deployment-owned.
7. [Private runtime boundary](private-runtime-boundary.md) — why deployment
   bindings and private evidence are not part of the public tree.

Do not skip the boundary documents. They explain why some apparently missing
features are absent by design.

## First 20 minutes

These commands inspect the repository and run local checks. They do not log in,
upload data, or start a service.

```sh
git status --short
python scripts/check_public_boundary.py
python -m json.tool configs/public/capabilities.json
python -m pytest -q tests/test_native_api.py tests/test_private_boundaries.py
```

When reading `configs/public/capabilities.json`, distinguish a declared
capability from an enabled deployment. The public catalog currently includes
CUDA-local, CPU-smoke, and OpenCL-smoke profiles; the smoke profiles are
disabled and experimental. A catalog entry is not hardware evidence.

The focused tests are a good first map of the intended behavior. In particular,
look for tests that reject absolute paths, authority fields, stale or mismatched
artifacts, receipt reuse, and private runtime material. A rejection is often a
successful boundary test, not a missing feature.

## The request flow

Read one request from outside in. The important distinction is between a
request being well-formed, a request being authorized, and a run being
observed as successful.

```text
untrusted request body
        |
        v
in-process admission and path/schema validation  (api/native_api.py)
        |
        v
governed wrapper supplies org/role/justification,
state transitions, audit, limits, and process supervision
        |
        v
explicit local native request  (src/request.cpp)
        |
        v
one-shot native executable  (src/main.cpp / trainer backends)
        |
        v
bounded status and local receipt
```

The native layer does not acquire organization authority. It must not be taught
to accept role, transition, audit, promotion, socket, queue, or worker control
fields as a shortcut around the governed wrapper. The wrapper's acceptance of
a request and the native process's observed result are separate facts.

## Pick a trail

| If you want to understand... | Start with... | Then compare... |
| --- | --- | --- |
| Request shape and rejection rules | `api/native_api.py` | `tests/test_native_api.py` and `include/ida_native/request.hpp` |
| Worker-to-native handoff | `scripts/neural_forge_worker.py` | `scripts/local_binding.py`, `scripts/local_receipt.py`, and `tests/test_neural_forge_worker.py` |
| Native JSON parsing | `src/request.cpp` | `src/main.cpp` and `include/ida_native/request.hpp` |
| Model and artifact compatibility | `scripts/model_contracts.py` | `scripts/compatibility.py` and `tests/test_model_portability.py` |
| Receipt integrity and replay behavior | `scripts/local_receipt.py` | `tests/test_private_boundaries.py` |
| CUDA/CPU/OpenCL portability | `CMakeLists.txt` | `STARTER_GUIDE.md`, `configs/examples/`, and `PORTABILITY.md` |
| What cannot be published | `docs/public-boundary.md` | `scripts/check_public_boundary.py` and `tests/test_public_boundary.py` |

For each trail, ask the same four questions:

1. What input is accepted?
2. What is intentionally rejected?
3. Which layer owns the decision?
4. What evidence would prove the behavior on a real machine?

The fourth question matters. Reading a code path can establish an intended
contract; it cannot establish GPU performance, model quality, production
readiness, or a live deployment state.

## Safe exercises

### Exercise 1: predict admission outcomes

Open `tests/test_native_api.py` and identify the smallest valid request. Before
running the tests, predict what should happen when you:

- replace a relative dataset filename with an absolute path;
- add a `role` or other authority field to the native request;
- request an unsupported runtime;
- use a device ordinal outside the allowed range.

Run the focused test file and compare the result with your predictions. Notice
that the validator rejects unsafe shapes without echoing machine-specific paths
back to the caller.

### Exercise 2: separate catalog, example, and evidence

Compare `configs/public/capabilities.json` with the example configurations in
`configs/examples/`. Mark each value as one of:

- a public contract or policy declaration;
- an example input;
- an observed result that still needs to be produced by a run.

Do not infer that an example configuration ran successfully, or that a declared
profile is enabled on your hardware.

### Exercise 3: follow a receipt boundary

Read `scripts/local_receipt.py` alongside the receipt tests. Find the fields
that are retained for replay/integrity checks and the fields deliberately
omitted from public evidence, such as raw commands, paths, or private native
execution details. Then explain why receipt reuse and manifest changes must not
silently collapse into one run.

### Exercise 4: trace one native field

Choose one ordinary field such as `optimizer_type`, `output_dir`, or
`architecture_contract`. Follow it from `include/ida_native/request.hpp` into
`src/request.cpp` and then to the execution code. Separately list fields that
must stop at the governed wrapper. This is a useful way to learn where a local
runtime contract ends without expanding native authority.

When changing behavior, write the failing boundary test first. Keep the happy
path test, the rejecting-path test, and the evidence/receipt assertion together
so a future change cannot make the system appear simpler by removing its
controls.

## Glossary

- **Admission**: validation that a request has an allowed shape and bounded
  values. Admission is not organization authorization.
- **Capability catalog**: a public declaration of profile/backend contracts. It
  does not claim that a profile is enabled or that a run succeeded.
- **Binding**: deployment-owned mapping from opaque identifiers to local assets
  and executable inputs. Bindings are not a way to publish local paths.
- **Native request**: the explicit local JSON contract consumed by the native
  executable.
- **Receipt**: bounded local evidence about a manifest/run outcome, designed for
  integrity and replay checks without exposing private execution detail.
- **Fail-closed**: reject or mark a result unavailable when a required fact is
  missing, stale, mismatched, or untrusted; do not fill the gap with a guess.
- **Observed result**: what a process actually reported or what a validator
  actually checked. It is different from an authorized decision or a promotion
  decision.
- **Public/private boundary**: the line between source and contracts safe to
  distribute and deployment-owned data, credentials, telemetry, or authority.

## Troubleshooting without guessing

| Symptom | Read next | Boundary to preserve |
| --- | --- | --- |
| CMake cannot configure | [Starter guide](../STARTER_GUIDE.md) and `CMakePresets.json` | A successful configure is not a training result. |
| No CUDA device is available | `PORTABILITY.md` and the CPU/OpenCL examples | Smoke builds are experimental and not production-equivalent. |
| A request rejects a path | `api/native_api.py` and `tests/test_native_api.py` | Use bounded local references; do not relax path checks to make a fixture pass. |
| An HTTP endpoint is missing | [Request-admission README](../api/README.md) | `api/` is not a server and must not become an authority bypass. |
| A private runtime or binding is absent | [Private runtime boundary](private-runtime-boundary.md) | Deployment-owned material is intentionally unavailable in the public repo. |
| A receipt is unavailable or mismatched | `scripts/local_receipt.py` and `tests/test_private_boundaries.py` | Treat missing or stale evidence as unavailable, not as success. |

## What this path does not prove

After completing these exercises, you should be able to navigate the source,
explain the request and evidence boundaries, run the public checks, and propose
a focused test. You still cannot claim that the repository proves:

- a specific GPU's throughput, memory headroom, or numerical quality;
- a model's usefulness or generalization;
- a production deployment, organization authorization, audit completion, or
  promotion approval;
- the availability of private datasets, checkpoints, kernels, bindings, or
  telemetry;
- that a local smoke build is equivalent to the enabled CUDA training path.

Those claims require separately identified execution and promotion evidence.
