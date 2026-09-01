# Neural Foundry Secure Operations Guide

Neural Foundry is the native training runtime. It provides a C++/CUDA full
trainer, explicit OpenCL/CPU smoke trainers, and a narrow in-process request
admission helper.

It is not an authority or governance plane. Authentication, organization
scope, role tier, justification, state transitions, audit/domain events,
approval, process supervision, resource limits, and telemetry publication
belong to the governed external wrapper.

The execution contract has three explicit backend identifiers: full CUDA, AMD
OpenCL smoke, and native CPU smoke. Only CUDA is currently enabled for
training. The Hub may queue a bounded native job, and the
local Neural Forge worker—the local Neural Foundry implementation component—
may translate that job into a native request manifest. CUDA/PyTorch method
calls, Python trainer selection, and arbitrary runtime options are not API
capabilities. Portable smoke backends are not silent fallbacks for CUDA
features.

The worker filename, binding names, and `NEURAL_FORGE_*` environment variables
are retained compatibility identifiers. They do not describe a separate
public product.

Some native examples also retain legacy `ida_native` executable, namespace,
include-path, and module names for compatibility with existing wrappers. Read
those as implementation identifiers; the public product identity is Neural
Foundry.

The public capability catalog is [`configs/public/capabilities.json`](../configs/public/capabilities.json).
It describes neutral backend capability names only; it contains no model
architecture, hyperparameters, local paths, dataset IDs, model IDs,
credentials, or Hub authority. `cuda-local` is a capability label, not a
personal training profile: users supply their own local config, dataset,
model, and checkpoint bindings. OpenCL and CPU profiles are disabled because their implemented
smoke update path is AdamW. `--allow-experimental` does not override a
disabled profile. Adam and AdamW are rejected at the Hub, worker, adapter, and
native request boundaries; enabled CUDA profiles accept Lion only.

## Boundary at a glance

```text
governed wrapper
    |
    | policy, identity, org scope, approval, audit, state transition
    v
api.native_api.NativeApiService.validate()
    |
    | native request only; no authority fields; paths checked under roots
    v
fresh request.json + controlled output directory
    |
    v
selected native target --request-json request.json [--device N]
    |
    | observed JSONL stdout/stderr and exit status
    v
governed wrapper records the result and performs any promotion transition
```

The admission helper does not bind a socket, accept bearer tokens, launch a
child process, or publish telemetry. There is no Neural Foundry HTTP endpoint
to configure, expose, or protect.

The Hub is the only remote control-plane surface. Its worker claim response
contains two separate objects: a public run projection and an approved worker
manifest. The manifest contains opaque IDs and policy-selected execution
attestation, never local paths or private code. The worker resolves those IDs
through a deployment-owned binding file at
`configs/local/neural-forge-binding.json`; that file is ignored by Git and
must be provisioned locally. The external profile and resource IDs are
translated through the separate ignored
`configs/local/neural-forge-deployment-map.json`; the public capability
catalog is never expanded with deployment or model identities.

For a live local Hub contract check, provision the Hub's private policy and
token environment, then run its
`scripts/neural-forge-local-integration.ps1` harness. That harness uses
Azurite and Azure Functions Core Tools and drives synthetic bounded events;
the native worker and the actual GPU smoke remain separate checks.

## 1. Build the native runtime

From the repository root:

```sh
cmake -S . -B build -DIDA_NATIVE_CUDA_ARCHITECTURES=90a
cmake --build build --config Release --target ida_native_train
```

Use the CUDA architecture matching the installed GPU and toolkit. The full
binary is written to `bin/ida_native_train`; portable targets are
`bin/ida_native_opencl_train` and `bin/ida_native_cpu_train`.

The build does not download dependencies. Provide CMake 3.28+, a C++20
compiler, and a local `nlohmann_json` 3.11.x installation. CUDA and OpenCL
development files are required only for their respective targets.

## 2. Keep inputs outside Git

Native datasets and checkpoints should live outside the repository. The
repository ignores datasets, artifacts, run output, credentials, certificates,
tokens, and common model formats.

Before staging a change, check both the worktree and the index:

```sh
git status --short
git diff --check
git ls-files | grep -Ei '(^|/)(\.env|secrets|.*token.*|.*private.*key.*|.*credential.*)'
```

The last command should return no real credential files. Example templates and
documentation are acceptable; actual values are not. Never put bearer tokens,
cloud credentials, private keys, or organization metadata in a native request
or command-line argument.

## 3. Prepare native data

The trainer consumes complete little-endian blocks:

- `tokens.u32`: contiguous `uint32` token IDs;
- `labels.i32`: contiguous `int32` labels with the same byte size;
- `segs.u16`: optional `uint16` segment offsets, with one value per token.

The runtime rejects empty, mismatched, partial, non-regular, or unavailable
blocks. If `segs.u16` is supplied, it must be exact; the runtime does not
silently fall back to another attention mode.

Use a fresh output directory for every run. An existing `metrics.json` is not
treated as a successful result. Resume requires a complete checkpoint with
both `model.safetensors` and `optimizer_state.safetensors`.

## 4. Validate an untrusted request

The admission payload is intentionally separate from the native CLI request.
It contains a device and a native request, but no output ownership:

```json
{
  "device": 0,
  "request": {
    "backend": "native",
    "precision_profile": "legacy_bf16",
    "optimizer_type": "lion",
    "model": {
      "hidden_size": 128,
      "intermediate_size": 512,
      "layers": 2,
      "heads": 4,
      "vocab_size": 4096
    },
    "training": {
      "microbatch": 1,
      "grad_accumulation": 1,
      "learning_rate": 0.0001,
      "max_steps": 1
    },
    "input": {
      "token_blocks": "sample/tokens.u32",
      "label_blocks": "sample/labels.i32",
      "batch_size": 1,
      "sequence_length": 2048
    }
  }
}
```

Dataset and artifact references are relative to roots configured by the
wrapper. Absolute paths, traversal components, missing files, and authority
fields are rejected. The wrapper assigns `output_dir` after validation; an
untrusted request cannot choose where results are written.

In-process validation is the supported Python usage:

```python
import os
from pathlib import Path

from api.native_api import ApiConfig, NativeApiService

service = NativeApiService(
    ApiConfig(
        dataset_root=Path(os.environ["NEURAL_FORGE_ROOT"]) / "datasets",
        artifact_root=Path(os.environ["NEURAL_FORGE_ROOT"]) / "artifacts",
        run_root=Path(os.environ["NEURAL_FORGE_ROOT"]) / "runs",
        binary=Path(os.environ["NEURAL_FORGE_ROOT"]) / "bin" / "ida_native_train",
    )
)

# payload is the untrusted JSON object shown above.
result = service.validate(payload)
assert result == {"valid": True}
```

Validation does not start training and does not create a run directory.

## 5. Execute only through the governed wrapper

After policy approval and successful validation, the wrapper creates a fresh
native request with a controlled output directory and invokes exactly one
one-shot process:

```sh
bin/ida_native_train \
  --request-json "$NEURAL_FORGE_ROOT/runs/run-2026-08-22/request.json" \
  --device 0
```

The wrapper must:

1. use `execute_transition()` for governed state changes;
2. emit the approved domain event through `append_domain_event(...)`;
3. enforce permission constants, organization scope, role tier, and
   justification before execution;
4. create a fresh request/output pair;
5. capture stdout, stderr, exit code, and produced files;
6. reject a run if the process fails or the completion object/metrics are
   missing;
7. preserve the failure rather than using stale, cached, or fabricated data.

The native runtime must never receive organization, role, justification,
transition, audit, domain-event, promotion, queue, worker, telemetry,
ontology, evidence, socket, or server fields. Those fields are rejected at
any nesting level.

## 6. Interpret runtime output safely

Stdout is newline-delimited JSON:

- `{"type":"step", ...}` reports observed optimizer progress;
- `{"type":"complete","status":"complete", ...}` marks successful
  completion.

Stderr is diagnostic only. It is useful for troubleshooting but cannot approve
a transition or turn a failed process into a successful run.

The wrapper should only publish metrics after all of the following are true:

- the child exits with code `0`;
- the canonical terminal status phase is present and valid;
- the status/metrics artifacts exist in the new output directory;
- the checkpoint files satisfy the expected contract;
- the governed state transition and audit/domain event succeed.

Native completion does not itself authorize a successful Hub run. The worker
publishes the observed process exit as a non-terminal `running` update, polls
the worker status route for evaluation-gate completion and controller
cancellation, and only then submits `succeeded` or `cancelled`. If required
evaluation gates do not complete before the manifest deadline, the worker
submits `failed`. The local receipt is marked terminal only after the Hub
accepts the corresponding governed update.

## 7. Expected failures

| Symptom | Meaning | Correct response |
| --- | --- | --- |
| Connection attempt refused | No Neural Foundry HTTP service exists | Use the governed wrapper and in-process validation |
| `authority fields are not accepted` | Governance data crossed the native boundary | Keep policy data in the wrapper |
| `request path is invalid` | Absolute or traversal path supplied | Resolve a relative reference under a configured root |
| `output ownership is wrapper-controlled` | Caller tried to choose output location | Let the wrapper allocate a fresh run directory |
| Existing `metrics.json` rejected | Output may be stale or reused | Choose a new output directory or provide a complete resume checkpoint |
| Non-zero native exit code | Training did not complete | Preserve the failure; do not invent metrics |

## 8. Verification checklist

Run the Python boundary tests:

```sh
python -m unittest discover -s tests -p "test_*.py"
```

Before release, confirm:

- no HTTP server, listener, bearer-token endpoint, or child-process launcher
  exists in `api/`;
- no route or port is documented as callable;
- no secrets, datasets, checkpoints, or generated logs are tracked;
- path traversal and cross-root references are rejected;
- authority fields are rejected at nested levels;
- stale output and incomplete checkpoints fail closed;
- the governed wrapper owns identity, scope, transitions, audit, and
  promotion.
- the worker selects only an attested native target and passes `--device` only
  to CUDA/OpenCL targets, never to the CPU target;
- the native binary is built and its request-contract test passes;
- no generated `native-request.json`, `worker-config.json`, binary, dataset,
  checkpoint, or credential is staged;
- `git diff --cached --name-only` contains only the intended publish files;
- the final publish review is complete before enabling the Hub feature flag.

## 9. Run the local Neural Foundry worker

The local implementation is called the Neural Forge worker in compatibility
routes and filenames. The Hub's worker routes do not receive native paths or
execute the trainer. Run this worker on the machine that owns the sensitive
code, data, checkpoints, and binary:

```sh
python scripts/neural_forge_worker.py \
  --hub-url https://hidden-canopy-hub-api.azurewebsites.net/ \
  --hub-allowed-host hidden-canopy-hub-api.azurewebsites.net \
  --worker-token "$NEURAL_FORGE_WORKER_TOKEN" \
  --org "$NEURAL_FORGE_ORG" \
  --config-root "$NEURAL_FORGE_ROOT/configs" \
  --dataset-root "$NEURAL_FORGE_ROOT/datasets" \
  --artifact-root "$NEURAL_FORGE_ROOT/artifacts" \
  --kernel-root "$NEURAL_FORGE_ROOT/kernels" \
  --run-root "$NEURAL_FORGE_ROOT/runs" \
  --catalog "$NEURAL_FORGE_ROOT/configs/local/neural-forge-capabilities.json" \
  --deployment-map "$NEURAL_FORGE_ROOT/configs/local/neural-forge-deployment-map.json" \
  --hardware-probe-root "$NEURAL_FORGE_ROOT/vendor/hardware-tools" \
  --binary "$NEURAL_FORGE_ROOT/bin/ida_native_train" \
  --function-key "$NEURAL_FORGE_FUNCTION_KEY" \
  --gpu-fingerprint-key "$NEURAL_FORGE_GPU_FINGERPRINT_KEY" \
  --leaderboard-attestation-key "$NEURAL_FORGE_LEADERBOARD_ATTESTATION_KEY" \
  --worker-id worker-01
```

Leaderboard publication is optional and requires both the local GPU
fingerprint key and a separate deployment attestation key. The latter must
match the Hub's `NEURAL_FORGE_LEADERBOARD_ATTESTATION_KEY`; it signs only the
run ID and already-sanitized vendor/model/fingerprint envelope. Missing or
forged signatures make the run ineligible for publication.

The worker accepts only Hub-issued opaque references. It translates each
external training profile to a distinct local profile through the deployment
map, resolves dataset/model/checkpoint IDs through the local binding, verifies
the binding's catalog version and SHA-256 catalog pin, creates a local run
directory, creates an ignored native request manifest, verifies the mapped
v3 backend/binary/trainer attestation, and invokes the selected target. The
resource-class entry in the deployment map selects the local device; the
manifest cannot select a GPU. Optimizer, precision, backend, paths, and
training settings come only from the local profile/configuration. V3 OpenCL
worker runs additionally resolve a private `kernel_ref` under `--kernel-root`
and verify it against the catalog's pinned SHA-256 digest. Legacy direct
    worker runs additionally require the Hub-issued timestamp, claim window,
    authority, and lineage envelopes. Those envelopes are checked for opaque IDs and recursively
reject local paths, commands, credentials, and private diagnostics before any
local binding is resolved. Legacy direct-request jobs and manifests containing
local execution fields are rejected.

Local bindings are schema v2 and require a SHA-256 pin for every config,
dataset, model, checkpoint, and native binary. An optional `hardware_probes`
collection can pin deployment-owned `nvidia-smi`/`rocm-smi` tools under the
ignored probe root. The binary must match both the Hub attestation and the
deployment-owned `binaries` entry. The worker passes only neutral system/temp
variables to native children and probes; `PATH`, loader variables, credentials,
and caller settings are not inherited.

The worker sends only bounded progress/status records back to the Hub. It
never uploads the native executable, local filesystem paths, dataset bytes,
checkpoints, or raw stderr. Hub redirects are rejected, responses are bounded,
and the child process runs in a dedicated process group so timeout cleanup
does not leave descendants running.

Each worker status mutation carries an opaque `operation_id`. The Hub records
that receipt together with the job and event mutation using compare-and-swap.
The worker may safely retry the exact same operation; the Hub returns the
original projection without duplicating the event. Reusing an operation ID
with a different body is rejected. The worker must poll for the controller's
`cancel_requested` state and may then finalize `cancelled`; it cannot bypass
the controller by submitting `cancelled` itself.

For an explicitly opted-in leaderboard run, the Hub claim projection includes
only `leaderboard_requested`. After a successful run, the worker may query the
selected local device with fixed arguments to an absolute, hash-pinned
`nvidia-smi` or `rocm-smi` tool from `hardware_probes`. The worker discards the
raw UUID/serial, HMACs it with the deployment-only
`NEURAL_FORGE_GPU_FINGERPRINT_KEY`, and sends only `vendor`, sanitized `model`,
and `fingerprint` in the final update. Missing drivers or a missing key make
the run ineligible for the optional leaderboard without failing training.

The local binding has this shape and must be created on the worker machine;
the references are relative to the roots passed above:

```json
{
  "schema_version": "neural-foundry-local-binding.v2",
  "catalog_version": "catalog-2026-08-23-public-blank",
  "catalog_sha256": "<sha256 of configs/public/capabilities.json>",
  "profiles": {
    "cuda-local": {
      "ref": "profiles/my-cuda-config.json",
      "sha256": "<sha256 of profiles/my-cuda-config.json>",
      "precision": "legacy_bf16",
      "optimizer": "lion"
    },
    "opencl-smoke": {
      "ref": "profiles/opencl-smoke.json",
      "sha256": "<sha256 of profiles/opencl-smoke.json>",
      "precision": "fp32",
      "optimizer": "lion",
      "kernel_ref": "opencl_smoke.cl"
    }
  },
  "datasets": { "dataset-001": { "ref": "dataset-001", "sha256": "<sha256 of the dataset tree>" } },
  "models": {},
  "checkpoints": {},
  "binaries": {
    "cuda": { "ref": "ida_native_train", "sha256": "<sha256 of the binary>" },
    "opencl": { "ref": "ida_native_opencl_train", "sha256": "<sha256 of the binary>" },
    "cpu": { "ref": "ida_native_cpu_train", "sha256": "<sha256 of the binary>" }
  },
  "private_runtimes": {
    "nf-private-kernel-v1": {
      "ref": "nf-private-kernel-v1.bundle",
      "sha256": "<sha256 of the private package>",
      "binary_sha256": "<sha256 of the approved binary>"
    }
  }
}
```

`private_runtimes` is optional and is used only when the Hub manifest carries
an approved private-runtime reference. The package file must live below the
ignored `NEURAL_FORGE_PRIVATE_RUNTIME_ROOT`; package and binary hashes are
checked before the native process is launched.

### Deployment-only profile mapping

The public model deployment uses a separate local capability catalog supplied
with `--catalog`. It may contain one local profile for each external workload
(`public-edge-full`, `public-ai`, `public-edge-swift`, and `public-moe`) while
the tracked `configs/public/capabilities.json` remains neutral. Every local
profile must also have a matching entry in the schema-v2 binding with a
SHA-256-pinned config and the local binary.

The four public-model workloads are intentionally mapped one-to-one. The map
must contain a separate local profile ID and execution allowlist for each
enabled workload; duplicate local profile IDs fail closed. The map is a
deployment-owned control file, not a public catalog, and contains no paths,
model names, credentials, or checkpoints. The Hub manifest supplies only the
external profile ID, dataset/model/checkpoint IDs, resource class, and
justification. The worker resolves all remaining settings from its local
catalog and binding.

The V3 integration is an explicit private exception to the public settings
catalog, never an implicit translation. A deployment-owned entry may include
an exact `native_execution` object with schema
`ida-native-execution-request.v1`, the canonical V3 profile ID, backend,
precision profile, optimizer, and attention backend. The local binding and
config must match those values exactly, and its `execution` entry must pin
`binary_sha256`. The worker then requires the Hub attestation, map pin, local
binding, and measured binary hash to agree. A V3 profile that would otherwise
require AdamW or WGMMA is rejected unless this private contract is present and
matched. It is never downgraded to the public Lion/scalar route.

The ignored deployment map contains IDs and execution allowlists only; it
must not contain paths, commands, model bytes, credentials, or training
settings:

```json
{
  "schema_version": "neural-foundry-deployment-map.v1",
  "hub_profiles": {
    "edge-full": {
      "local_profile_id": "public-edge-full",
      "trainer_versions": ["ida-native-v3"],
      "execution": {
        "backend": "cuda",
        "artifact_id": "ida-native-cuda-v3",
        "binary_names": ["ida_native_train", "canopy_foundry_train"]
      }
    }
  },
  "resource_classes": {
    "gpu-standard": { "device": 0 }
  }
}
```

Repeat the profile entry explicitly for every enabled external workload; a
profile never inherits another profile's local config. A missing map entry,
execution mismatch, trainer-version mismatch, native-contract mismatch,
invalid device, or missing local binding fails closed. Public model
availability does not authorize the worker to send model bytes or filesystem
paths to the Hub.

For a V3 private entry, the additional map section is shaped like this (the
binary digest must be replaced by the deployment's measured digest; the
placeholder is intentionally not executable):

```json
{
  "execution": {
    "backend": "cuda",
    "artifact_id": "ida-native-cuda-v3",
    "binary_names": ["ida_native_train"],
    "binary_sha256": "<sha256 of the approved V3 binary>"
  },
  "native_execution": {
    "schema_version": "ida-native-execution-request.v1",
    "profile_id": "edge-full",
    "backend": "cuda",
    "precision_profile": "legacy_bf16",
    "optimizer_type": "adamw",
    "attention_backend": "hopper_wgmma_packed_fp4"
  }
}
```

The worker records the validated native descriptor in the private run
directory and separately writes the low-level request consumed by the native
binary. The low-level V3 request intentionally omits Foundry supervisor paths
such as `repo_root`, `status_file`, `job_id`, and
`expected_terminal_phase`; the V3 parser rejects or does not consume those
fields.
V3 emits its completion receipt as `metrics.json`. For CUDA, Foundry accepts
that receipt only when it names the worker-owned output directory and a
present `model.safetensors`; CPU and OpenCL smoke receipts retain their
explicit no-checkpoint result. Neither the descriptor nor the low-level
request is returned to the Hub; worker updates contain only the bounded
public progress envelope.

### Use your own config, dataset, model, or checkpoint

The local binding is the supported bring-your-own-assets interface. People can
keep their own files outside Git and expose only opaque IDs to the Hub:

1. Copy an example config into a local-only profile, such as
   `configs/local/profiles/my-profile.json`, and edit the model, input, and
   training sections for your own run. Keep only the selected backend,
   precision, optimizer, and attention implementation inside the public
   capability allowlists.
2. Place the dataset under the configured dataset root. Place a base model or
   resume checkpoint under the configured artifact root. Model and checkpoint
   directories are accepted; every file is hashed before use.
3. Add opaque IDs to the local binding's `profiles`, `datasets`, `models`, or
   `checkpoints` map. Each entry needs a relative `ref` and its SHA-256 digest.
4. Submit the matching Hub-owned IDs (`training_profile_id`, `dataset_id`,
   `base_model_id`, or `checkpoint_id`). Absolute paths and raw model names do
   not cross the API boundary.

For example, a local-only binding can register a user model and checkpoint as:

```json
{
  "models": {
    "model-my-lab": {
      "ref": "models/my-lab",
      "sha256": "<sha256 of the model tree>"
    }
  },
  "checkpoints": {
    "checkpoint-my-lab": {
      "ref": "checkpoints/my-lab-step-100",
      "sha256": "<sha256 of the checkpoint tree>"
    }
  }
}
```

The worker rejects absolute paths, traversal, symbolic links, missing pins,
and assets whose bytes change after registration. It writes the resolved paths
only into the ignored local `native-request.json`; public progress contains
status and bounded metrics, never those paths or model bytes.

### Deployment-owned training settings

The deployment-local profile is the source of truth for model shape, input layout,
precision, optimizer, attention implementation, learning rate, batch sizes,
seed, and other native training settings. Copy an example into an ignored
local profile only when provisioning a deployment, then update the local
binding's SHA-256 pin. The Hub does not receive those settings and the worker
rejects a remote `settings` object, paths, commands, or local execution fields.

The Hub selects only opaque IDs and policy bounds: use `from_scratch` with no
parent, `fine_tune` with a locally pinned base model, or `resume` with a
locally pinned checkpoint. The Hub, native adapter, and worker reject a mode
that does not match its parent, and the worker enforces the approved
`max_steps` ceiling against the local config.

Build the public native target with private observability disabled. The
canonical public build also sets `CANOPY_FOUNDRY_PUBLIC_BUILD=ON` and fails
configuration if a stale cache enables private observability; it can therefore
never enter a native burn requiring ontology or analytics sinks. A private
deployment may explicitly set
`-DCANOPY_FOUNDRY_PUBLIC_BUILD=OFF -DIDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY=ON`;
that binary and its binding remain deployment-owned and must not be used as
public release evidence or copied into the public repository.
The private observability headers are ignored local overlay files; public
builds resolve the no-op contracts under `include-public/ida_native/` instead.
The public CPU boundary can be configured with `cmake --preset public-cpu` and
built with `cmake --build --preset public-cpu`.

Do not copy Hub policy JSON, bearer tokens, private training code, or absolute
paths into the public catalog or native request. Private Flight Deck
entitlements and donation/telemetry APIs remain Hub-owned and are not callable
from Neural Foundry.

### Hub evaluator and leaderboard provisioning

The evaluator token map is configured only in the Hub's private Function
settings as `NEURAL_FORGE_EVALUATOR_TOKENS`. The worker token cannot record an
evaluation gate, score, or publication confirmation. The Hub's private policy
may attach a bounded `leaderboard_metric` to an evaluation request and a
`public_id` to a model metadata entry; the public catalog and deployment map
remain neutral. Operators must opt in on submission and confirm publication
after a successful, evaluator-gated run. Configure
`NEURAL_FORGE_LEADERBOARD_ID_KEY` in the Hub and
`NEURAL_FORGE_LEADERBOARD_ATTESTATION_KEY` on the Hub together with
`NEURAL_FORGE_GPU_FINGERPRINT_KEY` on each worker as separate deployment
secrets. The worker maps CUDA to an NVIDIA probe and OpenCL to an AMD probe;
an unexpected vendor is ineligible. Never put any key in a manifest, binding,
catalog, or native request.

Keep the worker token separate from user and control tokens. Run it with a
dedicated local account whose permissions are limited to the configured data,
artifact, binary, and run roots. Promotion, approval, and other authority
transitions remain outside both the Hub facade and this worker.

## Ownership summary

Neural Foundry owns native computation, native request admission, checkpoint
compatibility, and observed runtime output. The governed external wrapper owns
who may run it, which organization and role the run belongs to, why it is
allowed, how state changes are audited, and whether any result may be
promoted.
