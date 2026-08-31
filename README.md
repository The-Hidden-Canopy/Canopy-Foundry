# Neural Foundry

Neural Foundry is built for users who want direct control over their data,
models, checkpoints, configuration, and execution environment.

The public repository contains the runtime capability. The Hidden Canopy Hub
remains the authenticated coordination and authority layer.

The Hub-to-worker integration accepts the canonical
`neural-forge-worker-manifest.v2` envelope. A deployment can explicitly map a
Hub profile to IDA Train v3 through a private
`ida-native-execution-request.v1` contract in the deployment map. That private
contract pins the V3 profile settings and, optionally, the binary digest; it
does not alter the public capability catalog or translate AdamW/WGMMA requests
to the public Lion/scalar route.

## Public Boundary and Runtime Hardening Update

The current public release is commit [`145c7b2`](https://github.com/The-Hidden-Canopy/Canopy-Foundry/commit/145c7b2fe169c19726d0d92074925b87f7f39eb8).

This update strengthens the boundary between local training capability and the
separate Hidden Canopy Hub coordination layer.

It adds:

- private-runtime package and binary integrity verification using SHA-256 pins
- exact model and checkpoint architecture-contract matching
- explicit acknowledgement for experimental profiles and model contracts
- manifest fingerprint validation against stale or substituted requests
- symlink and directory-substitution protection for local receipts
- history-aware public-boundary scanning for private paths and artifacts
- expanded adversarial coverage for local asset escapes, private fields, raw
  telemetry, contract substitution, and legacy request bypasses

The public tree contains no private runtime packages, model data, checkpoints,
credentials, local filesystem paths, raw logs, or private historical paths.
The public boundary and reachable Git history both pass the release checks, and
84 Foundry tests pass for this update.

This is a source-available release, not a claim of universal hardware or
production parity. Native CMake, CUDA, OpenCL, and hardware-specific
performance validation remain dependent on the operator's environment.

Do not add local backups, Hub control-plane files, v2 blackboard content,
datasets, models, checkpoints, credentials, logs, or private runtime packages
to this repository.

Neural Foundry is the public product and runtime identity. The local component
is called the Neural Forge worker in implementation documentation and
compatibility interfaces. Names such as `neural_forge_worker.py`,
`neural-forge-binding.json`, and `NEURAL_FORGE_*` are retained implementation
identifiers, not a second public product.

## License and Use

Neural Foundry is a public source-available repository under the [Hidden
Canopy Source-Available License v1.0](LICENSE). Personal, noncommercial
research and educational use are permitted. Commercial use, production
deployment, commercial incorporation, and paid model training require a
separate commercial license from The Hidden Canopy LLC.

## What Neural Foundry Does

Neural Foundry provides native local model training through a C++20/CUDA
runtime with explicit validation, device handling, checkpoint contracts, and
locally owned configuration.

Users bring their own:

- architecture configuration
- tokenizer and data layout
- datasets
- models
- checkpoints
- training schedules
- output locations

Supported workflows include:

- training from scratch
- training from a locally pinned base model
- resuming from a locally pinned checkpoint

The runtime does not require PyTorch or another Python training framework for
execution.

## Local Ownership

Sensitive training assets remain on the machine performing the work.

Neural Foundry does not send the Hub:

- dataset or model bytes
- checkpoints
- filesystem paths
- private kernels
- credentials
- raw stdout or stderr
- private telemetry
- private training configurations

The Hub coordinates jobs using bounded metadata, opaque identifiers, resource
classes, lineage, and approved policy fields.

Local deployment bindings resolve those identifiers into the actual models,
datasets, binaries, checkpoints, kernels, and hardware available on the
operator's machine.

The Hub coordinates the job. It does not take custody of the training
environment.

## Native Execution

The enabled training path runs directly through native C++20 and CUDA.

This provides explicit control over:

- device execution
- memory
- kernels
- precision
- checkpoint behavior
- request validation
- runtime state

Native execution is intended to reduce framework dependency and keep the
training runtime close to the underlying hardware.

Performance should be evaluated through hardware-specific benchmarks rather
than assumed from architecture alone.

## Fail-Closed by Design

Neural Foundry rejects ambiguous, incomplete, stale, or unsupported requests
instead of silently modifying them.

Examples include:

- stale output directories
- incomplete checkpoints
- partial datasets
- malformed tensor blocks
- absolute paths
- path traversal
- symbolic-link escapes
- arbitrary trainer selection
- caller-selected GPU ordinals
- caller-supplied private configuration paths
- unsupported optimizers
- authority-bearing request fields

The current enabled CUDA profile uses Lion. Adam and AdamW are intentionally
disabled in that profile.

## Model Portability

Native contracts cover several common model layouts, including:

- GPT-2-compatible models
- Llama-shaped dense models
- SmolLM-shaped dense models
- Mixtral/Qwen-shaped top-k SwiGLU models
- generic mixture-of-experts layouts
- native SafeTensors manifests
- explicit model, dataset, and checkpoint bindings

Model conversion occurs outside the public runtime.

The repository does not ship private model weights, automatic model downloads,
or a framework lockfile.

Compatibility is defined through explicit contracts rather than claims that
every architecture has already been fully validated.

## Integrity and Reproducibility

Local resources are referenced through opaque identifiers and SHA-256 integrity
pins.

Bindings may cover:

- models
- datasets
- checkpoints
- binaries
- kernels
- execution profiles

This allows an operator to establish which local assets were used, whether the
expected artifact was selected, and whether a resumed run came from a
complete checkpoint.

## Bounded Telemetry

Observability is intentionally limited.

Public builds are designed to avoid exposing private training information
through operational telemetry.

Worker events are:

- size-limited
- schema-bounded
- path-scrubbed
- free of raw stdout and stderr
- separated from private analytics and ontology paths

Leaderboard participation, where available, is opt-in and intended to expose
only approved public identifiers, sanitized hardware information, approved
metrics, and a deployment fingerprint.

## Portability

The repository includes explicit portability paths for:

- CUDA training
- CPU smoke testing
- AMD OpenCL smoke testing

CUDA is currently the enabled training path.

CPU and OpenCL profiles remain experimental or disabled and should not be
treated as production-equivalent backends.

## Getting Started

New users should begin with the included:

- [starter guide](STARTER_GUIDE.md)
- public example configurations in `configs/examples/`
- [secure operations documentation](docs/neural-foundry-guide.md)
- [private advanced-runtime boundary](docs/private-runtime-boundary.md)
- CLI guidance in the secure operations documentation
- offline [request-admission workflow](api/README.md)

An optional ignored Hugging Face/Git cache may be used for locally managed
assets without placing credentials, model repositories, or private artifacts
into the public repository. See [the local cache guide](docs/hf-git-cache.md).

Neural Foundry performs no hidden login, automatic model retrieval, or
background upload.

## Support the Work

Donations help fund continued research and development across The Hidden
Canopy ecosystem, including Neural Foundry, local training infrastructure,
evaluation, portability, and public technical work.

Support helps us keep experimenting, validating new approaches, testing across
hardware, and releasing useful capability publicly without turning user data
or private training environments into the product.

If you would like to support the work, donations can be made through the
[PayPal link on the Hidden Canopy Hub](https://thehiddencanopy.com/flight-deck.html#support).

## What Neural Foundry Is Not

Neural Foundry is not currently:

- a hosted model-training service
- a replacement for every PyTorch workflow
- a public model marketplace
- a model-weight repository
- an automatic Hugging Face downloader
- a general-purpose remote shell
- a public HTTP API exposing native methods
- proof of performance across every GPU
- a claim that CPU or OpenCL paths have production parity

## Design Principle

> Train on your own machine, with your own data and your own configuration.

The public repository contains capability, not authority.

Private deployment code, private kernels, local configurations, models,
datasets, checkpoints, credentials, and other sensitive training assets remain
local while the Hidden Canopy Hub provides a narrow coordination boundary for
approved workloads. Public runtime code remains in the repository as the
capability being shared.
