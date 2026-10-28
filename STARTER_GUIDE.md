# Canopy Foundry Starter Guide

If you are new to the codebase, begin with the [learning path](docs/learning-path.md)
before using this operator-oriented guide.

This guide builds and invokes the native runtime directly. The former
loopback adapter in `api/native_api.py` is retired; the governed wrapper still
owns authority, execution, audit, and promotion.

## Getting started in Foundry

Foundry opens with IDA. Choose **Train Brain** to switch to the Weave planning
workspace, and **Back to IDA** to return. **Help / Guide** (or `H`) reopens
the startup guide. The guide preference is local and does not affect a selected
request or a run.

After building the native `foundry` command, use `foundry --help` and
`foundry capabilities --pretty`. Check the actual device with `nvidia-smi` and
check the toolkit with `nvcc --version`. A capability inventory alone does
not prove that this device can run training. CUDA requires a supported GPU,
toolkit, and matching architecture; CPU and OpenCL training profiles are
disabled.

`foundry plan configs/examples/planning_only_request.json --pretty` exercises a
one-step request shape. Planning reads this file and does not create its placeholder
output paths or start a run. The example has no training dataset and is for planning
only; replace its paths and provide real inputs before any native execution. Run
the example command from the repository root, or from the installed
`share/canopy-foundry` directory. If it is rejected, inspect the error, correct the request or prerequisites,
and try again. In the shell, select your local JSON and choose **Resolve
selected request** to use the same planner. A plan is not a runtime receipt.

For checkpoint inspection use `foundry checkpoint DIRECTORY` or the shell's
**Inspect resume** after selecting a request with `resume_from_checkpoint`.
These check only file-pair presence and bounded header prefixes; the native
loader must still validate format, lineage, generation, and compatibility.
The shell's **Check Run** reports request, plan or checkpoint problems without
starting a run. The separate **Run**, **Refresh status**, and **Cancel** controls,
and the CLI's `foundry run REQUEST.json`, `foundry status RUN-ID`, and
`foundry cancel RUN-ID`, report that the local service is unavailable. They
cannot start or cancel a run in this build and do not control a Hub-managed job.
Local execution is not yet connected. The direct
native invocation below is a separate local path; it does not require a Hub
job, and it does not provide the new shell's run/cancel service. The existing
Hub-controlled worker requires an attested binary, local bindings, and a
Hub-issued job. A build inventory or CPU smoke executable does not make
training available.
See the [local guide](docs/neural-foundry-guide.md) for the native training
contract and [portability notes](PORTABILITY.md) for backend limits.

## 1. Build

From the repository root:

```sh
cmake -S . -B build -DIDA_NATIVE_CUDA_ARCHITECTURES=90a
cmake --build build --config Release --target ida_native_train
```

For portable smoke builds without CUDA:

```sh
cmake -S . -B build-cpu -DIDA_NATIVE_ENABLE_CUDA=OFF \
  -DIDA_NATIVE_AMD_ENABLE_OPENCL=OFF
cmake --build build-cpu --config Release --target ida_native_cpu_train
```

The OpenCL smoke target is `ida_native_opencl_train` and requires OpenCL
development files. The portable sources are retained for future Lion support,
but CPU/OpenCL profiles are currently disabled because their only implemented
optimizer path was AdamW. Adam and AdamW are rejected; the enabled CUDA path
uses Lion only.

For an A100-class device, configure with an appropriate architecture such as
`80`. The architecture must match the installed GPU and CUDA toolkit.

## 2. Prepare native data

The trainer consumes a directory containing:

- `tokens.u32`: contiguous little-endian `uint32` token IDs;
- `labels.i32`: contiguous little-endian `int32` labels with the same byte size;
- `segs.u16`: optional little-endian `uint16` segment offsets, with exactly one
  value per token when present.

The token and label blocks must be non-empty and contain complete rows of the
requested sequence length. A supplied segment block must be present and exact;
the runtime will reject a mismatch instead of falling back to full-causal
attention.

## 3. Write a request

The minimum shape is:

```json
{
  "backend": "native",
  "output_dir": "/absolute/path/to/run-output",
  "status_file": "/absolute/path/to/run-output/status.json",
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
    "token_blocks": "/absolute/path/to/data/tokens.u32",
    "label_blocks": "/absolute/path/to/data/labels.i32",
    "batch_size": 1,
    "sequence_length": 2048
  }
}
```

The request is intentionally a native capability contract. It must not carry
organization, role, justification, transition, audit, promotion, queue,
telemetry, ontology, or evidence fields, including nested fields.

## 4. Run

```sh
bin/ida_native_train \
  --request-json /absolute/path/to/request.json \
  --device 0
```

The CUDA process records setup, training, checkpoint, and terminal phases in
the request's status file and writes checkpoint/metrics artifacts under the
output directory. CPU and OpenCL smoke binaries emit bounded JSONL events on
stdout. Existing output and incomplete resume checkpoints are rejected rather
than treated as successful or reused.

## 5. Optional local kernel telemetry

Kernel telemetry is a separate, local-only opt-in. It is disabled unless the
operator supplies `--kernel-telemetry DIR` on a successful CUDA training run:

```sh
bin/ida_native_train \
  --request-json /absolute/path/to/request.json \
  --device 0 \
  --kernel-telemetry /absolute/path/to/fresh-telemetry
```

Foundry writes a fresh directory containing `kernel_ontology.jsonl`,
`training_metrics.jsonl`, and `receipt.json`. The bounded public schema covers
launch geometry, host enqueue observations, and step diagnostics. It does not
claim device elapsed kernel time, hardware-counter profiling, or promotion
eligibility. A non-empty output directory is rejected so records from separate
runs cannot be mixed. The producer fails closed at 32 MiB or 250,000 ontology
records and does not write a valid receipt after a telemetry write failure. The
public executable ignores ambient `IDA_NATIVE_ONTOLOGY` settings; only the
explicit flag arms the in-process local export.

The split is intentional: Canopy Foundry is the local producer and never makes
a network request; `IDA_CLI-Public` is the explicit validator and transport.
Inspect without network access using `ida telemetry share-kernel DIR
--dry-run`, or share only to an operator-supplied HTTPS endpoint with
`ida telemetry share-kernel DIR --url https://host/path`. There is no default
receiver, background upload, or credential persistence.

## 6. Wrapper boundary

The external Python wrapper owns config translation, process supervision,
operator-facing policy, and governed control-plane integration. The exact
native command, input/output files, event meanings, and failure semantics are
documented in [`docs/python-wrapper-contract.md`](docs/python-wrapper-contract.md).

The Neural Forge worker is the local implementation component of Neural Foundry
and the compatibility name for the local wrapper variant. It resolves
Hub-issued relative references under local roots, verifies the v2
backend-specific binary/hash attestation, writes an ignored native request
manifest, launches the selected native target, and sends only bounded status
back to the Hub. It does not select PyTorch or expose native methods through
HTTP.
