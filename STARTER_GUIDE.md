# Canopy Foundry Starter Guide

This guide builds and invokes the native runtime directly. The former
loopback adapter in `api/native_api.py` is retired; the governed wrapper still
owns authority, execution, audit, and promotion.

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

## 5. Wrapper boundary

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
