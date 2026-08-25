# External Python Wrapper Contract

This document defines the narrow boundary for a future Python wrapper. The
wrapper is an external caller, not a service inside Canopy Foundry.

The executable names shown below retain legacy `ida_native` compatibility
identifiers. They are implementation names for the Neural Foundry runtime,
not a separate public product.

## Responsibilities

The wrapper may:

- resolve a user-approved model/training specification into one native JSON
  request;
- validate and stage native dataset/checkpoint paths;
- start one attested native process for the selected backend;
- stream and persist observed stdout/stderr;
- translate the final native result into the caller's governed contract.

The wrapper remains responsible for authentication, organization scope, role
tier, justification, state transitions, audit/domain events, promotion, and
operator approval. None of those fields may be passed through as native
request data.

The reviewed backend contract contains full CUDA plus explicitly constrained
OpenCL and CPU smoke modes. A wrapper must not turn the native boundary into a
generic PyTorch, CUDA, or Python-method proxy. Each backend requires its own
approved executable identity and SHA-256 attestation.

## Process contract

For CUDA, the wrapper invokes:

```text
ida_native_train --request-json <request.json> --device <nonnegative GPU>
```

OpenCL uses `ida_native_opencl_train --request-json <request.json> --device N`;
CPU uses `ida_native_cpu_train --request-json <request.json>`. The CPU target
must not receive a device argument.

The child process uses exit code `0` only after the canonical terminal status
phase and checkpoint/metrics artifacts have been written. A non-zero exit code
means the run did not complete and the wrapper must surface the error without
inventing metrics or reusing a stale result.

The child emits newline-delimited JSON objects on stdout:

- Portable smoke binaries emit `{"type":"step", ...}` progress records and a
  `{"type":"complete","status":"complete", ...}` record on success. The
  canonical CUDA binary uses its request-owned status file and local metrics
  streams instead of a public listener.

Stderr is diagnostic only. The wrapper must retain it for troubleshooting but
must not treat a diagnostic line as a successful state transition.

## Request rules

The wrapper must create a fresh request/output pair for each run. It must not
send organization, role, justification, transition, audit, domain-event,
promotion, queue, worker, telemetry, ontology, evidence, or socket fields at
any nesting level.

The native process rejects:

- missing or non-regular token/label blocks;
- empty, mismatched, or partial dataset blocks;
- explicitly supplied segment blocks with missing or incorrect sizes;
- an existing `metrics.json` without an explicit complete resume checkpoint;
- resume directories missing either `model.safetensors` or
  `optimizer_state.safetensors`;
- simultaneous `init_from_model` and `resume_from_checkpoint` values.

The wrapper must preserve those failures. It must not substitute cached data,
stale metrics, fabricated checkpoint status, or a fallback dataset.

## Compatibility

The request and checkpoint payloads retain the v3 native field and tensor
contracts. The executable name is `ida_native_train`; wrappers should
resolve it through configuration rather than hard-coding a workspace-relative
path.
