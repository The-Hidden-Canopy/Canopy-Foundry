# Optional Windows local execution owner

The local owner is an unavailable prototype of the retained-owner contract.
Local Run requires an existing Hub-issued job. This prototype has no adapter
that verifies that admission, so setting `CANOPY_FOUNDRY_ENABLE_LOCAL_OWNER=ON`
fails configuration. An operator binary pin, endpoint ACL or containment test
does not grant run authority. The default and explicit OFF remain disabled.

Use the existing authenticated, organization-scoped Hub-managed worker in
`scripts/neural_forge_worker.py` for admitted jobs. This change does not add
a native Hub adapter, create a job, change worker permissions, or authorize
execution, evaluation or promotion.

The standalone native CLI and Hub-managed worker remain available through
their existing contracts. This owner does not create a Hub job, use a worker
token, record evaluator gates, publish telemetry, or authorize promotion.

When the owner build option is OFF, the CLI rejects explicit owner endpoints
and the shell/CLI service ignores the owner-endpoint environment setting. The
low-level owner interfaces remain compiled for synthetic contract fixtures;
their presence does not enable a foreground owner executable or a user run.
A disabled Windows build also compiles the bootstrap source as an object-only
check, without creating an executable. Synthetic fixtures do not authorize
enabling or operating the prototype.

## Build and verify the candidate

On Windows, build the native targets in an isolated build tree. Obtain the
project's coordinated build/test slot before running checks. The synthetic
fixture impersonates a native command protocol and runs no CUDA or training.

```text
cmake --build build --target foundry_local_backend_fixture_test foundry_shell_plan_service_contract_test foundry_owner_pipe_fixture_test foundry_plan --config Release
ctest --test-dir build -C Release -R "foundry_(local_backend_fixture|shell_plan_service_contract|owner_pipe_fixture|cli_unavailable)$" --output-on-failure
```

The default/OFF policy and explicit ON refusal are covered by the three
`foundry_local_owner_admission_*` CTest cases. No trainer or owner is launched.

## Unavailable prototype protocol

The following describes source-level containment and protocol mechanics for
review and synthetic fixtures. It is not an operational Run path. A future
adapter must validate an existing Hub-issued job using the established worker
contract before this prototype can be enabled. No operator-only exception or
caller-supplied job identifier establishes admission.

`PRIVATE_ENDPOINT`, `APPROVED_BINARY` and `LOCAL_RUN_ROOT` must be absolute local
paths. The endpoint is created exclusively with a restrictive ACL. The client
verifies the endpoint's ACL and the server PID plus process creation time.
Do not overwrite a stale endpoint or infer liveness from a PID alone.

On Windows, this source version of the Foundry and native trainer entry points
reads wide command-line arguments, and request JSON paths are decoded as UTF-8.
The pinned native executable must be built from a compatible version for this
to apply; an older binary may still fail on non-ASCII run paths. Local Unicode
path handling is not a substitute for runtime and artifact validation.

For the native shell, set `CANOPY_FOUNDRY_OWNER_ENDPOINT` in its launch
environment to that endpoint file. Run, Refresh status and Cancel use the
same transport. An empty local run ID refreshes `current`, which can recover
the ID after a lost start response. Without the setting, the existing
unconnected behavior is preserved. Check Run and plan resolution still inspect
the client build; the owner independently reloads and resolves the selected
request for the pinned public CUDA executable. Check Run does not contact the
owner or verify its availability. The displayed plan is cleared when Run is
requested. After an unknown start, status or cancellation response, the shell
clears its run ID; use Refresh status to reconcile through the current retained
owner before trying to cancel or start again.

The foreground owner stays alive until its job is verified empty. Ctrl+C only
requests a quiescent exit; use Cancel to terminate an active run. A timeout or
broken response means unknown, not a completed cancellation. Reconcile through
the retained owner before retrying. Unknown ownership is fenced against a
second launch. A fresh verified observation of the same retained job can
restore control; an unverifiable job remains fenced.

Start returns a run ID while input preparation runs in a separate worker; this
means preparation was accepted, not that a child or training started. Cancel
can be received during hashing. The worker checks cancellation between file
read chunks and before launch publication and process creation. A blocked OS
file read itself cannot be interrupted by this mechanism. A cancellation racing
with probe creation remains pending until the retained owner verifies the job
empty; it never advances a cancelled probe to training. Refresh status until
the owner reports the final state. New starts stay fenced until then.

Before starting that worker, the owner loads at most 2 MiB of request JSON,
matching the backend's existing source limit, and refuses larger files before
calling the backend. Check Run uses the same bound. Request loading and
planning still use the serialized control path; an individual stalled
filesystem read is not interruptible here.
The ordinary native request loader retains its existing size behavior.

Launch provenance is first flushed to `launch-provenance.pending`, then renamed
to `launch-provenance.json` only after the full record exists. Interrupted
preparation can leave the pending file and partial staged inputs for inspection;
they are not a completed launch record. Never overwrite them to force a retry.

For an uncertain process-record write, the retained owner stops its owned job,
preserves the original file and reports an unresolved result. `foundry current`
explicitly retries observation using retained handles and writes at most two
separately named recovery records. If these attempts fail, the run remains
fenced for operator investigation. A verified empty probe followed by failure
before trainer creation reports trainer setup failure, not training success.

After owner loss or restart, run `foundry inspect-local-run RUN_DIRECTORY` to
identify missing, partial, mismatched and parseable local records. This command
is read-only, always reports an unresolved outcome and does not adopt a PID or
old process handle. Review the original run directory and independently verify
the old execution and containment state before deciding whether a fresh run is
appropriate. A new owner uses a new endpoint and cannot reconcile old handles;
do not replace the stale endpoint or treat its contents as proof of outcome.

## Supported path and preserved boundaries

This first implementation admits from-scratch, single-device public CUDA with
`scalar_flash`, `legacy_bf16` or `legacy_fp8`, Lion, and existing public model
contracts. It preserves omitted request defaults rather than substituting a
new precision or backend. It rejects unknown top-level execution fields,
shared-expert layouts and foreign architecture contracts.

Fine-tune, checkpoint resume, model-parallel execution, accelerated private
formats and architecture-workbench graph packages require their own runtime
validators/adapters. The limited pair/header checkpoint inspector cannot
authorize any of those paths. Existing direct native and governed worker paths
are unchanged; this optional owner is not a silent fallback for them.

Native token/label files are pinned as nonempty u32/i32 blocks of matching
length; optional u16 segment blocks must have the matching byte count. Native
validation still owns shape, vocabulary, tensor and numerical checks. Relative
input paths retain native CLI current-directory semantics: the foreground
owner resolves them from its own startup working directory and freezes the
absolute paths. Use explicit absolute inputs when launching from another
directory.

The owner hashes and holds the executable and input files without write/delete
sharing. It rejects reparse points and locks path ancestors against replacement.
Before staging, it compares the exact pinned source JSON bytes with the bytes
parsed for admission. If any field changes, even one omitted from the planner's
hash, the owner refuses before creating a child. Re-select or re-run the check
after intentional edits; this comparison does not validate model outcomes.
It freezes the selected JSON into a new private run directory, assigns that
run's own output/status paths and ID, reparses the frozen request, and checks
its plan binding. Caller output directories are never reused. The source JSON,
staged request, binary and input hashes, plan hash and selected device are
recorded in local launch provenance.

The owner creates each child suspended, assigns it to a fresh Job Object before
resuming, disables breakaway, and inherits only the three standard handles.
The environment carries system basics and run-owned temporary paths, with no
inherited precision flags, accelerator selection, credentials or loader paths.
The executable must carry/find its required deployment libraries without an
inherited PATH.

The first child runs the pinned executable's existing `--self-check`, including
the requested architecture floor, cuBLASLt and private-pool checks. Only a
successful probe with zero remaining job processes can advance to training.
Cancellation and uncertain observations prevent that transition. Cancellation
uses the retained Job Object, never `taskkill` or recycled process IDs.

The default limits are one owned job, 3,600 seconds across probe/training,
8 GiB of run output, 32 MiB per stage log, and 10,000 output files. The time
limit can be set with `--max-seconds` (1–86,400). Library callers can set the
byte limits in `LocalBackendConfig`. Exceeding limits terminates only this
owned job and preserves the failure evidence. Larger workloads should use a
reviewed configuration or the established governed path.

`stopped` requires both a signaled root handle and zero active job processes;
a root exiting while descendants remain is still active. Stage stdout/stderr
are local files, flushed and hashed after the whole job empties. Process
records retain the actual exit code and cancellation state, and always say
`training_outcome: unverified` and `promotion_eligible: false`. Native metrics,
checkpoint integrity/resume, evaluation and promotion remain separate checks.

The job closes with kill-on-close if the owner is forcibly terminated, but
that OS cleanup is not a verified outcome or permission to fabricate a receipt.
The containment contract covers descendants created within the owned job;
external schedulers, services and out-of-process dispatch are not supported.
