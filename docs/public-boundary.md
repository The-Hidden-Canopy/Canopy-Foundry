# Canopy Foundry Public/Local Boundary

Canopy Foundry is the required local/native repository. The Hidden Canopy Hub
is the remote API and authority layer. A separate evidence blackboard is not a
runtime dependency and does not receive runtime imports or direct writes.

Some executable names, include paths, namespaces, and environment variables
retain legacy compatibility identifiers such as `ida_native` and
`NEURAL_FORGE_*`. These are implementation contracts, not additional public
products or authority layers.

## Boundary ownership

| Layer | Owns | Must not receive |
| --- | --- | --- |
| Canopy Foundry | Native C++/CUDA runtime, local assets, request admission, bounded status | Organization authority, promotion decisions, Hub policy, raw private telemetry |
| Hidden Canopy Hub | Authentication, organization/role permissions, job admission, opaque IDs, bounded status API | Local filesystem paths, dataset bytes, checkpoints, private kernels, raw stderr |
| Evidence blackboard | Reference history and evidence | Runtime imports, direct writes, public release dependencies |

## Public repository contents

The published tree may contain:

- native C++/CUDA and portability source;
- public request and capability schemas;
- safe example configurations and deterministic contract fixtures;
- the offline request-admission helper;
- the local worker contract, which publishes bounded status only;
- build, test, and boundary-review documentation.

The `api/` directory is deliberately not an HTTP API. It must not bind a
socket, accept bearer tokens, launch a child process, publish telemetry, or
make organization, role, audit, transition, or promotion decisions.

## Local-only contents

These remain on the training machine and are ignored by Git:

- `configs/local/` and `configs/private/`;
- worker bindings and generated request manifests;
- datasets, model weights, checkpoints, binaries, and run output;
- private observability headers and deployment-owned toolchains;
- credentials, certificates, raw stdout/stderr, and detailed telemetry.

Local references cross the Hub boundary as opaque IDs. The worker resolves
those IDs through its local binding, verifies integrity pins, and sends back
only bounded status and approved public event fields.

## Release check

Run the boundary validator before staging a public change:

```sh
python scripts/check_public_boundary.py
```

It checks the public working tree for nested repositories, private paths,
generated artifacts, raw telemetry, credentials, absolute machine paths, and
direct blackboard dependencies. A clean result is necessary but does not
replace review of the staged file list.

Before publishing a repository that has ever contained private source, also
run:

```sh
python scripts/check_public_boundary.py --history
```

History mode rejects private/generated source paths reachable from local Git
refs. A clean working tree alone is not evidence that old public commits are
safe to expose.
