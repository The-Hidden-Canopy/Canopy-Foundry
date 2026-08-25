# Canopy Foundry request admission

The former loopback HTTP adapter is retired. This directory contains only an
in-process request validator for the governed external wrapper; it does not
bind a port, accept bearer tokens, launch a child process, or publish
telemetry.

The native executable remains a one-shot CLI. The `ida_native_train` name is
retained for wrapper compatibility; the public runtime identity is Neural
Foundry.

```text
bin/ida_native_train \
  --request-json /path/to/request.json \
  --device 0
```

The governed wrapper owns organization scope, role tier, justification, state
transitions, audit/domain events, approval, process supervision, resource
limits, environment construction, and telemetry publication.
