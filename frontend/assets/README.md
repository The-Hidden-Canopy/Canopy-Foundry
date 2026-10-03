# Native presentation assets

`ida-cli-sprite.json` is the generated, terminal-ready IDA presentation atlas
used by the local native shells. It carries presentation frames only: no model
weights, runtime controls, telemetry, credentials, or execution evidence are
encoded in the asset.

The shell prefers the multi-pose `WALK` set and falls back to the single-pose
`IDA` set when a deployment supplies a reduced atlas. Animation is a local
window effect; it does not mean that a model is loaded, a run is active, or a
Hub job exists.

Copyright (c) 2026 The Hidden Canopy LLC. All rights reserved. This asset is an
authorized presentation port for the Canopy Foundry source-available product.
