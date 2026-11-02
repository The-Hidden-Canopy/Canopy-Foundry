# Native presentation assets

`ida-cli-sprite.json` is the generated, terminal-ready IDA presentation atlas
used by the local native shells. It carries presentation frames only: no model
weights, runtime controls, telemetry, credentials, or execution evidence are
encoded in the asset.

The shell prefers the `WALK` set and falls back to the single-pose `IDA` set
when a deployment supplies a reduced atlas. The optional `presentation_frames`
array selects the quiet idle and walk poses for the companion; the run and dash
source frames remain in the atlas. Without this array the shell plays all frames.
Animation is a local
window effect; it does not mean that a model is loaded, a run is active, or a
Hub job exists.

The optional `NATIVE_RGBA` entry is a 256 x 256 full-colour, transparent
single idle pose for native windows. It was generated at 1254 x 1254 from the
existing IDA visual reference and downsampled; it is a new interpretation,
not a claim that the original 50 x 58 animation cells contained 256-pixel
detail. The single pose breathes gently; the original terminal frames remain
unchanged and are used if the optional entry is absent or malformed.

Portions derived from The Hidden Canopy LLC's
[Ask_IDA_CLI](https://github.com/The-Hidden-Canopy/Ask_IDA_CLI) visual reference.
Used with permission for this requested presentation candidate.

Copyright (c) 2026 The Hidden Canopy LLC. All rights reserved. This asset is an
authorized presentation port for the Canopy Foundry source-available product.
