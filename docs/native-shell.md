# Native Foundry shell

The native shell is a small local presentation layer for Canopy Foundry. It
does not replace the native execution planner and it is not a Hub client.

## Build

The shell is enabled by default.

    cmake -S . -B build -DIDA_NATIVE_ENABLE_CUDA=OFF -DIDA_NATIVE_ENABLE_AMD=ON
    cmake --build build --target foundry_shell

Windows builds use the Win32/GDI backend and do not require a web runtime.
Linux builds use X11 when the development headers and library are discoverable:

    sudo apt install libx11-dev
    cmake -S . -B build -DCANOPY_FOUNDRY_ENABLE_X11_SHELL=ON
    cmake --build build --target foundry_shell

If X11 is unavailable, the target still builds a diagnostic shell binary that
supports --check and --headless but intentionally refuses to pretend that a
native window backend exists. Wayland-native presentation is a follow-up
backend, not an implicit X11 claim.

## Run

The native shell exposes the same local presentation contract on Windows and
Linux, but the build command depends on the platform.

### Windows / Win32-GDI

From PowerShell at the repository root:

    cmake -S . -B build -DIDA_NATIVE_ENABLE_CUDA=OFF
    cmake --build build --target foundry_shell --config Release
    .\\bin\\foundry_shell.exe --check
    .\\bin\\foundry_shell.exe --document path\\to\\request.json
    .\\bin\\foundry_shell.exe --simulate

The Windows backend is implemented with Win32/GDI. It does not require X11,
a browser runtime, or a web server.

### Linux / X11

Install the development package and enable the X11 backend:

    sudo apt install libx11-dev
    cmake -S . -B build -DCANOPY_FOUNDRY_ENABLE_X11_SHELL=ON
    cmake --build build --target foundry_shell
    ./bin/foundry_shell --check
    ./bin/foundry_shell --document path/to/request.json
    ./bin/foundry_shell --simulate

The Linux backend requires discoverable X11 headers and libraries. Without
them, the repository still builds a diagnostic/headless shell when possible,
but it deliberately does not pretend to provide a native window.

### Shared commands and controls

From the repository root, the platform-specific commands above support:

    bin/foundry_shell --document path/to/request-or-receipt.json

Useful diagnostics:

    bin/foundry_shell --check
    bin/foundry_shell --headless --document path/to/request.json

The shell can inspect request, plan, capability, and receipt-shaped JSON. It
does not verify a receipt, launch a worker, mutate a checkpoint, contact the
Hub, or expose a network endpoint. The frontend directory is installed beside
the binary as a static asset so the shell can verify its local presentation
surface without depending on a browser service.

The keyboard controls are shared by both native backends:

- `S` starts or restarts the execution theater.
- `P` or `Space` pauses/resumes it.
- `R` resets it.
- `1`–`5` select the existing inspection views.

The simulation is intentionally synthetic on Windows and Linux. It is an
instructional visualization of request, plan, route, execute, and checkpoint
stages; it is not a training run, checkpoint, Hub job, or execution receipt.

## IDA presentation atlas

`frontend/assets/ida-cli-sprite.json` is a generated, terminal-ready atlas
containing the IDA and multi-pose WALK sets used by the Ask IDA CLI's local
opening. The native shell prefers WALK, holds each pose for six animation
ticks, and falls back to the single IDA pose when a reduced atlas is supplied.
Windows and X11 render the same quadrant masks and palette through their native
painting APIs; no terminal, WebView, image library, model runtime, or network
connection is required.

The atlas is presentation-only. The shell deliberately does not use its frame
state to infer model availability, execution progress, receipt validity, or Hub
authority. A missing or malformed atlas leaves the shell usable and reports the
missing presentation asset through `--check` rather than claiming runtime
capability.

## Execution theater

Launch the deterministic local simulation with:

    bin/foundry_shell --simulate

The theater walks a toy MoE-shaped workload through `REQUESTED`, `RESOLVED`,
`ACCEPTED`, `ROUTING`, `EXECUTING`, `CHECKPOINTING`, and `PERSISTED`. It draws
synthetic route loads, token movement, layer selection, and memory pressure so
the architecture is understandable before a real backend is involved.

Controls:

- `S` starts or restarts the simulation.
- `P` or `Space` pauses/resumes it.
- `R` resets it.
- `1`–`5` retain the existing inspection views; the simulation is also
  available from the Playground row and the Overview button.

Every theater value is generated locally and labeled as synthetic. The theater
does not allocate a model, launch training, create a checkpoint, contact the
Hub, or upgrade a displayed stage into execution evidence.
