# AMD portability backends

This directory contains the AMD portability side of the Canopy Foundry native
tree.
The root native build can select it without requiring `nvcc`; the existing
CUDA target remains unchanged when CUDA is available.

It provides two intentionally narrow backends:

- `ida_native_opencl_train`: AMD OpenCL GPU smoke trainer for `gfx1036`.
- `ida_native_cpu_train`: FP32 CPU smoke trainer with an AMD AVX2/FMA path and
  scalar fallback.
- `ida_native_amd_train`: aggregate target that builds every enabled AMD
  backend.

These are native correctness/reference runtimes, not replacements for the CUDA
Swift trainer.
The source supports one dense layer, one attention head, batch size one,
sequence length 2048, FP32, fresh initialization, and gradient accumulation.
The public CPU/OpenCL profiles are disabled because their implemented update
path is AdamW. Adam and AdamW are rejected, and no Lion path is enabled on
these backends yet; they do not support BF16, FP8/FP4, MoE, LRSS/PSS,
checkpoint resume, or model parallelism.

The OpenCL executable also exposes a host-side C++ lifecycle check:

```bash
native/bin/ida_native_opencl_train --self-check --device 0
```

This exercises device selection, context and command-queue creation, OpenCL C
program compilation, RAII-managed buffers and kernels, argument binding,
dispatch, synchronization, and readback. The reusable wrapper is in
`native/amd/include/ida_native/opencl_cpp.hpp`; it uses the Khronos C API from
C++ and does not require the optional `opencl.hpp` bindings.

## Build

Install the Python profile and the host OpenCL development headers/ICD for the
AMD driver, then build without nvcc:

```bash
python3 -m pip install -r requirements/amd.txt
bash scripts/build_amd_opencl.sh
```

The CPU target can be built from the same v2 root without OpenCL headers:

```bash
cmake -S native -B native/build-amd-cpu \
  -DIDA_NATIVE_ENABLE_CUDA=OFF \
  -DIDA_NATIVE_ENABLE_AMD=ON \
  -DIDA_NATIVE_AMD_ENABLE_OPENCL=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build native/build-amd-cpu --target ida_native_cpu_train
```

To build only the OpenCL target, set
`IDA_AMD_TARGET=ida_native_opencl_train`. The default script target builds all
enabled AMD backends; when OpenCL is disabled it still builds the CPU reference
target.

## Request boundary

An OpenCL request must explicitly select the portable runtime and must fail
closed on unsupported shapes:

```json
{
  "backend": "native",
  "device": {
    "runtime": "opencl",
    "required_arch": "gfx1036",
    "precision": "fp32"
  },
  "precision_profile": "opencl_fp32",
  "optimizer_type": "lion",
  "attention_backend": "scalar_flash",
  "model": {
    "hidden_size": 32,
    "intermediate_size": 128,
    "layers": 1,
    "heads": 1,
    "vocab_size": 256
  },
  "training": {
    "microbatch": 1,
    "grad_accumulation": 1,
    "learning_rate": 0.0003,
    "max_steps": 3
  },
  "input": {
    "token_blocks": "/path/to/tokens.u32",
    "label_blocks": "/path/to/labels.i32",
    "batch_size": 1,
    "sequence_length": 2048
  },
  "status_file": "/path/to/run-output/status.json",
  "output_dir": "/path/to/run-output"
}
```

The runtime metadata is request-scoped. It is never inferred from the CUDA
worker environment and never silently falls back to CUDA.
