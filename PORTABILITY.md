# Native portability roadmap

The original `ida_native_train` target remains the CUDA implementation. The
portable `ida_native_opencl_train` target is a separate FP32 reference path for
the AMD Radeon `gfx1036` device, and `ida_native_cpu_train` is the corresponding
host execution path using AMD AVX2/FMA when available.

## Current support matrix

| Capability | CUDA native | AMD OpenCL smoke | Native CPU smoke |
| --- | --- | --- | --- |
| Dense embedding, RMSNorm, scalar causal attention, SwiGLU | Yes | Yes | Yes |
| Cross-entropy and analytical backward pass | Yes | Yes | Yes |
| Adam / AdamW | No | Disabled | Disabled |
| Gradient accumulation | Yes | Yes | Yes |
| FP32 | Yes | Yes | Yes |
| AMD AVX2/FMA host kernels | No | No | Yes, with scalar fallback |
| BF16, FP8, FP4 | Yes | No | No |
| Lion | Yes | Disabled | Disabled |
| Safetensors checkpoint/resume | Yes | No | No |
| MoE, LRSS, PSS | Yes | No | No |
| WGMMA/Blackwell kernels | NVIDIA-only | No | No |
| Model parallelism | CUDA-only | No | No |

The OpenCL smoke source supports one dense layer, one attention head, batch
size one, a 2,048-token sequence, and FP32, but its public profile is disabled.
Its only implemented update path is AdamW, which is no longer admitted. It
must not be launched until a Lion path is implemented and re-enabled in the
catalog.

The CPU smoke source uses the same tensor shapes and binary dataset contract,
but its public profile is also disabled because it only implements AdamW. Its
CPU kernel source is `kernels/cpu_amd_smoke.cpp`; it contains an AMD-friendly
AVX2/FMA dot product and a scalar fallback selected at runtime.

## Build and run

With the Windows OpenCL development headers/import library and a C++20 build
toolchain installed:

```powershell
cmake -S . -B build -DCMAKE_PREFIX_PATH=C:\OpenCLSDK -DIDA_NATIVE_ENABLE_CUDA=OFF
cmake --build build --config Release --target ida_native_opencl_train

bin\ida_native_opencl_train.exe `
  --request-json C:\path\to\opencl-request.json `
  --device 0
```

For the native CPU target, CUDA and OpenCL development files are not needed:

```powershell
cmake -S . -B build-cpu `
  -DIDA_NATIVE_AMD_ENABLE_CPU=ON `
  -DIDA_NATIVE_ENABLE_CUDA=OFF `
  -DIDA_NATIVE_AMD_ENABLE_OPENCL=OFF
cmake --build build-cpu --config Release --target ida_native_cpu_train

bin\ida_native_cpu_train.exe `
  --request-json C:\path\to\cpu-request.json
```

On Windows there are three practical build routes:

* Visual Studio's CMake generator, which produces an MSBuild solution and was
  used for local verification.
* Ninja, when run from a Visual Studio Developer PowerShell, using
  `cmake -S . -B build-ninja -G Ninja -DCMAKE_PREFIX_PATH=C:\OpenCLSDK
  -DIDA_NATIVE_ENABLE_CUDA=OFF` followed by `cmake --build build-ninja`.

The dataset must contain exactly one 2,048-token `tokens.u32` block and one
matching `labels.i32` block. The run writes metrics only; checkpoint/resume is
deferred until the portable checkpoint contract is implemented.

## Future phases

1. Probe and optionally support an OpenCL CPU device when the installed AMD
   driver exposes one; this remains separate from the native C++ CPU backend.
2. Add HIP/ROCm support for officially supported AMD GPUs using hipBLAS and
   hipBLASLt where available. The current `gfx1036` integrated GPU is not the
   validated HIP target.
3. Port optimizer variants, safetensors persistence/resume, BF16, and larger
   dense bodies.
4. Port MoE, LRSS/PSS, model parallelism, and architecture-specific kernels
   behind explicit capability checks.

CUDA-only precision profiles must continue to fail clearly on portable
backends; they must never silently fall back to a different numerical path.
