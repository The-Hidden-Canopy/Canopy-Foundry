cmake_minimum_required(VERSION 3.28)

project(ida_native_train LANGUAGES CXX)

include(CheckLanguage)
find_package(nlohmann_json 3.11 CONFIG REQUIRED)

option(IDA_NATIVE_ENABLE_CUDA
  "Build the CUDA trainer when a CUDA compiler is available"
  ON)
option(IDA_NATIVE_ENABLE_AMD
  "Build the native AMD portability backends"
  ON)
option(IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY
  "Enable private ontology, GEMM, pack, and analytics observability"
  OFF)

# Adam/AdamW is not part of the public/native execution graph. The legacy
# implementation is not part of the public source tree or any released target.
add_compile_definitions(IDA_NATIVE_ENABLE_ADAMW=0)

if(IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY)
  add_compile_definitions(IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY=1)
  set(IDA_NATIVE_PRIVATE_AWARE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include-public)
else()
  add_compile_definitions(IDA_NATIVE_ENABLE_PRIVATE_OBSERVABILITY=0)
  set(IDA_NATIVE_PRIVATE_AWARE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include-public
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()
include_directories(BEFORE ${IDA_NATIVE_PRIVATE_AWARE_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

add_executable(ida_native_request_governance_test
  tests/request_governance_test.cpp
  src/request.cpp
)
target_include_directories(ida_native_request_governance_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(ida_native_request_governance_test PRIVATE
  nlohmann_json::nlohmann_json
)
add_test(NAME ida_native_request_governance
  COMMAND ida_native_request_governance_test)

set(IDA_NATIVE_CUDA_ARCHITECTURES "90a"
  CACHE STRING
  "CUDA architecture target. Set IDA_NATIVE_CUDA_ARCHITECTURES to match the active GPU (86=RTX 3050, 90a=H100). The public target uses the scalar attention contract; deployment-owned accelerated packages are built separately.")

# CUDA is optional so the Canopy Foundry tree can build the AMD portability runtime
# on hosts that have no nvcc.  The default remains unchanged when CUDA is
# installed: the public CUDA trainer and public compatibility probes are
# enabled.
set(IDA_NATIVE_CUDA_FOUND FALSE)
if(IDA_NATIVE_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(IDA_NATIVE_CUDA_FOUND TRUE)
  else()
    message(WARNING
      "CUDA compiler not found; skipping the CUDA trainer and building enabled portability backends")
  endif()
endif()

if(IDA_NATIVE_CUDA_FOUND)
add_executable(ida_native_train
  src/main.cpp
  src/request.cpp
  src/status.cpp
  src/arena.cu
  src/trainer.cu
  src/checkpoint.cpp
  src/telemetry.cpp
  kernels/embedding.cu
  kernels/layernorm.cu
  kernels/rmsnorm.cu
  kernels/cross_entropy.cu
  kernels/lion.cu
  kernels/attention.cu
  kernels/private_backend_stubs.cu
  kernels/fp8.cu
  kernels/lrss.cu
  kernels/moe.cu
)

target_include_directories(ida_native_train PRIVATE
  ${IDA_NATIVE_PRIVATE_AWARE_INCLUDE_DIRS}
)

target_link_libraries(ida_native_train PRIVATE
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
  CUDA::cuda_driver
  nlohmann_json::nlohmann_json
)

set_target_properties(ida_native_train PROPERTIES
  CUDA_ARCHITECTURES "${IDA_NATIVE_CUDA_ARCHITECTURES}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

# Standalone CUDA-runtime readiness probe for start_mps_from_hardware_envelope.sh.
# Plain runtime-API calls only — no kernels, no CUDA-language compile needed.
add_executable(cuda_wait_ready tools/cuda_wait_ready.cpp)

target_link_libraries(cuda_wait_ready PRIVATE
  CUDA::cudart
)

set_target_properties(cuda_wait_ready PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

# RoPE (rotary position embedding) correctness probe -- unlike the
# Blackwell-only probes above, this validates trainer.cu's k_rope_apply_bf16
# (2026-08-13 standard-architecture portability port), which has no
# arch-specific instructions and needs to be correct on whatever
# IDA_NATIVE_CUDA_ARCHITECTURES the rest of the project is building for, not
# pinned to 120a. Deliberately EXCLUDE_FROM_ALL like the others: it is
# evidence to run on demand, not part of the production training binary.
add_executable(rope_probe EXCLUDE_FROM_ALL
  probes/rope_probe.cu
)

target_link_libraries(rope_probe PRIVATE
  CUDA::cudart
)

set_target_properties(rope_probe PROPERTIES
  CUDA_ARCHITECTURES "${IDA_NATIVE_CUDA_ARCHITECTURES}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

# ── Cross-backend numerical parity harness (Phase 4, 2026-07-22) ────────────
#
# Previously a fully manual tool (hand-run nvcc against the main target's
# already-built object files, per the header comment this replaces) — now a
# real CMake/ctest target. Compiles its own copy of kernels/moe.cu directly
# (every function the check calls -- moe_pool_by_sample, moe_topk_route_f32,
# gelu_forward, etc. -- is a real, public, non-file-static function in that
# file; only the ORCHESTRATION sequence in moe_router_parity_check.cu itself
# is duplicated from trainer.cu's file-static moe_router_forward/
# moe_expert_bank_forward, not any kernel math). Deliberately does NOT drop
# `static` linkage on those two trainer.cu functions to expose them directly
# -- that would require hoisting StepBuffers/Fp8Ctx (both large, internal to
# trainer.cu, referenced throughout that file) into a shared header, a much
# larger and riskier refactor than this harness's real value justifies right
# now. If moe_router_forward's/moe_expert_bank_forward's sequence changes,
# the duplicated sequence in moe_router_parity_check.cu must be updated to
# match by hand -- same caveat the file's own header comment already states.
enable_testing()

add_executable(moe_router_parity_check
  tests/moe_router_parity_check.cu
  kernels/moe.cu
  # moe.cu's own probe/quantization paths (unused by this specific check's
  # call sequence, but still referenced at link time as one compilation
  # unit) pull in cross_entropy.cu (softmax_inplace_f32) and fp8.cu
  # (fp8_amax_bf16) -- linking the same kernel set the main target uses
  # avoids chasing individual missing symbols one at a time.
  kernels/cross_entropy.cu
  kernels/fp8.cu
  kernels/embedding.cu
  kernels/rmsnorm.cu
  kernels/lion.cu
  kernels/attention.cu
  kernels/private_backend_stubs.cu
  kernels/lrss.cu
)

target_include_directories(moe_router_parity_check PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(moe_router_parity_check PRIVATE
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
  nlohmann_json::nlohmann_json
)

set_target_properties(moe_router_parity_check PROPERTIES
  CUDA_ARCHITECTURES "${IDA_NATIVE_CUDA_ARCHITECTURES}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

# Raw-byte E4M3 pack/unpack probe. It is architecture-independent storage
# evidence, not a native-FP8 GEMM claim; build it for sm_86 on the 3050.
add_executable(fp8_e4m3_probe EXCLUDE_FROM_ALL
  tests/fp8_e4m3_probe.cu
)
target_include_directories(fp8_e4m3_probe PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(fp8_e4m3_probe PRIVATE CUDA::cudart)
set_target_properties(fp8_e4m3_probe PROPERTIES
  CUDA_ARCHITECTURES "${IDA_NATIVE_CUDA_ARCHITECTURES}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

# Ampere fused decode-to-BF16 GEMM evidence probe.  It is intentionally not
# part of ida_native_train: the trainer keeps the validated BF16 cuBLAS path
# until this tile-local E4M3 decode is proven correct and faster on sm_86.
add_executable(ampere_fp8_fused_gemm_bench EXCLUDE_FROM_ALL
  benchmarks/ampere_fp8_fused_gemm_bench.cu
)
target_include_directories(ampere_fp8_fused_gemm_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(ampere_fp8_fused_gemm_bench PRIVATE
  CUDA::cudart
  CUDA::cublas
)
set_target_properties(ampere_fp8_fused_gemm_bench PROPERTIES
  CUDA_ARCHITECTURES "${IDA_NATIVE_CUDA_ARCHITECTURES}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

endif()

if(IDA_NATIVE_ENABLE_AMD)
  add_subdirectory(amd)
endif()

if(NOT IDA_NATIVE_CUDA_FOUND AND NOT IDA_NATIVE_ENABLE_AMD)
  message(FATAL_ERROR
    "No native backend enabled: enable CUDA or IDA_NATIVE_ENABLE_AMD")
endif()
