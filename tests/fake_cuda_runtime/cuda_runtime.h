#pragma once

// Fixture-only CUDA declarations: status writing is exercised without a GPU.
using cudaError_t = int;
inline constexpr cudaError_t cudaSuccess = 0;
enum cudaDeviceAttr { cudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMinor };
inline cudaError_t cudaGetDevice(int*) { return 1; }
inline cudaError_t cudaDeviceGetAttribute(int*, cudaDeviceAttr, int) { return 1; }
