#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <CL/cl.h>

#include "ida_native/opencl_runtime.hpp"

namespace ida_native::opencl {

// A small, dependency-free RAII layer over the Khronos C API. The official
// C++ bindings are optional; this wrapper keeps the AMD backend buildable with
// only the standard OpenCL headers and ICD loader.
template <typename Handle, cl_int (*ReleaseFunction)(Handle)>
class UniqueHandle {
public:
    UniqueHandle() = default;
    explicit UniqueHandle(Handle handle) : handle_(handle) {}
    ~UniqueHandle() { reset(); }

    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;

    UniqueHandle(UniqueHandle&& other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)) {}

    UniqueHandle& operator=(UniqueHandle&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    [[nodiscard]] Handle get() const noexcept { return handle_; }
    [[nodiscard]] explicit operator bool() const noexcept { return handle_ != nullptr; }

    [[nodiscard]] Handle release() noexcept {
        return std::exchange(handle_, nullptr);
    }

    void reset(Handle replacement = nullptr) noexcept {
        if (handle_ != nullptr) {
            (void)ReleaseFunction(handle_);
        }
        handle_ = replacement;
    }

private:
    Handle handle_{nullptr};
};

using Context = UniqueHandle<cl_context, &clReleaseContext>;
using CommandQueue = UniqueHandle<cl_command_queue, &clReleaseCommandQueue>;
using Program = UniqueHandle<cl_program, &clReleaseProgram>;
using Kernel = UniqueHandle<cl_kernel, &clReleaseKernel>;
using Buffer = UniqueHandle<cl_mem, &clReleaseMemObject>;
using Event = UniqueHandle<cl_event, &clReleaseEvent>;

Context create_context(cl_device_id device);
CommandQueue create_command_queue(cl_context context, cl_device_id device);
Program build_program(
    cl_context context,
    cl_device_id device,
    const std::string& source,
    const std::string& options = "-cl-std=CL1.2"
);
Kernel create_kernel(cl_program program, const char* name);
Buffer create_buffer(
    cl_context context,
    cl_mem_flags flags,
    std::size_t bytes,
    void* host_pointer = nullptr
);

void write_buffer(
    cl_command_queue queue,
    cl_mem destination,
    const void* source,
    std::size_t bytes
);

void read_buffer(
    cl_command_queue queue,
    cl_mem source,
    void* destination,
    std::size_t bytes
);

void enqueue_1d(cl_command_queue queue, cl_kernel kernel, std::size_t global_size);
void finish(cl_command_queue queue);

template <typename T>
void set_kernel_arg(cl_kernel kernel, cl_uint index, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    check_opencl(
        clSetKernelArg(kernel, index, sizeof(T), &value),
        "clSetKernelArg(value)");
}

inline void set_kernel_arg(cl_kernel kernel, cl_uint index, cl_mem value) {
    check_opencl(
        clSetKernelArg(kernel, index, sizeof(value), &value),
        "clSetKernelArg(buffer)");
}

struct VectorAddCheckResult {
    bool passed{false};
    float maximum_error{0.0f};
    std::size_t elements{0};
};

// Exercises the complete host-side lifecycle: program creation, kernel
// arguments, device buffers, queue submission, synchronization, and readback.
VectorAddCheckResult run_vector_add_check(const OpenCLRuntime& runtime);

}  // namespace ida_native::opencl
