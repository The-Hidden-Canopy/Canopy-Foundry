#include "ida_native/opencl_cpp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ida_native::opencl {

namespace {

constexpr char kVectorAddSource[] = R"CLC(
__kernel void ida_vector_add(
    __global const float* lhs,
    __global const float* rhs,
    __global float* output
) {
    const size_t index = get_global_id(0);
    output[index] = lhs[index] + rhs[index];
}
)CLC";

std::string build_log(cl_program program, cl_device_id device) {
    std::size_t bytes = 0;
    if (clGetProgramBuildInfo(
            program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &bytes) != CL_SUCCESS ||
        bytes == 0) {
        return {};
    }
    std::string log(bytes, '\0');
    (void)clGetProgramBuildInfo(
        program, device, CL_PROGRAM_BUILD_LOG, bytes, log.data(), nullptr);
    return log;
}

}  // namespace

Context create_context(cl_device_id device) {
    if (device == nullptr) {
        throw std::runtime_error("cannot create an OpenCL context from a null device");
    }
    cl_int status = CL_SUCCESS;
    Context context(clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status));
    check_opencl(status, "clCreateContext");
    return context;
}

CommandQueue create_command_queue(cl_context context, cl_device_id device) {
    if (context == nullptr || device == nullptr) {
        throw std::runtime_error("OpenCL context and device are required for a command queue");
    }

    cl_int status = CL_SUCCESS;
    cl_command_queue queue = nullptr;
#if defined(CL_VERSION_2_0)
    const cl_queue_properties properties[] = {0};
    queue = clCreateCommandQueueWithProperties(context, device, properties, &status);
#else
    queue = clCreateCommandQueue(context, device, 0, &status);
#endif
    check_opencl(status, "clCreateCommandQueue");
    return CommandQueue(queue);
}

Program build_program(
    cl_context context,
    cl_device_id device,
    const std::string& source,
    const std::string& options
) {
    if (context == nullptr || device == nullptr || source.empty()) {
        throw std::runtime_error("OpenCL context, device, and non-empty source are required");
    }

    const char* source_pointer = source.data();
    const std::size_t source_size = source.size();
    cl_int status = CL_SUCCESS;
    Program program(clCreateProgramWithSource(
        context, 1, &source_pointer, &source_size, &status));
    check_opencl(status, "clCreateProgramWithSource");

    status = clBuildProgram(
        program.get(), 1, &device, options.empty() ? nullptr : options.c_str(), nullptr, nullptr);
    if (status != CL_SUCCESS) {
        std::ostringstream message;
        message << "OpenCL program build failed: " << opencl_error_name(status);
        const std::string log = build_log(program.get(), device);
        if (!log.empty()) message << '\n' << log;
        throw std::runtime_error(message.str());
    }
    return program;
}

Kernel create_kernel(cl_program program, const char* name) {
    if (program == nullptr || name == nullptr || name[0] == '\0') {
        throw std::runtime_error("OpenCL program and kernel name are required");
    }
    cl_int status = CL_SUCCESS;
    Kernel kernel(clCreateKernel(program, name, &status));
    check_opencl(status, "clCreateKernel");
    return kernel;
}

Buffer create_buffer(cl_context context, cl_mem_flags flags, std::size_t bytes, void* host_pointer) {
    if (context == nullptr || bytes == 0) {
        throw std::runtime_error("OpenCL buffer requires a context and nonzero size");
    }
    cl_int status = CL_SUCCESS;
    Buffer buffer(clCreateBuffer(context, flags, bytes, host_pointer, &status));
    check_opencl(status, "clCreateBuffer");
    return buffer;
}

void write_buffer(cl_command_queue queue, cl_mem destination, const void* source, std::size_t bytes) {
    if (queue == nullptr || destination == nullptr || source == nullptr || bytes == 0) {
        throw std::runtime_error("invalid OpenCL write-buffer arguments");
    }
    check_opencl(
        clEnqueueWriteBuffer(queue, destination, CL_TRUE, 0, bytes, source, 0, nullptr, nullptr),
        "clEnqueueWriteBuffer");
}

void read_buffer(cl_command_queue queue, cl_mem source, void* destination, std::size_t bytes) {
    if (queue == nullptr || source == nullptr || destination == nullptr || bytes == 0) {
        throw std::runtime_error("invalid OpenCL read-buffer arguments");
    }
    check_opencl(
        clEnqueueReadBuffer(queue, source, CL_TRUE, 0, bytes, destination, 0, nullptr, nullptr),
        "clEnqueueReadBuffer");
}

void enqueue_1d(cl_command_queue queue, cl_kernel kernel, std::size_t global_size) {
    if (queue == nullptr || kernel == nullptr || global_size == 0) {
        throw std::runtime_error("invalid OpenCL 1D dispatch arguments");
    }
    check_opencl(
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr),
        "clEnqueueNDRangeKernel");
}

void finish(cl_command_queue queue) {
    if (queue == nullptr) throw std::runtime_error("cannot finish a null OpenCL queue");
    check_opencl(clFinish(queue), "clFinish");
}

VectorAddCheckResult run_vector_add_check(const OpenCLRuntime& runtime) {
    constexpr std::size_t count = 256;
    std::vector<float> lhs(count);
    std::vector<float> rhs(count);
    std::vector<float> output(count, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t i = 0; i < count; ++i) {
        lhs[i] = static_cast<float>(i) * 0.25f - 3.0f;
        rhs[i] = static_cast<float>(count - i) * 0.5f;
    }

    Context context = create_context(runtime.device);
    CommandQueue queue = create_command_queue(context.get(), runtime.device);
    Program program = build_program(context.get(), runtime.device, kVectorAddSource);
    Kernel kernel = create_kernel(program.get(), "ida_vector_add");
    Buffer lhs_buffer = create_buffer(context.get(), CL_MEM_READ_ONLY, sizeof(float) * count);
    Buffer rhs_buffer = create_buffer(context.get(), CL_MEM_READ_ONLY, sizeof(float) * count);
    Buffer output_buffer = create_buffer(context.get(), CL_MEM_WRITE_ONLY, sizeof(float) * count);

    write_buffer(queue.get(), lhs_buffer.get(), lhs.data(), sizeof(float) * count);
    write_buffer(queue.get(), rhs_buffer.get(), rhs.data(), sizeof(float) * count);
    set_kernel_arg(kernel.get(), 0, lhs_buffer.get());
    set_kernel_arg(kernel.get(), 1, rhs_buffer.get());
    set_kernel_arg(kernel.get(), 2, output_buffer.get());
    enqueue_1d(queue.get(), kernel.get(), count);
    read_buffer(queue.get(), output_buffer.get(), output.data(), sizeof(float) * count);
    finish(queue.get());

    float maximum_error = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        const float expected = lhs[i] + rhs[i];
        maximum_error = std::max(maximum_error, std::fabs(output[i] - expected));
    }
    return {maximum_error <= 1.0e-6f, maximum_error, count};
}

}  // namespace ida_native::opencl
