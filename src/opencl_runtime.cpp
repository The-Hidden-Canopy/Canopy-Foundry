#include "ida_native/opencl_runtime.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ida_native {

namespace {

std::string info_string(cl_platform_id platform, cl_platform_info key) {
    std::size_t size = 0;
    check_opencl(clGetPlatformInfo(platform, key, 0, nullptr, &size),
                 "clGetPlatformInfo(size)");
    std::string value(size, '\0');
    check_opencl(clGetPlatformInfo(platform, key, size, value.data(), nullptr),
                 "clGetPlatformInfo(value)");
    if (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

std::string info_string(cl_device_id device, cl_device_info key) {
    std::size_t size = 0;
    check_opencl(clGetDeviceInfo(device, key, 0, nullptr, &size),
                 "clGetDeviceInfo(size)");
    std::string value(size, '\0');
    check_opencl(clGetDeviceInfo(device, key, size, value.data(), nullptr),
                 "clGetDeviceInfo(value)");
    if (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

}  // namespace

std::string opencl_error_name(cl_int status) {
    switch (status) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    default: return "OpenCL error " + std::to_string(status);
    }
}

void check_opencl(cl_int status, const char* expression) {
    if (status != CL_SUCCESS) {
        throw std::runtime_error(std::string(expression) + " failed: " +
                                 opencl_error_name(status));
    }
}

OpenCLRuntime create_opencl_runtime(int device_index) {
    if (device_index < 0) throw std::runtime_error("OpenCL device index must be nonnegative");

    cl_uint platform_count = 0;
    check_opencl(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
    if (platform_count == 0) throw std::runtime_error("no OpenCL platforms are available");

    std::vector<cl_platform_id> platforms(platform_count);
    check_opencl(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
                 "clGetPlatformIDs(values)");

    std::vector<cl_device_id> devices;
    cl_platform_id selected_platform = nullptr;
    for (cl_platform_id platform : platforms) {
        cl_uint count = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count) != CL_SUCCESS ||
            count == 0) {
            continue;
        }
        std::vector<cl_device_id> found(count);
        check_opencl(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, found.data(), nullptr),
                     "clGetDeviceIDs(values)");
        selected_platform = platform;
        devices = std::move(found);
        break;
    }
    if (devices.empty()) throw std::runtime_error("no OpenCL GPU devices are available");
    if (static_cast<std::size_t>(device_index) >= devices.size()) {
        throw std::runtime_error("OpenCL device index " + std::to_string(device_index) +
                                 " is out of range; available GPU devices: " +
                                 std::to_string(devices.size()));
    }

    OpenCLRuntime runtime{};
    runtime.platform = selected_platform;
    runtime.device = devices[static_cast<std::size_t>(device_index)];
    runtime.platform_name = info_string(runtime.platform, CL_PLATFORM_NAME);
    runtime.device_name = info_string(runtime.device, CL_DEVICE_NAME);
    runtime.device_version = info_string(runtime.device, CL_DEVICE_VERSION);
    runtime.driver_version = info_string(runtime.device, CL_DRIVER_VERSION);
    check_opencl(clGetDeviceInfo(runtime.device, CL_DEVICE_MAX_COMPUTE_UNITS,
                                 sizeof(runtime.compute_units), &runtime.compute_units, nullptr),
                 "clGetDeviceInfo(compute units)");
    check_opencl(clGetDeviceInfo(runtime.device, CL_DEVICE_GLOBAL_MEM_SIZE,
                                 sizeof(runtime.global_memory_bytes), &runtime.global_memory_bytes,
                                 nullptr),
                 "clGetDeviceInfo(global memory)");

    cl_int status = CL_SUCCESS;
    runtime.context = clCreateContext(nullptr, 1, &runtime.device, nullptr, nullptr, &status);
    check_opencl(status, "clCreateContext");
    runtime.queue = clCreateCommandQueue(runtime.context, runtime.device, 0, &status);
    check_opencl(status, "clCreateCommandQueue");
    return runtime;
}

void destroy_opencl_runtime(OpenCLRuntime& runtime) {
    if (runtime.queue != nullptr) {
        clReleaseCommandQueue(runtime.queue);
        runtime.queue = nullptr;
    }
    if (runtime.context != nullptr) {
        clReleaseContext(runtime.context);
        runtime.context = nullptr;
    }
    runtime.device = nullptr;
    runtime.platform = nullptr;
}

cl_program build_opencl_program(
    const OpenCLRuntime& runtime,
    const std::filesystem::path& source_path
) {
    std::ifstream input(source_path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to open OpenCL kernel source: " + source_path.string());
    std::ostringstream contents;
    contents << input.rdbuf();
    const std::string source = contents.str();
    const char* source_ptr = source.data();
    const std::size_t source_size = source.size();

    cl_int status = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(
        runtime.context, 1, &source_ptr, &source_size, &status);
    check_opencl(status, "clCreateProgramWithSource");

    status = clBuildProgram(program, 1, &runtime.device, "-cl-std=CL2.0", nullptr, nullptr);
    if (status != CL_SUCCESS) {
        std::size_t log_size = 0;
        clGetProgramBuildInfo(program, runtime.device, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_size);
        std::string log(log_size, '\0');
        if (log_size != 0) {
            clGetProgramBuildInfo(program, runtime.device, CL_PROGRAM_BUILD_LOG,
                                  log_size, log.data(), nullptr);
        }
        clReleaseProgram(program);
        throw std::runtime_error("OpenCL kernel build failed: " + opencl_error_name(status) +
                                 "\n" + log);
    }
    return program;
}

}  // namespace ida_native
