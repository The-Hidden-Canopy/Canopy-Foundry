#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include <CL/cl.h>

namespace ida_native {

struct OpenCLRuntime {
    cl_platform_id platform{nullptr};
    cl_device_id device{nullptr};
    cl_context context{nullptr};
    cl_command_queue queue{nullptr};
    std::string platform_name;
    std::string device_name;
    std::string device_version;
    std::string driver_version;
    cl_uint compute_units{0};
    cl_ulong global_memory_bytes{0};
    std::size_t max_work_group_size{0};
};

void check_opencl(cl_int status, const char* expression);

OpenCLRuntime create_opencl_runtime(int device_index);
void destroy_opencl_runtime(OpenCLRuntime& runtime);

cl_program build_opencl_program(
    const OpenCLRuntime& runtime,
    const std::filesystem::path& source_path
);

std::string opencl_error_name(cl_int status);

}  // namespace ida_native
