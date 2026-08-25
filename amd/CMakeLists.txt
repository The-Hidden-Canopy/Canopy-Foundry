cmake_minimum_required(VERSION 3.28)

project(ida_native_amd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(IDA_NATIVE_AMD_ENABLE_OPENCL
  "Build the AMD OpenCL portability backend"
  ON)
option(IDA_NATIVE_AMD_ENABLE_CPU
  "Build the host CPU reference backend"
  ON)

if(NOT TARGET nlohmann_json::nlohmann_json)
  find_package(nlohmann_json 3.11 CONFIG REQUIRED)
endif()

if(IDA_NATIVE_AMD_ENABLE_OPENCL)
  find_package(OpenCL QUIET)
  if(OpenCL_FOUND AND TARGET OpenCL::OpenCL)
    add_executable(ida_native_opencl_train
      src/main_opencl.cpp
      src/opencl_cpp.cpp
      src/opencl_runtime.cpp
      src/opencl_trainer.cpp
      ../src/request.cpp
    )

    target_include_directories(ida_native_opencl_train PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${IDA_NATIVE_PRIVATE_AWARE_INCLUDE_DIRS}
    )

    target_compile_definitions(ida_native_opencl_train PRIVATE
      CL_TARGET_OPENCL_VERSION=200
    )

    target_link_libraries(ida_native_opencl_train PRIVATE
      OpenCL::OpenCL
      nlohmann_json::nlohmann_json
    )

    set_target_properties(ida_native_opencl_train PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
      RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
      RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
      RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
      RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
    )

    add_custom_command(TARGET ida_native_opencl_train POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/opencl_smoke.cl
        $<TARGET_FILE_DIR:ida_native_opencl_train>/opencl_smoke.cl
      VERBATIM
    )
  else()
    message(WARNING
      "OpenCL development files were not found; skipping ida_native_opencl_train")
  endif()
endif()

if(IDA_NATIVE_AMD_ENABLE_CPU)
  find_package(Threads REQUIRED)

  add_executable(ida_native_cpu_train
    src/main_cpu.cpp
    src/cpu_trainer.cpp
    kernels/cpu_amd_smoke.cpp
    ../src/request.cpp
  )

  target_include_directories(ida_native_cpu_train PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${IDA_NATIVE_PRIVATE_AWARE_INCLUDE_DIRS}
  )

  target_link_libraries(ida_native_cpu_train PRIVATE
    Threads::Threads
    nlohmann_json::nlohmann_json
  )

  if(MSVC)
    set_source_files_properties(kernels/cpu_amd_smoke.cpp PROPERTIES
      COMPILE_OPTIONS "/arch:AVX2"
    )
  endif()

  set_target_properties(ida_native_cpu_train PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
  )
endif()

if(NOT TARGET ida_native_opencl_train AND NOT TARGET ida_native_cpu_train)
  message(FATAL_ERROR
    "No AMD backend enabled or available: enable OpenCL and/or CPU")
endif()

add_custom_target(ida_native_amd_train)
if(TARGET ida_native_opencl_train)
  add_dependencies(ida_native_amd_train ida_native_opencl_train)
endif()
if(TARGET ida_native_cpu_train)
  add_dependencies(ida_native_amd_train ida_native_cpu_train)
endif()
