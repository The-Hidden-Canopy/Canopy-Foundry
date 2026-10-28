add_library(foundry_shell_core STATIC
  src/model.cpp
  src/sprite.cpp
  src/simulation.cpp
  src/plan_service.cpp
  src/guide.cpp
)
target_compile_features(foundry_shell_core PUBLIC cxx_std_20)
target_include_directories(foundry_shell_core
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(foundry_shell_core
  PUBLIC
    nlohmann_json::nlohmann_json
    foundry_execution_service
)

add_executable(foundry_shell
  src/main.cpp
)
target_compile_features(foundry_shell PRIVATE cxx_std_20)
target_link_libraries(foundry_shell PRIVATE foundry_shell_core)
target_include_directories(foundry_shell PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(WIN32)
  target_sources(foundry_shell PRIVATE src/platform_win32.cpp)
  target_compile_definitions(foundry_shell PRIVATE CANOPY_FOUNDRY_SHELL_WIN32=1)
  target_link_libraries(foundry_shell PRIVATE
    user32
    gdi32
    comdlg32
    shell32
  )
  set_target_properties(foundry_shell PROPERTIES WIN32_EXECUTABLE TRUE)
elseif(UNIX AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CANOPY_FOUNDRY_ENABLE_X11_SHELL)
  find_package(X11 QUIET)
  if(X11_FOUND)
    target_sources(foundry_shell PRIVATE src/platform_x11.cpp)
    target_compile_definitions(foundry_shell PRIVATE CANOPY_FOUNDRY_SHELL_X11=1)
    target_include_directories(foundry_shell PRIVATE ${X11_INCLUDE_DIR})
    find_package(Threads REQUIRED)
    target_link_libraries(foundry_shell PRIVATE ${X11_LIBRARIES} Threads::Threads)
  else()
    message(WARNING
      "X11 development files were not found; foundry_shell will provide "
      "headless diagnostics only on this Linux build")
    target_sources(foundry_shell PRIVATE src/platform_stub.cpp)
  endif()
else()
  target_sources(foundry_shell PRIVATE src/platform_stub.cpp)
endif()

set_target_properties(foundry_shell PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

add_executable(foundry_shell_model_test
  ../tests/shell_model_test.cpp
  src/model.cpp
  src/sprite.cpp
  src/simulation.cpp
)
target_compile_features(foundry_shell_model_test PRIVATE cxx_std_20)
target_include_directories(foundry_shell_model_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(foundry_shell_model_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME foundry_shell_model COMMAND foundry_shell_model_test)

add_executable(foundry_shell_simulation_test
  ../tests/shell_simulation_test.cpp
  src/simulation.cpp
)
target_compile_features(foundry_shell_simulation_test PRIVATE cxx_std_20)
target_include_directories(foundry_shell_simulation_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME foundry_shell_simulation COMMAND foundry_shell_simulation_test)

add_executable(foundry_shell_guide_test ../tests/shell_guide_test.cpp src/guide.cpp)
target_compile_features(foundry_shell_guide_test PRIVATE cxx_std_20)
target_include_directories(foundry_shell_guide_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME foundry_shell_guide COMMAND foundry_shell_guide_test
  ${CMAKE_CURRENT_BINARY_DIR}/shell-guide-test-fixture)

add_executable(foundry_checkpoint_inspection_test
  ../tests/checkpoint_inspection_test.cpp ../src/execution/checkpoint_inspection.cpp)
target_compile_features(foundry_checkpoint_inspection_test PRIVATE cxx_std_20)
target_include_directories(foundry_checkpoint_inspection_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME foundry_checkpoint_inspection COMMAND foundry_checkpoint_inspection_test
  ${CMAKE_CURRENT_BINARY_DIR}/checkpoint-inspection-test-fixture)

add_executable(foundry_execution_service_integration_test
  ../tests/execution_service_integration_test.cpp)
target_compile_features(foundry_execution_service_integration_test PRIVATE cxx_std_20)
target_link_libraries(foundry_execution_service_integration_test PRIVATE foundry_shell_core)
add_test(NAME foundry_execution_service_integration
  COMMAND foundry_execution_service_integration_test ${CMAKE_SOURCE_DIR})

add_executable(foundry_shell_plan_service_contract_test
  ../tests/shell_plan_service_contract_test.cpp
  ../src/execution/service.cpp
  ../src/execution/checkpoint_inspection.cpp
  src/plan_service.cpp
)
target_compile_features(foundry_shell_plan_service_contract_test PRIVATE cxx_std_20)
target_include_directories(foundry_shell_plan_service_contract_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME foundry_shell_plan_service_contract COMMAND foundry_shell_plan_service_contract_test)

if(WIN32)
  add_executable(foundry_owner_pipe_fixture_test
    ../tests/owner_pipe_fixture_test.cpp
    ../src/execution/service.cpp
    ../src/execution/owner_pipe.cpp)
  target_compile_features(foundry_owner_pipe_fixture_test PRIVATE cxx_std_20)
  target_include_directories(foundry_owner_pipe_fixture_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(foundry_owner_pipe_fixture_test PRIVATE nlohmann_json::nlohmann_json advapi32 bcrypt)
  if(MINGW)
    target_link_options(foundry_owner_pipe_fixture_test PRIVATE -municode)
  endif()
  add_test(NAME foundry_owner_pipe_fixture COMMAND foundry_owner_pipe_fixture_test
    ${CMAKE_CURRENT_BINARY_DIR}/owner-pipe-fixture)
endif()

add_executable(foundry_shell_mode_test ../tests/shell_mode_test.cpp)
target_compile_features(foundry_shell_mode_test PRIVATE cxx_std_20)
target_include_directories(foundry_shell_mode_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME foundry_shell_mode COMMAND foundry_shell_mode_test)

install(TARGETS foundry_shell RUNTIME DESTINATION bin)
install(FILES ${CMAKE_SOURCE_DIR}/STARTER_GUIDE.md ${CMAKE_SOURCE_DIR}/README.md
  ${CMAKE_SOURCE_DIR}/PORTABILITY.md ${CMAKE_SOURCE_DIR}/CMakePresets.json
  DESTINATION share/canopy-foundry)
install(FILES ${CMAKE_SOURCE_DIR}/docs/neural-foundry-guide.md
  ${CMAKE_SOURCE_DIR}/docs/learning-path.md ${CMAKE_SOURCE_DIR}/docs/python-wrapper-contract.md
  ${CMAKE_SOURCE_DIR}/docs/public-boundary.md ${CMAKE_SOURCE_DIR}/docs/private-runtime-boundary.md
  DESTINATION share/canopy-foundry/docs)
install(FILES ${CMAKE_SOURCE_DIR}/api/README.md DESTINATION share/canopy-foundry/api)
install(FILES ${CMAKE_SOURCE_DIR}/configs/public/capabilities.json
  DESTINATION share/canopy-foundry/configs/public)
install(FILES ${CMAKE_SOURCE_DIR}/configs/examples/planning_only_request.json
  DESTINATION share/canopy-foundry/configs/examples)
install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/frontend/
  DESTINATION share/canopy-foundry/frontend
  FILES_MATCHING
    PATTERN "*.html"
    PATTERN "*.css"
    PATTERN "*.js"
    PATTERN "*.md"
    PATTERN "*.json"
)
