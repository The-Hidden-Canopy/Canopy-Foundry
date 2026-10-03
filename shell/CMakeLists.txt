add_library(foundry_shell_core STATIC
  src/model.cpp
  src/sprite.cpp
  src/simulation.cpp
)
target_compile_features(foundry_shell_core PUBLIC cxx_std_20)
target_include_directories(foundry_shell_core
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(foundry_shell_core
  PUBLIC
    nlohmann_json::nlohmann_json
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
    target_link_libraries(foundry_shell PRIVATE ${X11_LIBRARIES})
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

install(TARGETS foundry_shell RUNTIME DESTINATION bin)
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
