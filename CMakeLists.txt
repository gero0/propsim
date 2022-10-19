cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

project(propsim LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp src/*.cu)

set(EXEC ${CMAKE_PROJECT_NAME}_exec)

add_executable(${EXEC} ${SOURCES})
set_property(TARGET ${EXEC} PROPERTY CUDA_ARCHITECTURES 52)
set_target_properties(${EXEC} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(libs/wxWidgets)
target_link_libraries(${EXEC} wx::net wx::core wx::base)
