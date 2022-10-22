cmake_minimum_required(VERSION 3.16)

include(CheckLanguage)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

project(propsim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

find_package(Qt6 REQUIRED COMPONENTS Widgets Gui)
qt_standard_project_setup()

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp mainwindow.ui)

set(EXEC ${CMAKE_PROJECT_NAME}_exec)

add_executable(${EXEC} ${SOURCES})

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA is OK")

        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS src/*.cu)
        target_sources(${EXEC} PRIVATE ${CUDA_SOURCES})
        set_property(TARGET ${EXEC} PROPERTY CUDA_ARCHITECTURES 52)
        set_target_properties(${EXEC} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_compile_definitions(${EXEC} PRIVATE CUDA_AVAL)
else()
        message(STATUS "No CUDA")
endif()

target_link_libraries(${EXEC} PRIVATE Qt6::Widgets Qt6::Gui)