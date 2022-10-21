cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
include(CheckLanguage)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

project(propsim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

set(EXEC ${CMAKE_PROJECT_NAME}_exec)

add_executable(${EXEC} ${SOURCES})

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA is OK")

        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 14)
        file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS src/*.cu)
        target_sources(${EXEC} PRIVATE ${CUDA_SOURCES})
        set_property(TARGET ${EXEC} PROPERTY CUDA_ARCHITECTURES 52)
        set_target_properties(${EXEC} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_compile_definitions(${EXEC} PRIVATE CUDA_AVAL)
else()
        message(STATUS "No CUDA")
endif()

include(FetchContent)

#if(MSVC)
  #FetchContent_Declare(
    #nana
    #GIT_REPOSITORY https://github.com/cnjinhao/nana.git
    #GIT_TAG        develop-1.8
  #)
#else()
FetchContent_Declare(
    nana
    GIT_REPOSITORY https://github.com/cnjinhao/nana.git
    GIT_TAG        master
)
#endif()

FetchContent_GetProperties(nana)
if(NOT nana_POPULATED)
  FetchContent_Populate(nana)
  add_subdirectory(${nana_SOURCE_DIR} ${nana_BINARY_DIR})
endif()

target_link_libraries(${EXEC}
	nana
)
