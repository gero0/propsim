cmake_minimum_required(VERSION 3.16)

project(propsim LANGUAGES CXX)

include(CheckLanguage)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

#windows-specific commands
if (WIN32)
    #QT PATH SETTING:
    set(QT_PATH "C:\\Qt\\6.4.0\\msvc2019_64")
    #Include the path in prefix so cmake can find qt
    set(CMAKE_PREFIX_PATH ${QT_PATH})

    set(CMAKE_CUDA_COMPILER nvcc)

    #FOr MSVC there is two versions of each dll that must be copied to binary folder depending of the build configuration
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        file(COPY "${QT_PATH}\\plugins\\platforms\\qwindowsd.dll" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}\\platforms")
        file(COPY "${QT_PATH}\\bin\\Qt6Cored.dll" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        file(COPY "${QT_PATH}\\bin\\Qt6Widgetsd.dll" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        file(COPY "${QT_PATH}\\bin\\Qt6Guid.dll" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    else()
        file(COPY "${QT_PATH}\\plugins\\platforms\\qwindows.dll" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}\\platforms")
        file(COPY "${QT_PATH}\\bin\\Qt6Core.dll" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        file(COPY "${QT_PATH}\\bin\\Qt6Widgets.dll" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        file(COPY "${QT_PATH}\\bin\\Qt6Gui.dll" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    endif()
endif()

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
        set_property(TARGET ${EXEC} PROPERTY CUDA_ARCHITECTURES 50)
        set_target_properties(${EXEC} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_compile_definitions(${EXEC} PRIVATE CUDA_AVAL)
else()
        message(STATUS "No CUDA")
endif()

target_link_libraries(${EXEC} PRIVATE Qt6::Widgets Qt6::Gui)