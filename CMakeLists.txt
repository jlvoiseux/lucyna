cmake_minimum_required(VERSION 3.18)
project(lucyna CUDA C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

file(GLOB BASE_SOURCES "src/ly*.cu")
file(GLOB TEST_SOURCES "test/ly*.cu")

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/test)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Main executable
add_executable(lucyna ${BASE_SOURCES} "src/main.cu")

set_target_properties(lucyna PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_SOURCE_PROPERTY_FORMAT C
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin"
)

if (MSVC)
    set_target_properties(lucyna PROPERTIES
            LINK_FLAGS "/NODEFAULTLIB:LIBCMT"
    )
endif ()

include(FetchContent)
FetchContent_Declare(
        unity
        GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
        GIT_TAG v2.6.1
)
FetchContent_MakeAvailable(unity)

add_executable(lucyna_test ${BASE_SOURCES} ${TEST_SOURCES})

# Add a dummy kernel to ensure proper CUDA linking
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy.cu
        "extern \"C\" __global__ void dummy_kernel() {}")
target_sources(lucyna_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/dummy.cu)

set_target_properties(lucyna_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_SOURCE_PROPERTY_FORMAT C
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

set_target_properties(lucyna_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_SOURCE_PROPERTY_FORMAT C
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin"
)

target_include_directories(lucyna_test PRIVATE ${unity_SOURCE_DIR}/src)
target_sources(lucyna_test PRIVATE
        ${unity_SOURCE_DIR}/src/unity.c
)

if (MSVC)
    set_target_properties(lucyna_test PROPERTIES
            LINK_FLAGS "/NODEFAULTLIB:LIBCMT"
    )
endif ()