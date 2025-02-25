cmake_minimum_required(VERSION 3.18)
project(lucyna C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

file(GLOB BASE_SOURCES "src/ly*.c")

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/test)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

include(FetchContent)
FetchContent_Declare(
        pcre
        GIT_REPOSITORY https://github.com/PCRE2Project/pcre2.git
        GIT_TAG pcre2-10.44
)
FetchContent_MakeAvailable(pcre)
include_directories(${pcre_BINARY_DIR})

add_library(lucyna_lib STATIC ${BASE_SOURCES})
target_link_libraries(lucyna_lib pcre2-8 ${OpenCL_LIBRARIES})
target_compile_definitions(lucyna_lib PUBLIC CL_TARGET_OPENCL_VERSION=300)

add_executable(lucyna "src/main.c")
target_link_libraries(lucyna lucyna_lib)

set_target_properties(lucyna PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin"
)

if (MSVC)
    set_target_properties(lucyna PROPERTIES
            LINK_FLAGS "/NODEFAULTLIB:LIBCMT"
    )
endif ()

FetchContent_Declare(
        unity
        GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
        GIT_TAG v2.6.1
)
FetchContent_MakeAvailable(unity)

function(configure_test_target TARGET_NAME)
    target_link_libraries(${TARGET_NAME} lucyna_lib)

    set_target_properties(${TARGET_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin"
    )

    target_include_directories(${TARGET_NAME} PRIVATE ${unity_SOURCE_DIR}/src)
    target_sources(${TARGET_NAME} PRIVATE ${unity_SOURCE_DIR}/src/unity.c)

    if (MSVC)
        set_target_properties(${TARGET_NAME} PROPERTIES
                LINK_FLAGS "/NODEFAULTLIB:LIBCMT"
        )
    endif ()
endfunction()

add_executable(lucyna_test_tokenizer "test/lyTokenizerTest.c")
add_executable(lucyna_test_transformer "test/lyTransformerTest.c")
add_executable(lucyna_test_tensor "test/lyTensorTest.c")
add_executable(lucyna_test_tensor_math "test/lyTensorMathTest.c")
add_executable(lucyna_test_attention "test/lyAttentionTest.c")
add_executable(lucyna_test_bfloat16 "test/lyBfloat16Test.c")

set(TEST_EXECUTABLES
        lucyna_test_tokenizer
        lucyna_test_transformer
        lucyna_test_tensor
        lucyna_test_tensor_math
        lucyna_test_attention
        lucyna_test_bfloat16
)

foreach (TEST_EXECUTABLE ${TEST_EXECUTABLES})
    configure_test_target(${TEST_EXECUTABLE})
endforeach ()
