# Android 端 NSFW 图像识别（ggml CPU 后端）
cmake_minimum_required(VERSION 3.22.1)

project("nsfwimagedetectiongguf" C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug 变体下 AGP 会给原生代码传 -O0，ggml 在无优化时慢一个数量级。
# 这里统一追加 -O3（命令行后出现的优化级别生效），保证调试包也有可用性能。
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O3)
endif()

# ---- 桌面工具目录（复用其中的 ggml 与 stb）----
# 当前文件位于 android/app/src/main/cpp，向上 5 级即仓库根目录
get_filename_component(NSFW_TOOL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../_nsfw_tool" ABSOLUTE)

# ---- ggml: 只保留 CPU 后端，静态链接 ----
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE         OFF CACHE BOOL "" FORCE)  # 交叉编译，禁止 -march=native
set(GGML_OPENMP         OFF CACHE BOOL "" FORCE)  # NDK 默认不带 libomp
set(GGML_VULKAN         OFF CACHE BOOL "" FORCE)
set(GGML_CUDA           OFF CACHE BOOL "" FORCE)
set(GGML_METAL          OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL         OFF CACHE BOOL "" FORCE)
set(GGML_SYCL           OFF CACHE BOOL "" FORCE)
set(GGML_HIP            OFF CACHE BOOL "" FORCE)

add_subdirectory("${NSFW_TOOL_DIR}/ggml" ggml-build)

add_library(${CMAKE_PROJECT_NAME} SHARED
        nsfw_core.cpp
        nsfw_jni.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        "${NSFW_TOOL_DIR}/ggml/include"
        "${NSFW_TOOL_DIR}/ggml/src"
        "${NSFW_TOOL_DIR}/third_party/stb")

# stb 只做内存解码
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE STBI_NO_STDIO)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -O3)

target_link_libraries(${CMAKE_PROJECT_NAME}
        ggml
        ggml-base
        ggml-cpu
        android
        log)
