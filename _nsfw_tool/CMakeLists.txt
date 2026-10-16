cmake_minimum_required(VERSION 3.22)
project(vit_nsfw LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- ggml: 关闭测试/示例，静态链接，开启 Vulkan (一套后端支持 NVIDIA/AMD/Intel) ----
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN         ON  CACHE BOOL "" FORCE)
set(GGML_CUDA           OFF CACHE BOOL "" FORCE)
set(GGML_METAL          OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL         OFF CACHE BOOL "" FORCE)
set(GGML_SYCL           OFF CACHE BOOL "" FORCE)
set(GGML_HIP            OFF CACHE BOOL "" FORCE)
# 静态链接 MSVC 运行时，产出单个 exe 便于分发
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_subdirectory(ggml)

# 静态库形式的后端，确保注册器不被链接器丢弃
foreach(bt IN ITEMS ggml ggml-base ggml-cpu ggml-vulkan)
    if(TARGET ${bt})
        list(APPEND GGML_BACKEND_TARGETS ${bt})
    endif()
endforeach()

function(configure_target tgt)
    set_target_properties(${tgt} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    target_link_libraries(${tgt} PRIVATE ${GGML_BACKEND_TARGETS})
    target_include_directories(${tgt} PRIVATE "${CMAKE_SOURCE_DIR}/third_party/stb")
    if(WIN32)
        target_compile_definitions(${tgt} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
        target_compile_options(${tgt} PRIVATE /utf-8)
    endif()
endfunction()

add_executable(vit_nsfw vit_nsfw.cpp)
configure_target(vit_nsfw)

add_executable(nsfw_audit nsfw_audit.cpp)
configure_target(nsfw_audit)
