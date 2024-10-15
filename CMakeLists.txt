cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE  STREQUAL "Debug")
    message("Debug mode: set min cuda arch")
set(CMAKE_CUDA_ARCHITECTURES 52)
endif()

project(DISTANCEN LANGUAGES CXX CUDA)

add_executable(main main.cu)
# target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
# target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
