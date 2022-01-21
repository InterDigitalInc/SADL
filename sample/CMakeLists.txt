cmake_minimum_required(VERSION 3.5)

project(sample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++14 -ffast-math -Wall -fstrict-aliasing") 
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG=1 -DDEBUG_PRINT=1")
set(CMAKE_CXX_FLAGS_DEBUG   "-g  -DDEBUG_COUNTERS=1 -DDEBUG_MODEL=1 -DDEBUG_SIMD=1 -DDEBUG_PRINT=1")

include_directories(..)
file(GLOB HEADER_FILES ../sadl/*.h )
add_executable(sample_generic   main.cpp ${HEADER_FILES})
add_executable(sample_simd      main.cpp ${HEADER_FILES})
add_executable(sample_simd512   main.cpp ${HEADER_FILES})
set_target_properties(sample_simd     PROPERTIES COMPILE_FLAGS "-mavx2 " )
set_target_properties(sample_simd512  PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mfma" )


