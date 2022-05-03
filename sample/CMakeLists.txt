cmake_minimum_required(VERSION 3.5)

project(sample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++14 -ffast-math -Wall -fstrict-aliasing") 
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(..)
file(GLOB HEADER_FILES helper.h ../sadl/*.h )

add_executable(sample_generic   sample.cpp ${HEADER_FILES})
add_executable(sample_simd256   sample.cpp ${HEADER_FILES})
add_executable(sample_simd512   sample.cpp ${HEADER_FILES})
set_target_properties(sample_simd256  PROPERTIES COMPILE_FLAGS "-mavx2 -DNDEBUG=1 " )
set_target_properties(sample_simd512  PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw  -DNDEBUG=1" )

add_executable(count_mac   count_mac.cpp ${HEADER_FILES})
set_target_properties(count_mac  PROPERTIES COMPILE_FLAGS "-DNDEBUG=1 " )  # must build in scalar mode to count MAC

add_executable(debug_model   debug_model.cpp ${HEADER_FILES})
set_target_properties(debug_model  PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw" ) # must build in SIMD mode to debug SIMD issue



