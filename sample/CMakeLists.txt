cmake_minimum_required(VERSION 3.5)

project(sample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(..)
file(GLOB HEADER_FILES ../sadl/*.h ) # just for IDE

add_executable(sample_generic   main.cpp ${HEADER_FILES})
add_executable(sample_simd      main.cpp ${HEADER_FILES})
add_executable(sample_simd512   main.cpp ${HEADER_FILES})

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG=1")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -DDEBUG_COUNTERS=1 -DDEBUG_MODEL=1 -DDEBUG_SIMD=1")

if( MSVC )       
  set_target_properties(sample_simd     PROPERTIES COMPILE_FLAGS "/arch:AVX2" )
  set_target_properties(sample_simd512  PROPERTIES COMPILE_FLAGS "/arch:AVX512" )
elseif( UNIX OR MINGW )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -ffast-math -fstrict-aliasing") 
  set_target_properties(sample_simd     PROPERTIES COMPILE_FLAGS "-mavx2 " )
  set_target_properties(sample_simd512  PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mfma" )
endif()

