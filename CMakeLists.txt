cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(filter_benchmark)

#find_package(pybind11 CONFIG REQUIRED)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory(vendor/pybind11)
include_directories(vendor/Halide/include)
link_directories(vendor/Halide/lib)

add_library(filterbench MODULE src/benchmark.cxx)
set_target_properties(filterbench PROPERTIES PREFIX "")
target_link_libraries(filterbench PUBLIC png jpeg Halide pybind11::module)
