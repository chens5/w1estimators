cmake_minimum_required(VERSION 3.4)
project(PDEstimators)

add_compile_options(-O3)
add_compile_options(-Wall)
add_compile_options(-march=native)

add_subdirectory(external/pybind11)
include_directories(external/eigen)

pybind11_add_module(pd_estimators src/pd_estimators.cpp)
