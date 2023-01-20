cmake_minimum_required(VERSION 3.1)
project(geopyv)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DEIGEN_DONT_PARALLELIZE")
message(${CMAKE_CXX_FLAGS})

add_subdirectory(external)

pybind11_add_module(_image_extensions src/geopyv/_image.cpp)
target_link_libraries(_image_extensions PUBLIC Eigen3::Eigen)
install(TARGETS _image_extensions DESTINATION ./)

pybind11_add_module(_subset_extensions src/geopyv/_subset.cpp)
target_link_libraries(_subset_extensions PUBLIC Eigen3::Eigen)
install(TARGETS _subset_extensions DESTINATION ./)
