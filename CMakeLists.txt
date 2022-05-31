cmake_minimum_required(VERSION 3.1)
project(geopyv)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DEIGEN_DONT_PARALLELIZE")
message(${CMAKE_CXX_FLAGS})

find_package (Eigen3 3.3 REQUIRED)
find_package (pybind11 REQUIRED)

pybind11_add_module(_image_extensions src/geopyv/image/_image.cpp)
target_link_libraries(_image_extensions PUBLIC Eigen3::Eigen)
install(TARGETS _image_extensions DESTINATION ./image)

pybind11_add_module(_subset_extensions src/geopyv/subset/_subset.cpp)
target_link_libraries(_subset_extensions PUBLIC Eigen3::Eigen)
install(TARGETS _subset_extensions DESTINATION ./subset)
