cmake_minimum_required(VERSION 3.15)
project(geopyv LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DEIGEN_DONT_PARALLELIZE")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")

# Find dependencies
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# Install the Python package files (e.g. pure Python modules)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/geopyv/
  DESTINATION geopyv
  FILES_MATCHING PATTERN "*.py")

# Build the `_image_extensions` module
pybind11_add_module(_image_extensions src/geopyv/_image.cpp)
target_link_libraries(_image_extensions PUBLIC Eigen3::Eigen)
install(TARGETS _image_extensions LIBRARY DESTINATION geopyv)

# Build the `_subset_extensions` module
pybind11_add_module(_subset_extensions src/geopyv/_subset.cpp)
target_link_libraries(_subset_extensions PUBLIC Eigen3::Eigen)
install(TARGETS _subset_extensions LIBRARY DESTINATION geopyv)
