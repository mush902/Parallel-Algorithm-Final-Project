cmake_minimum_required(VERSION 3.10)

project(DirectSum)

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})


find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found")
endif()
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)

file(GLOB SOURCES *.h *.cu)

cuda_add_executable(DirectSum ${SOURCES})

target_link_libraries(DirectSum ${OpenCV_LIBS})