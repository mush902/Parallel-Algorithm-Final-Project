cmake_minimum_required(VERSION 3.21)

# set the project name and version
project(nBodyCPU)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")

find_package(OpenGL REQUIRED)

if(NOT OPENGL_FOUND)
    message(FATAL_ERROR "OpenGL not found")
endif()

find_package(glfw3 3.3 REQUIRED)

if(NOT glfw3_FOUND)
    message(FATAL_ERROR "GLFW not found")
endif()

file(GLOB SRC_FILES *.cpp)

add_executable(nBodyCPU main.cpp ${SRC_FILES})
target_include_directories(nBodyCPU PRIVATE ${PROJECT_SOURCE_DIR} ${GL_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS})
target_link_libraries(nBodyCPU OpenGL::GL glfw)