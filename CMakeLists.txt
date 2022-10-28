# required cmake version
cmake_minimum_required(VERSION 3.16.0)

set(CMAKE_CXX_STANDARD 11)

project(pet_ray_tracer_in_cuda)

find_package(CUDA)

INCLUDE_DIRECTORIES (src)

file(GLOB_RECURSE CURRENT_HEADERS  
    src/camera/*.h 
    src/camera/*.hpp 
    src/camera/*.cuh
    src/geometry/*.h
    src/geometry/*.hpp
    src/geometry/*.cuh
    src/helper/*.h
    src/helper/*.hpp
    src/helper/*.cuh
    src/material/*.h
    src/material/*.hpp
    src/material/*.cuh
    )
file(GLOB CURRENT_SOURCES  
    src/camera/*.cpp 
    src/camera/*.cu
    src/geometry/*.cpp
    src/geometry/*.cu
    src/helper/*.cpp
    src/helper/*.cu
    src/material/*.cpp
    src/material/*.cu
    src/*.cpp
    src/*.cu
    )

# Visual Studio source folders
SOURCE_GROUP (core/camera REGULAR_EXPRESSION src/camera/.*)
SOURCE_GROUP (core/geometry REGULAR_EXPRESSION src/geometry/.*)
SOURCE_GROUP (core/helper REGULAR_EXPRESSION src/helper/.*)
SOURCE_GROUP (core/material REGULAR_EXPRESSION src/material/.*)

CUDA_ADD_EXECUTABLE(tracer ${CURRENT_HEADERS} ${CURRENT_SOURCES})