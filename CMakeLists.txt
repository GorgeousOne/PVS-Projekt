cmake_minimum_required(VERSION 3.20)
project(Projekt)

set(CMAKE_CXX_STANDARD 14)
# enable openmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

# make mpi required
find_package(MPI REQUIRED)

add_executable(Projekt
        stb_image.h
        stb_image_write.h
        template_matching.cpp)

# link mpi
target_link_libraries(Projekt ${MPI_LIBRARIES})

include_directories(${MPI_INCLUDE_PATH})