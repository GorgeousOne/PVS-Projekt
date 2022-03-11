cmake_minimum_required(VERSION 3.20)
project(finding_nemo)

set(CMAKE_CXX_STANDARD 14)
# enable openmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")

# make mpi required
find_package(MPI REQUIRED)
find_package(OpenCL REQUIRED)

add_executable(finding_nemo
        stb_image.h
        stb_image_write.h
        template_matching.cpp)

add_executable(finding_nemo_mp
        stb_image.h
        stb_image_write.h
        template_matching_mp.cpp)

add_executable(finding_nemo_mpV2
        stb_image.h
        stb_image_write.h
        template_matching_mpV2.cpp)

add_executable(finding_nemo_mpi
        stb_image.h
        stb_image_write.h
        template_matching_mpi.cpp)

add_executable(finding_nemo_mpi_old
        stb_image.h
        stb_image_write.h
        template_matching_mpi_old.cpp)

add_executable(finding_nemo_cl
        stb_image.h
        stb_image_write.h
        template_matching_cl.cpp
        kernel_source.cl)

# link mpi
target_link_libraries(finding_nemo_mpi ${MPI_LIBRARIES})
target_link_libraries(finding_nemo_mpi_old ${MPI_LIBRARIES})

target_link_libraries(finding_nemo_cl OpenCL::OpenCL)

include_directories(${MPI_INCLUDE_PATH})