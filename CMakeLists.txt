# cmake_minimum_required(VERSION 3.17)
project(topology_inference)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-function -O3 -march=broadwell")
set(CMAKE_CXX_STANDARD 20)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)
find_program(MPIRUN /opt/amazon/openmpi/bin/mpirun)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(latency_calculator latency_calculator.cc rdma_context.cc rdma_resources.cc)

target_link_libraries(latency_calculator PUBLIC ${MPI_LIBRARIES} ${CUDA_LIBRARIES} ibverbs efa pthread)

add_executable(ring_latency_calculator ring_latency_calculator.cc rdma_context.cc rdma_resources.cc)

target_link_libraries(ring_latency_calculator PUBLIC ${MPI_LIBRARIES} ${CUDA_LIBRARIES} ibverbs efa pthread)