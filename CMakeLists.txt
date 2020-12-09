cmake_minimum_required(VERSION 3.10)

project(CompAvanzados LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

IF(CUDA_FOUND)
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_20,code=sm_20)
    cuda_compile(CudaKernel CudaKernels.cu)
    cuda_add_library(CudaKernels CudaKernel)
ENDIF()

include_directories(${OpenCV_INCLUDE_DIRS})
set(SOURCES main.cpp defines.hpp)
add_executable(main ${SOURCES})

target_link_libraries(main CudaKernels ${OpenCV_LIBS})
