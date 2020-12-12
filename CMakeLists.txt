cmake_minimum_required(VERSION 3.8)

project(CompAvanzados LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

message(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
set(CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets)

set(SOURCES main.cu defines.h)
#add_executable(main ${SOURCES})
cuda_add_executable( main ${SOURCES} )

target_link_libraries(main ${OpenCV_LIBS})
