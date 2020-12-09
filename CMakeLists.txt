cmake_minimum_required(VERSION 3.8)

project(CompAvanzados LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

enable_language("CUDA")
set(SOURCES main.cpp CudaKernel.cu defines.h)
add_executable(main ${SOURCES})

target_link_libraries(main ${OpenCV_LIBS})
