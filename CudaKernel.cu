#include <assert.h>
#include <iostream>
#include <math.h>
#include "defines.h"

#define GRIDVAL 20.0

inline 
void check(cudaError_t salidafuncapi, const char* nombrefunc) {
  if (salidafuncapi != cudaSuccess) {
    printf("Error %s (en la llamada a %s)\n", cudaGetErrorString(salidafuncapi),nombrefunc);
    assert(salidafuncapi == cudaSuccess);
  }
}


__global__ void kernel_conv(unsigned char* src_img, unsigned char* dst_img, int width, int height) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst_img[y*width + x] = 0;
}

void kernel_wrapper(unsigned char *src_img, unsigned char *dst_img, int cols, int rows) {

  cudaError_t ret;
  int elements = rows*cols;
  int size = elements*sizeof(unsigned char);
  unsigned char *src_dev_img, *dst_dev_img;

  // create context
  cudaFree(0); 

  // allocate device memory
  ret = cudaMalloc((void**)&src_dev_img, size);
  if ( ret != cudaSuccess ) {
    printf("cudaMalloc() [src_dev_img] error in device memory allocation: %s\n", cudaGetErrorString(ret));
  }

  ret = cudaMalloc((void**)&dst_dev_img, size);
  if ( ret != cudaSuccess ) {
    printf("cudaMalloc() [dst_dev_img] error in device memory allocation: %s\n", cudaGetErrorString(ret));
  }

  // copy the data host --> device
  ret = cudaMemcpy(src_dev_img, src_img, size, cudaMemcpyHostToDevice);
  if ( ret != cudaSuccess ) {
    printf("cudaMemcpy() H -> D error: %s\n", cudaGetErrorString(ret));
  }

  dim3 tpb(GRIDVAL, GRIDVAL, 1);
  dim3 nBlocks(ceil(cols/GRIDVAL), ceil(rows/GRIDVAL), 1);

  // kernel call
  kernel_conv <<<1, 32>>> (src_dev_img, dst_dev_img, cols, rows);
  ret = cudaGetLastError();
  if ( ret != cudaSuccess ) {
    printf("kernel error: %s\n", cudaGetErrorString(ret));
  }

  // copy the result device --> host
  ret = cudaMemcpy(dst_img, dst_dev_img, size, cudaMemcpyDeviceToHost);
  if ( ret != cudaSuccess ) {
    printf("cudaMemcpy() D -> H error: %s\n", cudaGetErrorString(ret));
  }
  
  // free device memory
  cudaFree(src_dev_img);
  cudaFree(dst_dev_img);
}
