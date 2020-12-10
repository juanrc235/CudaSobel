#include <assert.h>
#include <iostream>
#include "defines.h"

inline 
void check(cudaError_t salidafuncapi, const char* nombrefunc) {
  if (salidafuncapi != cudaSuccess) {
    printf("Error %s (en la llamada a %s)\n", cudaGetErrorString(salidafuncapi),nombrefunc);
    assert(salidafuncapi == cudaSuccess);
  }
}


__global__ void kernel_conv(unsigned char* src_img, unsigned char* dst_img, int cols, int rows) {
  
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

  // kernel call
  kernel_conv <<<1, 1>>> (src_dev_img, dst_dev_img, cols, rows);

  // copy the result device --> host
  ret = cudaMemcpy(dst_img, dst_dev_img, size, cudaMemcpyDeviceToHost);
  if ( ret != cudaSuccess ) {
    printf("cudaMemcpy() D -> H error: %s\n", cudaGetErrorString(ret));
  }
  
  // free device memory
  cudaFree(src_dev_img);
  cudaFree(dst_dev_img);
}

void gpu_stats () {

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Device name: %s\n", prop.name);
  printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
  printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  int cores = prop.multiProcessorCount;
  switch (prop.major) {
    case 2: // Fermi
      if (prop.minor == 1) cores *= 48;
      else cores *= 32; break;
    case 3: // Kepler
      cores *= 192; break;
    case 5: // Maxwell
      cores *= 128; break;
    case 6: // Pascal
      if (prop.minor == 1) cores *= 128;
      else if (prop.minor == 0) cores *= 64;
      break;
  }
  printf("CUDA cores: %d \n", cores);

}