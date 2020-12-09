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


__global__ void kernel_conv(unsigned char *src_img, unsigned char *dst_img, int cols, int rows) {
  
}

void kernel_wrapper(unsigned char *src_img, unsigned char *dst_img, int cols, int rows) {
  
  cudaDeviceSynchronize();
}