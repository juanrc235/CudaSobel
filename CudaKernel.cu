#include <assert.h>
#include <iostream>
#include "wrapper.h"

__global__ void kernel() {

}

inline 
void check(cudaError_t salidafuncapi, const char* nombrefunc) {
  if (salidafuncapi != cudaSuccess) {
    printf("Error %s (en la llamada a %s)\n", cudaGetErrorString(salidafuncapi),nombrefunc);
    assert(salidafuncapi == cudaSuccess);
  }
}

void my_cuda_func(){
    kernel<<<1,1>>>();
    printf("##################################\n");
    cudaDeviceSynchronize();
}