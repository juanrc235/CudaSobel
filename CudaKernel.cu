#include <assert.h>
#include <iostream>

__global__ void kernel(int *g_idata, int *g_odata) {
	
}

inline 
void check(cudaError_t salidafuncapi, const char* nombrefunc) {
  if (salidafuncapi != cudaSuccess) {
    printf("Error %s (en la llamada a %s)\n", cudaGetErrorString(salidafuncapi),nombrefunc);
    assert(salidafuncapi == cudaSuccess);
  }
}