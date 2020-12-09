#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>


#define KERNEL_DIM 3

using namespace cv;
using namespace std;

double kernel_gaus[KERNEL_DIM][KERNEL_DIM] = { {1, 1, 1}, {1, 1, 1}, {1, 1, 1} };

int kernel_sx[KERNEL_DIM][KERNEL_DIM] = { {1, 0, -1}, 
                                          {2, 0, -2}, 
                                          {1, 0, -1} };

int kernel_sy[KERNEL_DIM][KERNEL_DIM] = { {1, 2, 1}, 
                                          {0, 0, 0}, 
                                          {-1, -2, -1} };

