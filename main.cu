#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include "opencv2/imgcodecs.hpp"

#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <assert.h>
#include <chrono>
#include <ctime>

#include "cxxopts.hpp"
#include "defines.h"

using namespace cv;
using namespace std;

// https://qiita.com/naoyuki_ichimura/items/8c80e67a10d99c2fb53c

// HEADERS
Mat sobel_opencv(Mat img);
Mat sobel_gpu (Mat src_img);
Mat sobel_cpu (Mat src_img);
void performance(string path);
void webcam (int use);
string type2str(int type);
void kernel_wrapper(unsigned char *src_img, unsigned char *dst_img, int cols, int rows); 

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

    float dx, dy;
    if( x > 0 && y > 0 && x < width-1 && y < height-1) { // avoid edges

        dx = (-1* src_img[(y-1)*width + (x-1)]) + (-2*src_img[y*width+(x-1)]) + (-1*src_img[(y+1)*width+(x-1)]) +
             (    src_img[(y-1)*width + (x+1)]) + ( 2*src_img[y*width+(x+1)]) + (   src_img[(y+1)*width+(x+1)]);

        dy = (    src_img[(y-1)*width + (x-1)]) + ( 2*src_img[(y-1)*width+x]) + (   src_img[(y-1)*width+(x+1)]) +
             (-1* src_img[(y+1)*width + (x-1)]) + (-2*src_img[(y+1)*width+x]) + (-1*src_img[(y+1)*width+(x+1)]);

        dst_img[y*width + x] = sqrt( (dx*dx) + (dy*dy) );
    }

}

void kernel_wrapper(unsigned char *src_img, unsigned char *dst_img, int cols, int rows) {

  cudaError_t ret;
  int elements = rows*cols;
  int size = elements*sizeof(unsigned char);
  unsigned char *src_dev_img, *dst_dev_img;

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
  
  /* *
  * GeForce 920MX 2GB 
  * Compute capability: 5.0
  * CUDA core: 256
  * Threads per block: 1024
  * */ 
  dim3 threadsPerBlock(20.0, 20.0);
  dim3 numBlocks(ceil(rows/20.0), ceil(cols/20.0));

  // kernel call
  kernel_conv <<<numBlocks, threadsPerBlock>>> (src_dev_img, dst_dev_img, rows, cols);
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


void mat2array (Mat img, unsigned char *array_img) {

  for(int i = 0; i < img.rows; ++i) {
      for(int j = 0; j < img.cols; ++j) {
          array_img[i*img.cols + j] = ( img.at<Vec3b>(i, j)[0] + 
                                        img.at<Vec3b>(i, j)[1] + 
                                        img.at<Vec3b>(i, j)[2] ) /3 ;
      }
  }
}


Mat array2mat ( unsigned char array_img[], int row, int col) {

    Mat img_m(row, col, CV_8UC1);
  
    for(int i = 0; i < row; ++i) {
        for(int j = 0; j < col; ++j) {
            img_m.at<uchar>(i, j) = array_img[i*col + j];
        }
    }

    return img_m;
}


int main(int argc, char *argv[]) {

    // https://github.com/jarro2783/cxxopts
    cxxopts::Options options("CONVOLUTIONer", "By using this program you can compare how faster the convolution is done on GPU vs CPU");
    try {

      options.add_options()
        ("h", "Print usage and exit")
        ("p", "Performance test using image specified", cxxopts::value<string>())
        ("w", "Use webcam video stream [CPU: 0, GPU: 1]", cxxopts::value<int>());
            
      auto result = options.parse(argc, argv);

      if (result.count("h")) {
        cout << options.help() << endl;
        return 0;
      }
  
      // create context
      cudaFree(0); 
  
      if (result.count("p")) {
        performance( result["p"].as<string>() );
      } 
      
      if (result.count("w")) {
        webcam( result["w"].as<int>() );
      }

      if (result.count("h") == 0 && result.count("p") == 0 && result.count("w") == 0) {
        cout << options.help() << endl;
      }
  
    } catch (const cxxopts::OptionException& e) {
      cout << options.help() << endl;
      return 0;
    }
    
    return 0;
}


Mat sobel_opencv(Mat img) {
    Mat src, src_gray, grad_x, grad_y, abs_grad_x, abs_grad_y, grad;
    int ksize = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    // apply Gaussian blur
    GaussianBlur(img, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

    // to gray scale
    cvtColor(img, src_gray, COLOR_BGR2GRAY);

    Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    return grad;
}

Mat sobel_gpu (Mat src_img) {

   unsigned char dst_array[src_img.cols*src_img.rows];
   unsigned char src_array[src_img.cols*src_img.rows];

   mat2array(src_img, src_array);

   kernel_wrapper(src_array, dst_array, src_img.rows, src_img.cols);

   return array2mat(dst_array, src_img.rows, src_img.cols);
}

Mat sobel_cpu (Mat src_img) {

  unsigned char dst_array[src_img.cols*src_img.rows];
  unsigned char src_array[src_img.cols*src_img.rows];

  mat2array(src_img, src_array);
  
  int width = src_img.cols;
  int height = src_img.rows;

  float dx, dy;

  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {

      dx = 0; dy = 0;
      for (int x = -1; x < KERNEL_DIM - 1; x++) {
        for (int y = -1; y < KERNEL_DIM - 1; y++) {
            dx += src_array[(i + x)*width + j + y] * kernel_sx[x + 1][y + 1];
            dy += src_array[(i + x)*width + j + y] * kernel_sy[x + 1][y + 1];
        }
      }

      dst_array[i*width + j] = sqrt( (dx*dx) + (dy*dy) );
    }
  }

  return array2mat(dst_array, src_img.rows, src_img.cols);
}

void performance(string path) {

  Mat img = imread(path, IMREAD_COLOR);
  if (img.empty()) {
    cout << "Error opening the file" << endl;
    exit(-1);
  }

  cout << "Image: " << path << endl;
  cout << " - resolution: " << img.cols << "x" << img.rows << endl;
  cout << " - channels: " << img.channels() << endl;
  cout << " - type: " << type2str(img.type()) << endl;
  
  auto start = chrono::system_clock::now();
  sobel_cpu(img);
  auto end = chrono::system_clock::now();
  
  std::chrono::duration<double> elapsed_seconds = end-start;
  
  cout << "[CPU] time: " << elapsed_seconds.count() << "s\n";

  start = chrono::system_clock::now();
  sobel_gpu(img);
  end = chrono::system_clock::now();
  
  elapsed_seconds = end-start;
  
  cout << "[GPU] time: " << elapsed_seconds.count() << "s\n";

  exit(0);

  start = chrono::system_clock::now();
  sobel_opencv(img);
  end = chrono::system_clock::now();
  
  elapsed_seconds = end-start;
  
  cout << "[OPENCV] time: " << elapsed_seconds.count() << "s\n";


}

void webcam (int use) {

  VideoCapture cap(0); 
   
  // Check if camera opened successfully
  if(!cap.isOpened()){
      cout << "Error opening video stream or file" << endl;
      exit(-1);
  }

  cap.set(CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G') );
  cap.set(CAP_PROP_FRAME_WIDTH, 1920);
  cap.set(CAP_PROP_FRAME_HEIGHT, 1080);

  if (use == 0) {
    cout << "Using CPU function" << endl;
  } else if (use == 1) {
    cout << "Using GPU function" << endl;
  } else {
    cout << "Using OPENCV function" << endl;
  }

  cout << "Video stream sucesfully opened!\nPress [ESC] to quit." << endl;

  Mat frame, img;
  std::chrono::duration<double> elapsed_seconds;
  auto start = chrono::system_clock::now();
  auto end = chrono::system_clock::now();
  while(1){

      cap >> frame;

      if (use == 0) {
        img = sobel_cpu(frame);
      } else if (use == 1) {
        img = sobel_gpu(frame);
      } else if (use == 2) {
        img = sobel_opencv(frame);
      } else {
        img = frame;
      }
      
      start = chrono::system_clock::now();
      elapsed_seconds = (start-end);
      end = start;
  
      putText(img, to_string((int)(1/elapsed_seconds.count())), Point(5, 25), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 0, 0), 2);
      imshow( "SOBEL IMAGE", img);
      if((char)waitKey(25) == 27) {
          break;
      }

  }
  
  // free resources
  cap.release();
  destroyAllWindows();
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
