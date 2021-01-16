//#include <opencv2/opencv.hpp>

#include <opencv2/core/core.hpp>
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

#define SOBEL_KERNEL_DIM 3
#define GAUSS_KERNEL_DIM 5

__constant__ char kernel_gaus[GAUSS_KERNEL_DIM][GAUSS_KERNEL_DIM] = {{1, 4, 6, 4, 1}, 
                                                                     {4, 16, 24, 16, 4}, 
                                                                     {6, 24, 36, 24, 6},
                                                                     {4, 16, 24, 16, 4}, 
                                                                     {1, 4, 6, 4, 1} }; 
int kernel_sx[SOBEL_KERNEL_DIM][SOBEL_KERNEL_DIM] = { {1, 0, -1}, {2, 0, -2}, {1, 0, -1} };
int kernel_sy[SOBEL_KERNEL_DIM][SOBEL_KERNEL_DIM] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };


using namespace cv;
using namespace std;

/* *
* GeForce 920MX 2GB 
* Compute capability: 5.0
* CUDA core: 256
* Threads per block: 1024
* */ 

// HEADERS
void sobel_gpu (Mat src_img);
void sobel_cpu (Mat src_img);
void performance_img(string path);
void performance_video (const string path);
void show_img (string path, int mode);
void show_video (string path);
void webcam (int use);
string type2str(int type);
void apply_sobel_gpu(Mat src_img, Mat dst_img); 
void apply_gauss_gpu(Mat src_img, Mat dst_img); 
void apply_grayscale_gpu(Mat src_img, Mat dst_img); 

inline 
void check(cudaError_t salidafuncapi, const char* nombrefunc) {
  if (salidafuncapi != cudaSuccess) {
    printf("Error %s (en la llamada a %s)\n", cudaGetErrorString(salidafuncapi),nombrefunc);
    assert(salidafuncapi == cudaSuccess);
  }
}

__global__ void sobel_gpu(unsigned char* src_img, unsigned char* dst_img, int width, int height) {

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  float dx, dy;
  if( x > 1 || y > 1 || x < width-1 || y < height-1) { // avoid edges

    dx = (-1* src_img[(y-1)*width + (x-1)]) + (-2*src_img[y*width+(x-1)]) + (-1*src_img[(y+1)*width+(x-1)]) +
          (    src_img[(y-1)*width + (x+1)]) + ( 2*src_img[y*width+(x+1)]) + (   src_img[(y+1)*width+(x+1)]);

    dy = (    src_img[(y-1)*width + (x-1)]) + ( 2*src_img[(y-1)*width+x]) + (   src_img[(y-1)*width+(x+1)]) +
          (-1* src_img[(y+1)*width + (x-1)]) + (-2*src_img[(y+1)*width+x]) + (-1*src_img[(y+1)*width+(x+1)]);

    if (dx < 0) { dx = 0; } if (dx > 255) { dx = 255; }
    if (dy < 0) { dy = 0; } if (dy > 255) { dy = 255; }

    dst_img[y*width + x] = static_cast<unsigned char>(sqrt( (dx*dx) + (dy*dy) ) );
  } 

}

__global__ void gauss_gpu(unsigned char* src_img, unsigned char* dst_img, int width, int height) {

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  float gaus = 0;
  for (int i = -2; i < GAUSS_KERNEL_DIM - 2; i++) {
    for (int j = -2; j < GAUSS_KERNEL_DIM - 2; j++) {
      if( x > 2 || y > 2 || x < width - 2 || y < height - 2 ) {
        gaus +=  kernel_gaus[i+2][j+2] * src_img[(y+j)*width + (x+i)];
      }
    }
  }

  gaus = gaus/256.0;

  if (gaus < 0) { gaus = 0; } 
  if (gaus > 255) { gaus = 255; }
  
  dst_img[y*width + x] = static_cast<unsigned char>(gaus);

}

__global__ void grayscale_gpu(unsigned char* src_img, unsigned char* dst_img, int width, int height) {

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  float grayscale = 0;
  
  grayscale = 0.144f*src_img[y*width*3 + (3*x)] + 0.587f*src_img[y*width*3 + (3*x) + 1] + 0.299f*src_img[y*width*3 + (3*x) + 2];

  if (grayscale < 0) { grayscale = 0.0; }
  if (grayscale > 255) { grayscale = 255.0; }
  
  dst_img[y*width + x] = static_cast<unsigned char>(grayscale);

}

void apply_gauss_gpu(Mat src_img, Mat dst_img) {
  int rows = src_img.cols;
  int cols = src_img.rows;
  cudaError_t ret;
  int elements = rows*cols;
  int size = elements*sizeof(unsigned char);
  unsigned char *src_dev_img = {0}, *dst_dev_img = {0};
  
  // allocate device memory src img
  ret = cudaMalloc((void**)&src_dev_img, size);
  if ( ret != cudaSuccess ) {
    printf("cudaMalloc() [src_dev_img] error in device memory allocation: %s\n", cudaGetErrorString(ret));
  }

  // allocate device memory dest img
  ret = cudaMalloc((void**)&dst_dev_img, size);
  if ( ret != cudaSuccess ) {
    printf("cudaMalloc() [dst_dev_img] error in device memory allocation: %s\n", cudaGetErrorString(ret));
  }

  // copy the data host --> device
  ret = cudaMemcpy(src_dev_img, src_img.ptr(), size, cudaMemcpyHostToDevice);
  if ( ret != cudaSuccess ) {
    printf("cudaMemcpy() H -> D error: %s\n", cudaGetErrorString(ret));
  }
  
  dim3 threadsPerBlock(16.0, 16.0);
  dim3 numBlocks( ceil( rows/16.0 ), ceil( cols/16.0 ) );
 
  // kernel call
  gauss_gpu <<<numBlocks, threadsPerBlock>>> (src_dev_img, dst_dev_img, rows, cols);
  ret = cudaGetLastError();
  if ( ret != cudaSuccess ) {
    printf("kernel error: %s\n", cudaGetErrorString(ret));
  }

  // copy the result device --> host
  ret = cudaMemcpy(dst_img.ptr(), dst_dev_img, size, cudaMemcpyDeviceToHost);
  if ( ret != cudaSuccess ) {
    printf("cudaMemcpy() D -> H error: %s\n", cudaGetErrorString(ret));
  }
  
  // free device memory
  cudaFree(src_dev_img);
  cudaFree(dst_dev_img);
}

void apply_sobel_gpu(Mat src_img, Mat dst_img) {

  int rows = src_img.cols;
  int cols = src_img.rows;
  cudaError_t ret;
  int elements = rows*cols;
  int size = elements*sizeof(unsigned char);
  unsigned char *src_dev_img = {0}, *dst_dev_img = {0};
  
  // allocate device memory src img
  ret = cudaMalloc((void**)&src_dev_img, size);
  if ( ret != cudaSuccess ) {
    printf("cudaMalloc() [src_dev_img] error in device memory allocation: %s\n", cudaGetErrorString(ret));
  }

  // allocate device memory dest img
  ret = cudaMalloc((void**)&dst_dev_img, size);
  if ( ret != cudaSuccess ) {
    printf("cudaMalloc() [dst_dev_img] error in device memory allocation: %s\n", cudaGetErrorString(ret));
  }

  // copy the data host --> device
  ret = cudaMemcpy(src_dev_img, src_img.ptr(), size, cudaMemcpyHostToDevice);
  if ( ret != cudaSuccess ) {
    printf("cudaMemcpy() H -> D error: %s\n", cudaGetErrorString(ret));
  }
  
  dim3 threadsPerBlock(16.0, 16.0);
  dim3 numBlocks( ceil( rows/16.0 ), ceil( cols/16.0) );
 
  // kernel call
  sobel_gpu <<<numBlocks, threadsPerBlock>>> (src_dev_img, dst_dev_img, rows, cols);
  ret = cudaGetLastError();
  if ( ret != cudaSuccess ) {
    printf("kernel error: %s\n", cudaGetErrorString(ret));
  }

  // copy the result device --> host
  ret = cudaMemcpy(dst_img.ptr(), dst_dev_img, size, cudaMemcpyDeviceToHost);
  if ( ret != cudaSuccess ) {
    printf("cudaMemcpy() D -> H error: %s\n", cudaGetErrorString(ret));
  }
  
  // free device memory
  cudaFree(src_dev_img);
  cudaFree(dst_dev_img);
}

void apply_grayscale_gpu(Mat src_img, Mat dst_img) {
  int cols = src_img.cols;
  int rows = src_img.rows;
  cudaError_t ret;
  int elements = rows*cols;
  int size_in = 3*elements*sizeof(unsigned char);
  int size_out = elements*sizeof(unsigned char);
  unsigned char *src_dev_img = {0}, *dst_dev_img = {0};
  
  // allocate device memory src img
  ret = cudaMalloc((void**)&src_dev_img, size_in);
  if ( ret != cudaSuccess ) {
    printf("cudaMalloc() [src_dev_img] error in device memory allocation: %s\n", cudaGetErrorString(ret));
  }

  // allocate device memory dest img
  ret = cudaMalloc((void**)&dst_dev_img, size_out);
  if ( ret != cudaSuccess ) {
    printf("cudaMalloc() [dst_dev_img] error in device memory allocation: %s\n", cudaGetErrorString(ret));
  }

  // copy the data host --> device
  ret = cudaMemcpy(src_dev_img, src_img.ptr(), size_in, cudaMemcpyHostToDevice);
  if ( ret != cudaSuccess ) {
    printf("cudaMemcpy() H -> D error: %s\n", cudaGetErrorString(ret));
  }
  
  dim3 threadsPerBlock(16.0, 16.0);
  dim3 numBlocks( ceil( cols/16.0 ), ceil( rows/16.0) );
 
  // kernel call
  grayscale_gpu <<<numBlocks, threadsPerBlock>>> (src_dev_img, dst_dev_img, cols, rows);
  ret = cudaGetLastError();
  if ( ret != cudaSuccess ) {
    printf("kernel error: %s\n", cudaGetErrorString(ret));
  }

  // copy the result device --> host
  ret = cudaMemcpy(dst_img.ptr(), dst_dev_img, size_out, cudaMemcpyDeviceToHost);
  if ( ret != cudaSuccess ) {
    printf("cudaMemcpy() D -> H error: %s\n", cudaGetErrorString(ret));
  }
  
  // free device memory
  cudaFree(src_dev_img);
  cudaFree(dst_dev_img);
}

int main(int argc, char *argv[]) {

    // Argv lib from: https://github.com/jarro2783/cxxopts
    cxxopts::Options options("CONVOLUTIONer", "By using this program you can compare how faster the convolution is done on GPU vs CPU");
    try {

      options.add_options()
        ("h", "Print usage and exit")
        ("pi", "Performance test using image specified", cxxopts::value<string>())
        ("pv", "Performance test using video specified", cxxopts::value<string>())
        ("si_gpu", "Show the image specified (GPU)", cxxopts::value<string>())
        ("si_cpu", "Show the image specified (CPU)", cxxopts::value<string>())
        ("sv", "Show the video specified (GPU)", cxxopts::value<string>())
        ("w", "Use webcam video stream [CPU: 0, GPU: 1, No filter: 2]", cxxopts::value<int>());
            
      auto result = options.parse(argc, argv);

      if (result.count("h")) {
        cout << options.help() << endl;
        return 0;
      }
  
      // create context
      cudaFree(0); 
  
      if (result.count("pi")) {
        performance_img( result["pi"].as<string>() );
      } 

      if (result.count("pv")) {
        performance_video( result["pv"].as<string>() );
      } 
      
      if (result.count("w")) {
        webcam( result["w"].as<int>() );
      }

      if (result.count("si_cpu")) {
        show_img(result["si_cpu"].as<string>(), 0);
      }

      if (result.count("si_gpu")) {
        show_img(result["si_gpu"].as<string>(), 1);
      }

      if (result.count("sv")) {
        show_video(result["sv"].as<string>() );
      }

      if (result.count("h") == 0 && result.count("pi") == 0 && 
          result.count("pv") == 0 && result.count("w") == 0 && 
          result.count("sv") == 0 && result.count("si_cpu") == 0 &&
          result.count("si_gpu") == 0 ) {
        cout << options.help() << endl;
      }
  
    } catch (const cxxopts::OptionException& e) {
      cout << options.help() << endl;
      return 0;
    }
    
    return 0;
}

void sobel_gpu (Mat src_img, Mat dst_img) {

  Mat src_img_gray(src_img.rows, src_img.cols, CV_8UC1);
  Mat img_blur(src_img.rows, src_img.cols, CV_8UC1);

  apply_grayscale_gpu(src_img, src_img_gray);

  apply_gauss_gpu(src_img_gray, img_blur);
 
  apply_sobel_gpu(img_blur, dst_img);
}

void sobel_cpu (Mat src_img, Mat dst_img) {

  Mat src_img_gray (src_img.rows, src_img.cols, CV_8UC1);
  Mat img_blur(src_img.rows, src_img.cols, CV_8UC3);

  GaussianBlur(src_img, img_blur, Size(3, 3), 0, 0, BORDER_DEFAULT);
  cvtColor(img_blur, src_img_gray, COLOR_BGR2GRAY);
   
  uchar *src_array = src_img_gray.ptr();

  int width = src_img_gray.cols; 
  int height = src_img_gray.rows; 

  float dx, dy;
  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {

      dx = 0; dy = 0;
      for (int x = -1; x < SOBEL_KERNEL_DIM - 1; x++) {
        for (int y = -1; y < SOBEL_KERNEL_DIM - 1; y++) {
            dx += src_array[(i + x)*width + j + y] * kernel_sx[x + 1][y + 1];
            dy += src_array[(i + x)*width + j + y] * kernel_sy[x + 1][y + 1];
        }
      }

      if (dx < 0) { dx = 0; } 
      if (dx > 255) { dx = 255; }
      if (dy < 0) { dy = 0; } 
      if (dy > 255) { dy = 255; }

      dst_img.ptr()[i*width + j] =  static_cast<unsigned char>(sqrt( (dx*dx) + (dy*dy) ));
    }
  }

}

void performance_img(string path) {

  Mat img = imread(path, IMREAD_COLOR);
  if (img.empty()) {
    cout << "Error opening the file" << endl;
    exit(-1);
  }

  cout << "Image: " << path << endl;
  cout << " - resolution: " << img.cols << "x" << img.rows << endl;
  cout << " - channels: " << img.channels() << endl;
  cout << " - type: " << type2str(img.type()) << endl;

  Mat sobel_img (img.rows, img.cols, CV_8UC1);
  auto start = chrono::system_clock::now();
  sobel_cpu(img, sobel_img);
  auto end = chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end-start;

  cout << "[CPU] time: " << 1000*elapsed_seconds.count() << "ms\n";

  start = chrono::system_clock::now();
  sobel_gpu(img, sobel_img);
  end = chrono::system_clock::now();

  elapsed_seconds = end-start;

  cout << "[GPU] time: " << 1000*elapsed_seconds.count() << "ms\n";

  waitKey(0);
}

void performance_video (string path) {

  VideoCapture cap;
  if (cap.open(path, cv::CAP_ANY) == false) {
    cout << "Could not open video" << endl;
    exit(-1);
  }

  cout << "Video: " << path << endl;
  cout << " - resolution: " << cap.get(CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
  cout << " - fps: " << cap.get(CAP_PROP_FPS ) << endl;
  cout << " - nframes: " <<  cap.get(CAP_PROP_FRAME_COUNT) << endl;
  cout << " - duration: " << cap.get(CAP_PROP_FRAME_COUNT)/cap.get(CAP_PROP_FPS) << " seconds " << endl;

  Mat img (cap.get(CAP_PROP_FRAME_HEIGHT), cap.get(CAP_PROP_FRAME_WIDTH), CV_8UC1);
  Mat frame;
  auto start = chrono::system_clock::now();
  while(1){

    cap >> frame;
    if(frame.empty()) {
      break;
    }
 
    sobel_gpu(frame, img);

  }
  auto end = chrono::system_clock::now();


  std::chrono::duration<double> elapsed_seconds = end-start;

  cout << "[GPU] - time: " << elapsed_seconds.count() << "s\n";
  cout << "      - fps: " << cap.get(CAP_PROP_FRAME_COUNT)/elapsed_seconds.count() << endl;

  cap.set(CAP_PROP_POS_AVI_RATIO, 0);

  start = chrono::system_clock::now();
  while(1){

    cap >> frame;
    if(frame.empty()) {
      break;
    }
 
    sobel_cpu(frame, img);

  }
  end = chrono::system_clock::now();


  elapsed_seconds = end-start;

  cout << "[CPU] - time: " << elapsed_seconds.count() << "s\n";
  cout << "      - fps: " << cap.get(CAP_PROP_FRAME_COUNT)/elapsed_seconds.count() << endl;
 
  cap.release();
  destroyAllWindows(); 
}

void webcam (int use) {

  VideoCapture cap(0); 
   
  // Check if camera opened successfully
  if(!cap.isOpened()){
      cout << "Error opening video stream or file" << endl;
      exit(-1);
  }

  cout << "Webcam: " << endl;
  cout << " - resolution: " << cap.get(CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;

  if (use == 0) {
    cout << "Using CPU function" << endl;
  } else if (use == 1) {
    cout << "Using GPU function" << endl;
  } 

  cout << "Video stream sucesfully opened!\nPress [ESC] to quit." << endl;

  Mat frame;
  Mat img (cap.get(CAP_PROP_FRAME_HEIGHT), cap.get(CAP_PROP_FRAME_WIDTH), CV_8UC1);
  while(1){

      cap >> frame;

      if (use == 0) {
        sobel_cpu(frame, img);
      } else if (use == 1) {
        sobel_gpu(frame, img);
      } else {
        img = frame;
      }
      
      imshow( "SOBEL IMAGE", img);
      if((char)waitKey(25) == 27) {
          break;
      }

  }
  
  // free resources
  cap.release();
  destroyAllWindows();
}

void show_img (string path, int mode) {

  Mat img = imread(path, IMREAD_COLOR);
  if (img.empty()) {
    cout << "Error opening the file" << endl;
    exit(-1);
  }

  cout << "Image: " << path << endl;
  cout << " - resolution: " << img.cols << "x" << img.rows << endl;
  cout << " - channels: " << img.channels() << endl;
  cout << " - type: " << type2str(img.type()) << endl;

  Mat sobel_img (img.rows, img.cols, CV_8UC1);
  if (mode == 0) {
    sobel_cpu(img, sobel_img);
  } else if (mode == 1) {
    sobel_gpu(img, sobel_img);
  }
  
  if (img.cols >= 1280) {
    Mat r_img;
    resize(sobel_img, r_img, Size(1280, 720));
    imshow( "SOBEL IMAGE", r_img );
  } else {
    imshow( "SOBEL IMAGE", sobel_img);
  }

  waitKey(0);
}

void show_video (string path) {
  
  VideoCapture cap;
  if (cap.open(path, cv::CAP_ANY) == false) {
    cout << "Could not open video" << endl;
    exit(-1);
  }

  cout << "Video: " << path << endl;
  cout << " - resolution: " << cap.get(CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
  cout << " - fps: " << cap.get(CAP_PROP_FPS) << endl;
  cout << " - nframes: " <<  cap.get(CAP_PROP_FRAME_COUNT) << endl;
  cout << " - duration: " << cap.get(CAP_PROP_FRAME_COUNT)/cap.get(CAP_PROP_FPS) << " seconds " << endl;

  Mat frame, r_img;;
  Mat img (cap.get(CAP_PROP_FRAME_HEIGHT), cap.get(CAP_PROP_FRAME_WIDTH), CV_8UC1);
  Size custom_size = img.size();
  if (img.cols >= 1280) {
    custom_size = Size (1280, 720);
  }
  while(1){

    cap >> frame;
    if(frame.empty()) {
      break;
    }
  

    sobel_gpu(frame, img);
    resize(img, r_img, custom_size);
    imshow( "SOBEL VIDEO", r_img);

    if((char)waitKey(25) == 27) {
        break;
    }

  }

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