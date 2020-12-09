#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

// HEADERS
Mat sobel_opencv(Mat img);
Mat sobel_gpu (Mat src_img);
void kernel_wrapper(unsigned char *src_img, unsigned char *dst_img, int cols, int rows); 

void mat2array (Mat img, unsigned char *array_img) {

    for(int i = 0; i < img.rows; ++i) {
        for(int j = 0; j < img.cols; ++j) {
            array_img[i*img.cols + j] = ( img.at<Vec3b>(i, j)[0] + 
                                          img.at<Vec3b>(i, j)[0] + 
                                          img.at<Vec3b>(i, j)[0] ) /3 ;
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


int main() {

    VideoCapture cap(0); 
   
    // Check if camera opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    printf("Video stream sucesfully opened!\nPress [ESC] to quit.\n");
        
    while(1){

        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            break;
        }

        imshow( "SOBEL" , sobel_gpu(frame) );

        if((char)waitKey(25) == 27) {
            break;
        }
       
    }
    
    // free resources
    cap.release();
    destroyAllWindows();

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
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

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

   return array2mat(src_array, src_img.rows, src_img.cols);
}
