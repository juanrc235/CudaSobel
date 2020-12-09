#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>

#define KERNEL_DIM 3

using namespace cv;
using namespace std;

double kernel[KERNEL_DIM][KERNEL_DIM] = { {1, 0, -1}, {0, 0, 0}, {-1, 0, 1} };

int kernel_sx[KERNEL_DIM][KERNEL_DIM] = { {1, 0, -1}, 
                                          {2, 0, -2}, 
                                          {1, 0, -1} };

int kernel_sy[KERNEL_DIM][KERNEL_DIM] = { {1, 2, 1}, 
                                          {0, 0, 0}, 
                                          {-1, -2, -1} };

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


vector<vector<uchar>> mat2array (Mat img) {

    vector<vector<uchar>> array_img; 
    array_img.resize(img.rows, vector<uchar>(img.cols, 0) );

    for(int i = 0; i < img.rows; ++i) {
        for(int j = 0; j < img.cols; ++j) {
            array_img[i][j] = ( img.at<Vec3b>(i, j)[0] + 
                                img.at<Vec3b>(i, j)[0] + 
                                img.at<Vec3b>(i, j)[0] ) /3 ;
        }
    }

    return array_img;
}


Mat array2mat ( vector<vector<uchar>> array_img, int row, int col) {

    Mat img_m(row, col, CV_8UC1);
  
    for(int i = 0; i < row; ++i) {
        for(int j = 0; j < col; ++j) {
            img_m.at<uint8_t>(i, j) = array_img[i][j];
        }
    }

    return img_m;
}

vector<vector<uchar>> conv_sobel_cpu (vector<vector<uchar>> img, int row, int col) {

    char DESP = 1;
    vector<vector<uchar>> dst_img; 
    dst_img.resize(row, vector<uchar>(col, 0) );
    int tmpX = 0, tmpY = 0;
    uchar tmp = 0;

    for (int x = 1; x < row - 1; x++) {
        for (int y = 1; y < col - 1; y++) {
            
            for  (char i = -DESP; i < KERNEL_DIM - DESP; i++) {
				for (char j = -DESP; j < KERNEL_DIM - DESP; j++) {

					tmpX += img[i + x][j + y] * kernel_sx[i + DESP][j + DESP];
                    tmpY += img[i + x][j + y] * kernel_sy[i + DESP][j + DESP];

				}
			}

            tmp = sqrt( tmpX*tmpX + tmpY*tmpY );
           
            if ( tmp < 0 ) tmp = 0;
	        if ( tmp > 255 ) tmp = 255;  

            dst_img[x][y] = tmp;
        }
    }

    return dst_img;
}


Mat sobel_cpu (Mat img) {
        
    vector<vector<uchar>> array_img; 
    array_img.resize(img.rows, vector<uchar>(img.cols, 0) );
    vector<vector<uchar>> result; 
    result.resize(img.rows, vector<uchar>(img.cols, 0) );

    array_img = mat2array( img );  

    result = conv_sobel_cpu(array_img, img.rows, img.cols);

    return array2mat( result, img.rows, img.cols );
}

int main() {

    VideoCapture cap(0); 
    char c;
   
    // Check if camera opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    printf("Video stream sucesfully opened! Press ESC to quit\n");
        
    while(1){

        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            break;
        }


        imshow( "SOBEL" , sobel_cpu(frame) );

        if((char)waitKey(25) == 27) {
             break;
        }
       
    }
    
    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();

   
    return 0;
}

/**
 * Mat img = imread(samples::findFile("car.bmp"), IMREAD_COLOR);
    
    if(img.empty()) {
        std::cout << "Could not read the image " << std::endl;
        return 1;
    }
    //resize(img, img, Size(img.cols*0.40f, img.rows*0.40f));

    printf("Resolution: rows:%d cols:%d\n", img.rows, img.cols);
  
    Mat sobel_img = sobel_opencv(img);
    Mat cpu_img = sobel_cpu(img);

    //imshow("Original Image", img );
    imshow("Sobel OpenCV", sobel_img);
    imshow("Sobel CPU", cpu_img);
    waitKey(0); 

    //imwrite("starry_night.png", img);
 * 
 * 
 * */