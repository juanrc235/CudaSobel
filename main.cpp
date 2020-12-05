#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
//Mat kernelX = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
//Mat kernelY = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

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

/* *
 * 
 * Basic convolution on cpu
 * 
 * IN: img -> grayscale image // kernel -> image kernel to apply
 * OUT: dst_img -> output imgae
 * 

Mat conv_cpu (Mat img, Mat kernel) {

    char KERNEL_DIM = kernel.row;
    char DESP = KERNEL_DIM/2;
    Mat dst_img = Mat( img.rows, img.cols, CV_64FC3, CV_RGB(0,0,0) );

    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            
            for  (char i = -DESP; i < KERNEL_DIM - DESP; i++) {
				for (char j = -DESP; j < KERNEL_DIM - DESP; j++) {
					tmpX += img.at<Vec3d>(x+1,y+1) * kernelX[i + DESP][j + DESP];
				}
			}

            

        }
    }

    return dst_img;
}

Mat sobel_cpu (Mat img) {

    

}
 * */

std::string type2str(int type) {
    std::string r;

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

/**
 * openCV Mat object to std::vector.
 * 
 * The img is in grayscale.
 *
 * @param img the Mat object.
 * @return the vector.
 */
std::vector<uchar> mat2vector (Mat img) {

    std::vector<uint8_t> array(img.rows*img.cols);

    uint8_t* pixelPtr = (uint8_t*)img.data;
    int cn = img.channels();

    for(int i = 0; i < img.rows; ++i) {
        for(int j = 0; j < img.cols; ++j) {
            array.at(img.cols*i + j) = ( pixelPtr[i*img.cols*cn + j*cn + 0] +  
                                         pixelPtr[i*img.cols*cn + j*cn + 1] + 
                                         pixelPtr[i*img.cols*cn + j*cn + 2] ) / 3;
        }
    }

    return array;
}

/**
 * std::vector to openCV Mat object.
 *  
 * @param array the std::vector.
 * @return the Mat object.
 */
Mat vector2mat (std::vector<uchar> img, int row, int col) {

    Mat img_m(col, row, CV_8UC1);
  
    for(int i = 0; i < row; ++i) {
        for(int j = 0; j < col; ++j) {
            img_m.at<uint8_t>(i, j) = img.at(col*i + j);
        }
    }
    return img_m;
}

int main() {

    Mat img = imread(samples::findFile("car.bmp"), IMREAD_COLOR);
    
    if(img.empty()) {
        std::cout << "Could not read the image " << std::endl;
        return 1;
    }
    //resize(img, img, Size(img.cols*0.40f, img.rows*0.40f));

    Mat img_gray;
    
    std::vector<uint8_t> v;
    v = mat2vector(img);    
    Mat img2 = vector2mat(v, img.rows, img.cols);

    printf("Resolution: %dx%d\n", img.rows, img.cols);

    Mat sobel_img = sobel_opencv(img);

    imshow("Original Image", img );
    imshow("Sobel Image", sobel_img);
    imshow("f Image", img2);
    waitKey(0); 

    //imwrite("starry_night.png", img);
   
    return 0;
}
