#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

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

int main() {

    Mat img = imread(samples::findFile("car.bmp"), IMREAD_COLOR);
    
    if(img.empty()) {
        std::cout << "Could not read the image " << std::endl;
        return 1;
    }
    //resize(img, img, Size(img.cols*0.40f, img.rows*0.40f));

    printf("Resolution: %dx%d\n", img.cols, img.rows);

    Mat sobel_img = sobel_opencv(img);

    imshow("Original Image", img );
    imshow("Sobel Image", sobel_img);
    waitKey(0); 

    //imwrite("starry_night.png", img);
   
    return 0;
}
