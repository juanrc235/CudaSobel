#include "defines.hpp"


// HEADERS
Mat sobel_opencv(Mat img);

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
            img_m.at<uchar>(i, j) = array_img[i][j];
        }
    }

    return img_m;
}

vector<vector<uchar>> conv_gaus_cpu (vector<vector<uchar>> img, int row, int col) {

    char DESP = 1;
    vector<vector<uchar>> dst_img; 
    dst_img.resize(row, vector<uchar>(col, 0) );
    int tmpX = 0, tmpY = 0;
    int tmp = 0;

    for (int x = 1; x < row - 1; x++) {
        for (int y = 1; y < col - 1; y++) {
            
            for  (char i = -DESP; i < KERNEL_DIM - DESP; i++) {
				for (char j = -DESP; j < KERNEL_DIM - DESP; j++) {

					tmp += img[i + x][j + y] * kernel_gaus[i + DESP][j + DESP];
            
				}
			}

            tmp = (int) tmp/9;

            if ( tmp < 0 ) tmp = 0;
	        if ( tmp > 255 ) tmp = 255;  

            dst_img[x][y] = (uchar) tmp;
        }
    }

    return dst_img;
}

vector<vector<uchar>> conv_sobel_cpu (vector<vector<uchar>> img, int row, int col) {

    char DESP = 1;
    vector<vector<uchar>> dst_img; 
    dst_img.resize(row, vector<uchar>(col, 0) );
    int tmpX = 0, tmpY = 0;
    int tmp = 0;

    for (int x = 1; x < row - 1; x++) {
        for (int y = 1; y < col - 1; y++) {
            
            for  (char i = -DESP; i < KERNEL_DIM - DESP; i++) {
				for (char j = -DESP; j < KERNEL_DIM - DESP; j++) {

					tmpX += img[i + x][j + y] * kernel_sx[i + DESP][j + DESP];
                    tmpY += img[i + x][j + y] * kernel_sy[i + DESP][j + DESP];

				}
			}

            tmp = (int)sqrt( tmpX*tmpX + tmpY*tmpY );
           
            if ( tmp < 0 ) tmp = 0;
	        if ( tmp > 255 ) tmp = 255;  

            dst_img[x][y] = tmp;
        }
    }

    return dst_img;
}

int main() {

    VideoCapture cap(0); 
    char c;
   
    // Check if camera opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    printf("Video stream sucesfully opened!\n Press [ESC] to quit.\n");
        
    while(1){

        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            break;
        }


        imshow( "SOBEL" , sobel_opencv(frame) );

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
