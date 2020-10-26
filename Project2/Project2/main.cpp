#define _CRT_SECURE_NO_WARNINGS

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <stdio.h>
#include <Windows.h>
#include <math.h>
#define Width 320
#define Height 240
using namespace cv;
using namespace std;

#define pi 3.14


typedef unsigned char uchar;

void Block_histogram(int arr[8][8], float* rst, int num) {
	float angle, magn;
	float h[9] = { 0 };
	float sum = 0;
	int xy[3][3] = { 0 };
	int xfilter[3][3] = { {-1,-1,-1}, {0,0,0}, {1,1,1} };
	int yfilter[3][3] = { {-1,0,1}, {-1,0,1} ,{-1,0,1} };

	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			int xresult = 0, yresult = 0;

			//for exception progress at edge
			int xs = 0, xe = 3, ys = 0, ye = 3;
			if (i == 0)ys = 1;
			if (i == 7)ye = 2;
			if (j == 0)xs = 1;
			if (j == 7)xe = 2;
			
			for (int k = ys; k < ye; k++) {
				for (int l = xs; l < xe; l++) {
					xy[k][l] = arr[i - 1 + k][j - 1 + l];
				}
			}

			for (int k = 0; k < 3; k++) {//inner product with sobel filter
				for (int l = 0; l < 3; l++) {
					xresult += xy[k][l] * xfilter[k][l];
					yresult += xy[k][l] * yfilter[k][l];
				}
			}
			//printf("%d %d//", xresult, yresult);
			if (xresult == 0) {//exception when xresult is 0
				if (yresult == 0) {//when yresult is 0
					angle = 0;
				}
				else {
					angle = 90;
				}
			}
			else {
				angle = (float)atan((double)yresult / (double)xresult) * 180.0 / pi;//phase calc
			}

			magn = sqrt((xresult * xresult) + (yresult * yresult)); //magnitude calc

			if (angle < 0) { //make phase positive
				angle += 180;
			}
			if (angle <= 20) { //devide by phase, sum magnitude
				h[0] += magn;
			}
			else if (20 < angle && angle <= 40) {
				h[1] += magn;
			}
			else if (40 < angle && angle <= 60) {
				h[2] += magn;
			}
			else if (60 < angle && angle <= 80) {
				h[3] += magn;
			}
			else if (80 < angle && angle <= 100) {
				h[4] += magn;
			}
			else if (100 < angle && angle <= 120) {
				h[5] += magn;
			}
			else if (120 < angle && angle <= 140) {
				h[6] += magn;
			}
			else if (140 < angle && angle <= 160) {
				h[7] += magn;
			}
			else if (160 < angle && angle <= 180) {
				h[8] += magn;
			}
		}
	}

	for (int i = 0; i < 9; i++) {//sum value for L2 normalization
		sum += (float)(h[i] * h[i]);
	}
	sum = sqrt(sum);

	for (int i = 0; i < 9; i++) {//L2 normalization
		if (sum == 0) {//exception at sum is 0
			rst[9 * num + i] = 0;
		}
		else {
			rst[9 * num + i] = h[i] / sum;
		}
	}
}

void input_hist(Mat img, float* hog, int xn, int yn) {//devide by block
	int model[8][8];//size of hog window

	int xs = 0, ys = 0;
	for (int i = 0; i < yn; i++) {
		for (int j = 0; j < xn; j++) {
			for (int k = 0; k < 8; k++) {
				for (int l = 0; l < 8; l++) {
					model[k][l] = img.at<uchar>(ys * 4 + k, xs * 4 + l);//insert value to model array
				}
			}
			Block_histogram(model, hog, i * xn + j);

			xs++;
		}
		xs = 0;
		ys++;
	}
}

void compare36x36(Mat ref, Mat rst, float *cmpdata) {
	float max = 0, min = 100000000;
	//Mat pal = Mat::zeros(36, 36, CV_8UC1);

	float **pallete = (float**)malloc(sizeof(float *)*ref.rows);
	for (int i = 0; i < ref.rows; i++) {
		*(pallete + i) = (float*)malloc(sizeof(float)*ref.cols);
	}
	
	for (int i = 0; i < ref.rows; i++) {//initialize pallate
		for (int j = 0; j < ref.cols; j++) {
			pallete[i][j] = 0;
		}
	}

	//int cnt = 1;
	for (int i = 0; i < ref.rows; i++) {
		for (int j = 0; j < ref.cols; j++) {
			int xn, yn, blockn;
			Mat pal = Mat::ones(36, 36, CV_8UC1);
			int xs = 0, xe = 36;
			int ys = 0, ye = 36;
			xn = pal.cols / 4 - 1;//xn=8
			yn = pal.rows / 4 - 1;//yn=8
			blockn = xn * yn;
			float* hog = (float*)malloc(sizeof(float) * blockn * 9);

			//exception handling at edge, make 36x36 window
			if (j < 18)xs = 18 - j;
			if (i < 18)ys = 18 - i;
			if (j > ref.cols - 18)xe = 18 + ref.cols - j;
			if (i > ref.rows - 18)ye = 18 + ref.rows - i;
			for (int k = ys; k < ye; k++) {
				for (int l = xs; l < xe; l++) {
					pal.at<uchar>(k, l) = ref.at<uchar>(i - 18 + k, j - 18 + l);
				}
			}

			input_hist(pal, hog, xn, yn);

			for (int k = 0; k < 576; k++) {//difference between reference and target
				pallete[i][j] += fabs(hog[k] - cmpdata[k]);
			}
		}
	}
	
	for (int i = 0; i < ref.rows; i++) {//get maximum and minimun value for normalization
		for (int j = 0; j < ref.cols; j++) {
			if (pallete[i][j] > max)max = pallete[i][j];
			if (pallete[i][j] < min)min = pallete[i][j];
		}
	}

	for (int i = 0; i < ref.rows; i++) {//normalization
		for (int j = 0; j < ref.cols; j++) {
			rst.at<uchar>(i, j) = (int)(255 * (pallete[i][j] - min) / (max - min)+0.5);//rounds value
		}
	}
	
}

void th_pointing(Mat ref, int th, Mat tar) {
	for (int i = 0; i < ref.rows; i++) {
		for (int j = 0; j < ref.cols; j++) {
			if (ref.at<uchar>(i, j) < th) {//be black if value is small(compare with th)
				tar.at<uchar>(i, j) = 0;
			}
			else {
				tar.at<uchar>(i, j) = ref.at<uchar>(i, j);
			}
		}
	}
}

void main() {
	int xn, yn;
	int blockn;

	Mat asgn3 = imread("face_ref.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	xn = asgn3.cols / 4 - 1;//xn=8
	yn = asgn3.rows / 4 - 1;//yn=8
	blockn = xn * yn;
	float* hog3 = (float*)malloc(sizeof(float) * blockn * 9);//make float array
	input_hist(asgn3, hog3, xn, yn);
	int th;
	printf("input threshold :");
	scanf("%d", &th);
	
	Mat cmp1 = imread("face_tar.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat result = Mat::zeros(cmp1.rows, cmp1.cols, CV_8UC1);
	compare36x36(cmp1, result, hog3);
	Mat point_result = Mat::zeros(cmp1.rows, cmp1.cols, CV_8UC1);
	th_pointing(result, th, point_result);

	imshow("face", asgn3);
	imshow("refimg", cmp1);
	imshow("rstimg", result);
	imshow("result", point_result);
	waitKey(0);
	
}