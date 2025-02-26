//
//  Created by 黃俊穎 on 2020.10.20.
//  Copyright © 2020 Jin. All rights reserved.
//



#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> // resize
#include "opencv2/videoio.hpp" // videocapture
#include <opencv2/imgcodecs.hpp> // imread
#include <opencv2/imgproc.hpp> // phase correlation

#include <opencv2/calib3d.hpp> // find E

#include "opencv2/opencv.hpp"
#include "ceres/ceres.h"

#include <math.h> // pow, round
#include <stdlib.h> // abs rand
#include <iostream>
#include <vector> // vector
#include <iomanip> // number of digits after decimal point
#include <algorithm> // random shuffle

#define POC_LAYERS 5
#define SUB_PIXEL_ITER 5
#define POC_BIG_WINDOW_SIZE 71
#define POC_SMALL_WINDOW_SIZE 71


#define SHOW_POC_BAD_CORNERS 0
#define SHOW_POC_GOOD_CORNERS 0


/* ---------------function prototypes---------------*/

cv::Mat pre_process(cv::Mat *resized_frame, cv::Mat *gray_pre_done);
void poc_pixel(cv::Mat p_img, cv::Mat q_img, std::vector<cv::Point2d> p_corners, std::vector<cv::Point2d> *p_corners_new, std::vector<cv::Point2d> *q_corners, std::vector<std::vector<cv::Point2d>> *corners, int *num_current_corners, std::vector<cv::Point2d> *last_good_corners, std::vector<int> *last_good_idx, std::vector<cv::Point3d> &Xs_tmp);
void poc_sub_pixel(cv::Mat p_img, cv::Mat q_img, std::vector<cv::Point2d> *p_corners, std::vector<cv::Point2d> *q_corners, std::vector<std::vector<cv::Point2d>> *corners, int *num_current_corners, std::vector<cv::Point2d> *last_good_corners, std::vector<int> *last_good_idx, std::vector<cv::Point3d> &Xs_tmp);

/*---------------------------------------------------------------------------*/



