//  This implementation is the revised version as the paper (Robust Skin-Feature Tracking in Free-Hand Smartphone Video, to
//  Enable Localization and Guidance of Hand-Held Clinical Tools).
//  *Uses FFT based scale and rotation invariant phase correlation
//  *implemented by Jin and avoids fftw3
//
//  main.cpp
//  probe_v4
//
//  Created by Chun-Yin Huang on 2020/8/7.
//  Copyright © 2020 Jin. All rights reserved.
//

#include "opencv2/opencv.hpp"

#include "ceres/ceres.h"
#include <math.h> // pow, round
#include <stdlib.h> // abs rand
#include <iostream>
#include <vector> // vector
#include <iomanip> // number of digits after decimal point
#include <algorithm> // random shuffle
#include <time.h> // timing

#define INIT_FRAME 100 //100
#define FRAME_COUNT 1000 //1800
#define FRAME_INTERVAL 20 // 4 16 20 remember to change if FPS=120
#define RECONSTRUCT_INTERVAL 4 // 3 1 4
//#define SCALE_FACTOR 1
#define GFT_CORNER_NUM 500
#define GFT_MIN_DIST 10
#define POC_LAYERS 5
#define SUB_PIXEL_ITER 5
#define POC_BIG_WINDOW_SIZE 71
#define POC_SMALL_WINDOW_SIZE 71

#define XS_THRESHOLD 1
#define ENABLE_SUB_POC 1
#define SHOW_FEATURE_TRACKING 0
#define SHOW_POC_BAD_CORNERS 0
#define SHOW_POC_GOOD_CORNERS 0

/*---------------------------------------------------------------------------*/


/* ---------------below are function prototypes inherit from Jin's code---------------*/

cv::Mat pre_process(cv::Mat *resized_frame, cv::Mat *gray_pre_done);
void poc_pixel(cv::Mat p_img, cv::Mat q_img, std::vector<cv::Point2d> p_corners, std::vector<cv::Point2d> *p_corners_new, std::vector<cv::Point2d> *q_corners, std::vector<std::vector<cv::Point2d>> *corners, int *num_current_corners, std::vector<cv::Point2d> *last_good_corners, std::vector<int> *last_good_idx, std::vector<cv::Point3d> &Xs_tmp);
void poc_sub_pixel(cv::Mat p_img, cv::Mat q_img, std::vector<cv::Point2d> *p_corners, std::vector<cv::Point2d> *q_corners, std::vector<std::vector<cv::Point2d>> *corners, int *num_current_corners, std::vector<cv::Point2d> *last_good_corners, std::vector<int> *last_good_idx, std::vector<cv::Point3d> &Xs_tmp);

/*---------------------------------------------------------------------------*/


/* ---------------below is HEADER functions for reconstruction---------------*/

#ifndef __BUNDLE_ADJUSTMENT__

#define __BUNDLE_ADJUSTMENT__

#include "opencv2/opencv.hpp"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

// Reprojection error for bundle adjustment
struct ReprojectionError
{
    ReprojectionError(const cv::Point2d& _x, double _f, const cv::Point2d& _c) : x(_x), f(_f), c(_c) { }

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const
    {
        // X' = R*X + t
        T X[3];
        ceres::AngleAxisRotatePoint(camera, point, X);
        X[0] += camera[3];
        X[1] += camera[4];
        X[2] += camera[5];

        // x' = K*X'
        T x_p = f * X[0] / X[2] + c.x;
        T y_p = f * X[1] / X[2] + c.y;

        // residual = x - x'
        residuals[0] = T(x.x) - x_p;
        residuals[1] = T(x.y) - y_p;
//        residuals[0] = x_p - T(x.x);
//        residuals[1] = y_p - T(x.y);
        return true;
    }

    static ceres::CostFunction* create(const cv::Point2d& _x, double _f, const cv::Point2d& _c)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(new ReprojectionError(_x, _f, _c)));
    }

private:
    const cv::Point2d x;
    const double f;
    const cv::Point2d c;
};

#endif // End of '__BUNDLE_ADJUSTMENT__'


#ifndef __SFM__
#define __SFM__

#include "opencv2/opencv.hpp"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
//#include "bundle_adjustment.hpp"
#include <unordered_map>

// Reprojection error for bundle adjustment with 7 DOF cameras
// - 7 DOF = 3 DOF rotation + 3 DOF translation + 1 DOF focal length
struct ReprojectionError7DOF
{
    ReprojectionError7DOF(const cv::Point2d& _x, const cv::Point2d& _c) : x(_x), c(_c) { }

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const
    {
        // X' = R*X + t
        T X[3];
        ceres::AngleAxisRotatePoint(camera, point, X);
        X[0] += camera[3];
        X[1] += camera[4];
        X[2] += camera[5];

        // x' = K*X'
        const T& f = camera[6];
        T x_p = f * X[0] / X[2] + c.x;
        T y_p = f * X[1] / X[2] + c.y;

        // residual = x - x'
        residuals[0] = T(x.x) - x_p;
        residuals[1] = T(x.y) - y_p;
        return true;
    }

    static ceres::CostFunction* create(const cv::Point2d& _x, const cv::Point2d& _c)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError7DOF, 2, 7, 3>(new ReprojectionError7DOF(_x, _c)));
    }

private:
    const cv::Point2d x;
    const cv::Point2d c;
};

class SFM
{
public:

    typedef cv::Vec<double, 9> Vec9d;

    typedef std::unordered_map<uint, uint> VisibilityGraph;

    static inline uint genKey(uint cam_idx, uint obs_idx) { return ((cam_idx << 16) + obs_idx); }

    static inline uint getCamIdx(uint key) { return ((key >> 16) & 0xFFFF); }

    static inline uint getObsIdx(uint key) { return (key & 0xFFFF); }

    static bool addCostFunc7DOF(ceres::Problem& problem, const cv::Point3d& X, const cv::Point2d& x, const Vec9d& camera, double loss_width = -1)
    {
        double* _X = (double*)(&(X.x));
        double* _camera = (double*)(&(camera[0]));
        ceres::CostFunction* cost_func = ReprojectionError7DOF::create(x, cv::Point2d(camera[7], camera[8]));
        ceres::LossFunction* loss_func = NULL;
        if (loss_width > 0) loss_func = new ceres::CauchyLoss(loss_width);
        problem.AddResidualBlock(cost_func, loss_func, _camera, _X);
        return true;
    }

    static bool addCostFunc6DOF(ceres::Problem& problem, const cv::Point3d& X, const cv::Point2d& x, const Vec9d& camera, double loss_width = -1)
    {
        double* _X = (double*)(&(X.x));
        double* _camera = (double*)(&(camera[0]));
        ceres::CostFunction* cost_func = ReprojectionError::create(x, camera[6], cv::Point2d(camera[7], camera[8]));
        ceres::LossFunction* loss_func = NULL;
        if (loss_width > 0) loss_func = new ceres::CauchyLoss(loss_width);
        problem.AddResidualBlock(cost_func, loss_func, _camera, _X);
        return true;
    }
};

#endif // End of '__SFM__'

/*---------------------------------------------------------------------------*/

/*------------------- below is Jin's implementation of FFT phase correlation ---------------------*/
#ifndef FFT_REGISTRATION_JIN
#define FFT_REGISTRATION_JIN

#include <complex>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenRowMatrix;
typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > EigenMap;

class FFTRegistration
{
    public:
        FFTRegistration(const int rows, const int cols);
//        ~FFTRegistration();
        
        // registration
        void processImage(const cv::Mat &gray0_, const cv::Mat &gray1_);
        void processImage_(const cv::Mat &gray0_, const cv::Mat &gray1_);
        void getScaleRotation(cv::Point2d rs_shift);
        void rotateAndScale();
        cv::Point2d registerImage(const bool allow_hann, double &response1, double &response2);

        // transform
        void apodize(const cv::Mat &in, cv::Mat &out);
        Eigen::VectorXd getHanningWindow(int size);
        Eigen::MatrixXd getApodizationWindow(int rows, int cols, int radius);

        // dft
        Eigen::MatrixXd getHighPassFilter();
        void cv_dft(const cv::Mat in1, const cv::Mat in2);
        void fftShift(cv::InputOutputArray _out);
        void magSpectrums( cv::InputArray _src, cv::OutputArray _dst);

        

    protected:

        // registration
        int rows_;
        int cols_;
        int log_polar_size_;
        double rotation, scale;
        cv::Point2d translation;

        cv::Mat high_pass_filter_cv;

        cv::Mat gray0;
        cv::Mat gray1;
        cv::Mat log_polar0;
        cv::Mat log_polar1;
        cv::Mat rectified0;


        // transform
        double logBase_;
        cv::Mat appodizationWindow;

        // FFT
        cv::Mat FFT1;
        cv::Mat FFT2;

};


#endif


FFTRegistration::FFTRegistration(const int rows, const int cols) :
    rows_(rows), cols_(cols), log_polar_size_(std::max(rows_, cols_))
{

    // prepare for polar registration
    Eigen::MatrixXd win = getApodizationWindow(rows_, cols_, (int)((0.12)*std::min(rows_, cols_)));
    cv::eigen2cv(win, appodizationWindow);
    appodizationWindow.convertTo(appodizationWindow, CV_32F);

}


//--------------------- registration



void FFTRegistration::processImage_(const cv::Mat &gray0_, const cv::Mat &gray1_)
{
    // * This function has made 2 modifications
    // 1. Uses OpenCV built in log polar warping function
    // 2. Alters the high pass filter(suggested by the arthur) by ones-gaussian filter
    
    gray0_.convertTo(gray0, CV_32F, 1.0/255.0);
    gray1_.convertTo(gray1, CV_32F, 1.0/255.0);

    cv::Mat apodized0, apodized1, FFT1_, FFT2_;

    apodize(gray0, apodized0);
    apodize(gray1, apodized1);
    cv_dft(apodized0, apodized1);
    
    magSpectrums(FFT1, FFT1_);
    magSpectrums(FFT2, FFT2_);
    
    
    fftShift(FFT1_);
    fftShift(FFT2_);
    
    
    cv::Mat gaussian_kernel = cv::getGaussianKernel(POC_BIG_WINDOW_SIZE, -1, CV_32F);
    cv::Mat gaussian_kernel_t;
    cv::transpose(gaussian_kernel, gaussian_kernel_t);
    gaussian_kernel = gaussian_kernel * gaussian_kernel_t;
    FFT1 = FFT1_.mul(gaussian_kernel);
    FFT2 = FFT2_.mul(gaussian_kernel);
    cv::subtract(FFT1_, FFT1, FFT1);
    cv::subtract(FFT2_, FFT2, FFT2);

    cv::Point2f center( (float)FFT1.cols / 2, (float)FFT1.rows / 2 );
    double maxRadius = 0.7*std::min(center.y, center.x);
    
//    // uses OpenCV logPolar fcn ... no big difference compared to warpPolar
//    double M = FFT1.cols / log(maxRadius);
//    cv::logPolar(FFT1, log_polar0, center, M, cv::INTER_LINEAR+cv::WARP_FILL_OUTLIERS);
//    cv::logPolar(FFT2, log_polar1, center, M, cv::INTER_LINEAR+cv::WARP_FILL_OUTLIERS);
    
    // uses OpenCV warpPolar fcn
    cv::warpPolar(FFT1, log_polar0, cv::Size(FFT1.cols, FFT1.rows), center, maxRadius, cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS + cv::WARP_POLAR_LOG);
    cv::warpPolar(FFT2, log_polar1, cv::Size(FFT2.cols, FFT2.rows), center, maxRadius, cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS + cv::WARP_POLAR_LOG);
    

}


void FFTRegistration::getScaleRotation(cv::Point2d rs_shift)
{
    rotation = -M_PI * rs_shift.y  / FFT1.rows;
    rotation = rotation * 180.0 / M_PI;
    rotation += 360.0 / 2.0;
    rotation = fmod(rotation, 360.0);
    rotation -=  360.0 / 2.0;
    rotation = -rotation;
    scale = exp(rs_shift.x);
    scale = 1.0 / scale;
}

void FFTRegistration::rotateAndScale()
{
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point(gray0.cols/2, gray0.rows/2), rotation, scale);
    cv::warpAffine(gray0, rectified0, rotationMatrix, gray0.size());
}

cv::Point2d FFTRegistration::registerImage(const bool allow_hann, double &response1, double &response2)
{
    cv::Point2d rs_shift;
    
    if(allow_hann)
    {
        cv::Mat hann_rs;
        cv::createHanningWindow(hann_rs, cv::Size(log_polar0.cols, log_polar0.rows), CV_32F);
        rs_shift = cv::phaseCorrelate(log_polar0, log_polar1, hann_rs, &response1);
    }
    else
    {
        rs_shift = cv::phaseCorrelate(log_polar0, log_polar1);
    }
    
    getScaleRotation(rs_shift);
    rotateAndScale();

    if(allow_hann)
    {
        cv::Mat hann_t;
        cv::createHanningWindow(hann_t, cv::Size(gray1.cols, gray1.rows), CV_32F);
        translation = cv::phaseCorrelate(rectified0, gray1, hann_t, &response2);
    }
    else
    {
        translation = cv::phaseCorrelate(rectified0, gray1);
    }
    return translation;
}

//--------------------- transform


void FFTRegistration::apodize(const cv::Mat &in, cv::Mat &out)
{
    out = in.mul(appodizationWindow);
}

Eigen::VectorXd FFTRegistration::getHanningWindow(int size)
{
    Eigen::VectorXd window(size);
    for (int i = 0; i < size; i++)
    {
        window(i) = 0.5 - 0.5 * std::cos((2 * M_PI * i)/(size - 1));
    }
    return window;
}

Eigen::MatrixXd FFTRegistration::getApodizationWindow(int rows, int cols, int radius)
{
    Eigen::VectorXd hanning_window = getHanningWindow(radius * 2);

    Eigen::VectorXd row = Eigen::VectorXd::Ones(cols);
    row.segment(0, radius) = hanning_window.segment(0, radius);
    row.segment(cols - radius, radius) = hanning_window.segment(radius, radius);

    Eigen::VectorXd col = Eigen::VectorXd::Ones(rows);
    col.segment(0, radius) = hanning_window.segment(0, radius);
    col.segment(rows - radius, radius) = hanning_window.segment(radius, radius);

    return col * row.transpose();
}


//------------------------ dft

Eigen::MatrixXd FFTRegistration::getHighPassFilter()
{
    Eigen::VectorXd yy = Eigen::VectorXd::LinSpaced(rows_, -M_PI / 2.0, M_PI / 2.0);
    Eigen::VectorXd yy_vec = Eigen::VectorXd::Ones(cols_);
    Eigen::MatrixXd yy_matrix = yy * yy_vec.transpose(); // identical cols, each row is linspace

    Eigen::VectorXd xx = Eigen::VectorXd::LinSpaced(cols_, -M_PI / 2.0, M_PI / 2.0);
    Eigen::VectorXd xx_vec = Eigen::VectorXd::Ones(rows_);
    Eigen::MatrixXd xx_matrix = xx_vec * xx.transpose();

    Eigen::MatrixXd filter = (yy_matrix.cwiseProduct(yy_matrix) + xx_matrix.cwiseProduct(xx_matrix)).cwiseSqrt().array().cos();
    filter = filter.cwiseProduct(filter);
    filter = -filter;
    filter = filter.array() + 1.0;
    return filter;
}

void FFTRegistration::cv_dft(const cv::Mat in1, const cv::Mat in2)
{
//    int M = cv::getOptimalDFTSize(in1.rows);
//    int N = cv::getOptimalDFTSize(in1.cols);
    int M = log_polar_size_;
    int N = log_polar_size_;

    cv::Mat padded1, padded2;

    if(M != in1.rows || N != in1.cols)
    {
        cv::copyMakeBorder(in1, padded1, 0, M - in1.rows, 0, N - in1.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        cv::copyMakeBorder(in2, padded2, 0, M - in2.rows, 0, N - in2.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    }
    else
    {
        padded1 = in1;
        padded2 = in2;
    }

    // execute phase correlation equation
    // Reference: http://en.wikipedia.org/wiki/Phase_correlation
    cv::dft(padded1, FFT1, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(padded2, FFT2, cv::DFT_COMPLEX_OUTPUT);
}

void FFTRegistration::fftShift(cv::InputOutputArray _out)
{
    cv::Mat out = _out.getMat();

    if(out.rows == 1 && out.cols == 1)
    {
        // trivially shifted.
        return;
    }

    std::vector<cv::Mat> planes;
    cv::split(out, planes);

    int xMid = out.cols >> 1;
    int yMid = out.rows >> 1;

    bool is_1d = xMid == 0 || yMid == 0;

    if(is_1d)
    {
        int is_odd = (xMid > 0 && out.cols % 2 == 1) || (yMid > 0 && out.rows % 2 == 1);
        xMid = xMid + yMid;

        for(size_t i = 0; i < planes.size(); i++)
        {
            cv::Mat tmp;
            cv::Mat half0(planes[i], cv::Rect(0, 0, xMid + is_odd, 1));
            cv::Mat half1(planes[i], cv::Rect(xMid + is_odd, 0, xMid, 1));

            half0.copyTo(tmp);
            half1.copyTo(planes[i](cv::Rect(0, 0, xMid, 1)));
            tmp.copyTo(planes[i](cv::Rect(xMid, 0, xMid + is_odd, 1)));
        }
    }
    else
    {
        int isXodd = out.cols % 2 == 1;
        int isYodd = out.rows % 2 == 1;
        for(size_t i = 0; i < planes.size(); i++)
        {
            // perform quadrant swaps...
            cv::Mat q0(planes[i], cv::Rect(0,    0,    xMid + isXodd, yMid + isYodd));
            cv::Mat q1(planes[i], cv::Rect(xMid + isXodd, 0,    xMid, yMid + isYodd));
            cv::Mat q2(planes[i], cv::Rect(0,    yMid + isYodd, xMid + isXodd, yMid));
            cv::Mat q3(planes[i], cv::Rect(xMid + isXodd, yMid + isYodd, xMid, yMid));

            if(!(isXodd || isYodd))
            {
                cv::Mat tmp;
                q0.copyTo(tmp);
                q3.copyTo(q0);
                tmp.copyTo(q3);

                q1.copyTo(tmp);
                q2.copyTo(q1);
                tmp.copyTo(q2);
            }
            else
            {
                cv::Mat tmp0, tmp1, tmp2 ,tmp3;
                q0.copyTo(tmp0);
                q1.copyTo(tmp1);
                q2.copyTo(tmp2);
                q3.copyTo(tmp3);

                tmp0.copyTo(planes[i](cv::Rect(xMid, yMid, xMid + isXodd, yMid + isYodd)));
                tmp3.copyTo(planes[i](cv::Rect(0, 0, xMid, yMid)));

                tmp1.copyTo(planes[i](cv::Rect(0, yMid, xMid, yMid + isYodd)));
                tmp2.copyTo(planes[i](cv::Rect(xMid, 0, xMid + isXodd, yMid)));
            }
        }
    }

    cv::merge(planes, out);
}

void FFTRegistration::magSpectrums( cv::InputArray _src, cv::OutputArray _dst)
{
    cv::Mat src = _src.getMat();
    int depth = src.depth(), cn = src.channels(), type = src.type();
    int rows = src.rows, cols = src.cols;
    int j, k;

    CV_Assert( type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 );

    if(src.depth() == CV_32F)
        _dst.create( src.rows, src.cols, CV_32FC1 );
    else
        _dst.create( src.rows, src.cols, CV_64FC1 );

    cv::Mat dst = _dst.getMat();
    dst.setTo(0);//Mat elements are not equal to zero by default!

    bool is_1d = (rows == 1 || (cols == 1 && src.isContinuous() && dst.isContinuous()));

    if( is_1d )
        cols = cols + rows - 1, rows = 1;

    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    if( depth == CV_32F )
    {
        const float* dataSrc = src.ptr<float>();
        float* dataDst = dst.ptr<float>();

        size_t stepSrc = src.step/sizeof(dataSrc[0]);
        size_t stepDst = dst.step/sizeof(dataDst[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataSrc += cols - 1, dataDst += cols - 1;
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( rows % 2 == 0 )
                    dataDst[(rows-1)*stepDst] = dataSrc[(rows-1)*stepSrc]*dataSrc[(rows-1)*stepSrc];

                for( j = 1; j <= rows - 2; j += 2 )
                {
                    dataDst[j*stepDst] = (float)std::sqrt((double)dataSrc[j*stepSrc]*dataSrc[j*stepSrc] +
                                                          (double)dataSrc[(j+1)*stepSrc]*dataSrc[(j+1)*stepSrc]);
                }

                if( k == 1 )
                    dataSrc -= cols - 1, dataDst -= cols - 1;
            }
        }

        for( ; rows--; dataSrc += stepSrc, dataDst += stepDst )
        {
            if( is_1d && cn == 1 )
            {
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( cols % 2 == 0 )
                    dataDst[j1] = dataSrc[j1]*dataSrc[j1];
            }

            for( j = j0; j < j1; j += 2 )
            {
                dataDst[j] = (float)std::sqrt((double)dataSrc[j]*dataSrc[j] + (double)dataSrc[j+1]*dataSrc[j+1]);
            }
        }
    }
    else
    {
        const double* dataSrc = src.ptr<double>();
        double* dataDst = dst.ptr<double>();

        size_t stepSrc = src.step/sizeof(dataSrc[0]);
        size_t stepDst = dst.step/sizeof(dataDst[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataSrc += cols - 1, dataDst += cols - 1;
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( rows % 2 == 0 )
                    dataDst[(rows-1)*stepDst] = dataSrc[(rows-1)*stepSrc]*dataSrc[(rows-1)*stepSrc];

                for( j = 1; j <= rows - 2; j += 2 )
                {
                    dataDst[j*stepDst] = std::sqrt(dataSrc[j*stepSrc]*dataSrc[j*stepSrc] +
                                                   dataSrc[(j+1)*stepSrc]*dataSrc[(j+1)*stepSrc]);
                }

                if( k == 1 )
                    dataSrc -= cols - 1, dataDst -= cols - 1;
            }
        }

        for( ; rows--; dataSrc += stepSrc, dataDst += stepDst )
        {
            if( is_1d && cn == 1 )
            {
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( cols % 2 == 0 )
                    dataDst[j1] = dataSrc[j1]*dataSrc[j1];
            }

            for( j = j0; j < j1; j += 2 )
            {
                dataDst[j] = std::sqrt(dataSrc[j]*dataSrc[j] + dataSrc[j+1]*dataSrc[j+1]);
            }
        }
    }
}
/*-----------------------------------------------------------------------------------------*/


int main()
{
    // intrinsic mat
    /********* new calibration result ( first 1/3 +2 )
     Focal Length:          fc = [ 1610.51324   1613.05183 ] +/- [ 11.73317   11.55720 ]
     Principal point:       cc = [ 954.94718   548.71677 ] +/- [ 7.85361   9.57045 ]
     Skew:             alpha_c = [ 0.00000 ] +/- [ 0.00000  ]   => angle of pixel axes = 90.00000 +/- 0.00000 degrees
     Distortion:            kc = [ 0.18660   -0.63251   0.01228   0.00237  0.00000 ] +/- [ 0.02503   0.15011   0.00269   0.00211  0.00000 ]
     Pixel error:          err = [ 0.53526   0.37291 ]
     ***********/
    
    clock_t time_begin = clock();
    
    double f_init =  1610.51324, cx_init = 954.94718, cy_init = 548.71677, Z_init = 0.5, Z_limit = 100, ba_loss_width = 9; // Negative 'loss_width' makes BA not to use a loss function.
    int min_inlier_num = 200, ba_num_iter = 200; // Negative 'ba_num_iter' uses the default value for BA minimization
    
    
    // video input
//    char input_name[] = "/Users/jin/Desktop/CMU/BIG/probe/slowmo/dist_1.mov";
//    char input_name[] = "/Users/jin/Desktop/CMU/BIG/probe/apriltest.MOV";
    char input_name[] = "./scan_tag_8.mov";
    cv::VideoCapture cap;
    cap.open(input_name + cv::CAP_ANY);
    std::cout<<cap.get(cv::CAP_PROP_FRAME_COUNT)<<std::endl;
    if(!cap.isOpened()){
        std::cout<<"INVALID VIDEO PATH!!"<<std::endl;
        return -1;
    }

    
    // set least frame number -- 2mn>=11m+3n-15
    int least_frame = ceil(double(11*(RECONSTRUCT_INTERVAL+1)-15)/double(2*(RECONSTRUCT_INTERVAL+1)-3));
    

    // corner template
    std::vector<cv::Point2d> good_corners;
    std::vector<std::unordered_map<int, int>> global_feature_map;
    std::vector<std::vector<std::vector<cv::Point2d>>> global_features;
    
    
    // initialize cameras
    std::vector<std::vector<SFM::Vec9d>> cameras;

    
    // Global 3D point sets
    std::vector<cv::Point3d> Xs_tmp;
    std::vector<cv::Point3d> Xs;
    std::vector<int> Xs_count;
    std::vector<cv::Vec3b> Xs_rgb;
    
    
    // initialize frame count
    int count = INIT_FRAME;
    
    while(cap.isOpened() && count<FRAME_COUNT)
    {

        // good feature to track on first image
        
        
        //read frame0
        cv::Mat frame0, frame1;
        cap.set(cv::CAP_PROP_POS_FRAMES,count);
        cap.read(frame0);

        
        // pre process
        cv::Mat color0 = cv::Mat::zeros(frame0.rows, frame0.cols, CV_8UC3);
        cv::Mat color1 = cv::Mat::zeros(frame0.rows, frame0.cols, CV_8UC3);
        frame0.copyTo(color0);
        cv::Mat gray0 = cv::Mat::zeros(frame0.rows, frame0.cols, CV_8U);
        cv::Mat mask = pre_process(&color0, &gray0);
//        cv::Mat mask_;
//        resize(mask, mask_, cv::Size(float(mask.cols)*0.5, float(mask.rows)*0.5));
//        imshow("mask", mask_);


        // make ROI for good features to track smaller
        cv::Mat kernel = cv::Mat::ones(5, 5, CV_8U);
        cv::erode(mask, mask, kernel);

        
        // set idx for good corners
        std::vector<int> good_corner_idx;
        for(int i=0; i<good_corners.size(); i++){
            good_corner_idx.push_back(i);
        }
        
        
        // add corner constraints to the mask
        if(good_corners.size()!=0){
            cv::Mat kernel1 = cv::Mat::ones(11, 11, CV_8U);
            cv::Mat occlude = cv::Mat::zeros(frame0.rows, frame0.cols, CV_8U);
            for( int i = 0; i < good_corners.size(); i++ )
            {
                occlude.at<uchar>((int)good_corners[i].y, (int)good_corners[i].x) = 255;
            }
            dilate(occlude, occlude, kernel1);
            bitwise_and(occlude, mask, occlude);
            bitwise_xor(mask, occlude, mask);
        }
        
        
        // good features to track for first frame in this set
        std::vector<cv::Point2d> corner0;
        int max_corners = GFT_CORNER_NUM - (int)good_corners.size();
        double quality_level = 0.01;
        double min_distance = GFT_MIN_DIST;
        int block_size = 3;
        bool use_harris = false;
        double k = 0.04;
        cv::goodFeaturesToTrack(gray0,
                                corner0,
                                max_corners,
                                quality_level,
                                min_distance,
                                mask,
                                block_size,
                                use_harris,
                                k);
        
        
        // add back last new corners
        if(good_corners.size()!=0){
            for( int i = 0; i < good_corners.size(); i++ )
            {
                corner0.push_back(good_corners[i]);
            }
        }
        
        
        // now lets go into video frames
        std::vector<std::vector<cv::Point2d>> corners;
        corners.push_back(corner0);
        

        // do reconstruction in specified number of frames (a set)
        for(int rid=0; rid<RECONSTRUCT_INTERVAL; rid++){

            std::cout<<"dealing with frame# "<<count+rid*FRAME_INTERVAL<<" and #"<<count+(rid+1)*FRAME_INTERVAL<<std::endl;

            
            // read consecutive frames
            cap.set(cv::CAP_PROP_POS_FRAMES,count+rid*FRAME_INTERVAL);
            cap.read(frame0);
            cap.set(cv::CAP_PROP_POS_FRAMES,count+(rid+1)*FRAME_INTERVAL);
            cap.read(frame1);

            
            // check if read video succeeded
            if ((frame1).empty()) {
                std::cerr << "ERROR! blank frame grabbed\n";
                break;
            }


            // pre process
            frame0.copyTo(color0);
            frame1.copyTo(color1);
            cv::Mat gray0 = cv::Mat::zeros(frame0.rows, frame0.cols, CV_8U);
            cv::Mat gray1 = cv::Mat::zeros(frame1.rows, frame1.cols, CV_8U);
            cv::Mat mask0 = pre_process(&color0, &gray0);
            cv::Mat mask1 = pre_process(&color1, &gray1);


            // POC
            std::vector<cv::Point2d> p_corner, q_corner;
            poc_pixel(gray0, gray1, corners[rid], &p_corner, &q_corner, &corners, &max_corners, &good_corners, &good_corner_idx, Xs_tmp);
            if(ENABLE_SUB_POC){
                poc_sub_pixel(gray0, gray1, &p_corner, &q_corner, &corners, &max_corners, &good_corners, &good_corner_idx, Xs_tmp);
            }
            corners.push_back(q_corner);


            
            // show feature tracking result
            if(SHOW_FEATURE_TRACKING){
                cv::Mat outout;
                color1.copyTo(outout);
                for( int i = 0; i < q_corner.size(); i++ )
                {
                    if(i < max_corners)
                        circle( outout, q_corner[i], 5, cv::Scalar(255, 255, 0), -1);
                    else
                        circle( outout, q_corner[i], 5, cv::Scalar(255, 0, 0), -1);
                }
                resize(outout, outout, cv::Size(float(color1.cols)*0.5, float(color1.rows)*0.5));
                imshow("q_corners", outout);
                cv::waitKey();
            }
            
            // find instant camera pose-------------------------
            if(count > INIT_FRAME+300){
                cv::Mat_<double> cameraMatrix(3, 3);
                cameraMatrix<<f_init, 0, cx_init, 0, f_init, cy_init, 0, 0, 1;
                cv::Matx33d R;
                cv::Vec3d rvec, tvec;
                std::vector<cv::Point2d> q_corner_tmp;
                auto idx_tmp = good_corners.size()-Xs_tmp.size();
                for(auto ii=max_corners+idx_tmp; ii<q_corner.size(); ii++)
                    q_corner_tmp.push_back(q_corner[ii]);
                bool whatsthis = cv::solvePnPRansac(Xs_tmp, q_corner_tmp, cameraMatrix, cv::noArray(), rvec, tvec);
                cv::Rodrigues(rvec, R);
                cv::Vec3d t(tvec[0], tvec[1], tvec[2]);
                cv::Vec3d p = -R.t() * t;
            }
            // ---------------------------------------
        }
    
        
        // store the good corners of #(RECONSTRUCT_INTERVAL-1) for next frame intervals
        good_corners = corners[RECONSTRUCT_INTERVAL];
        
        // repeat last frame from the last set
        count+=(RECONSTRUCT_INTERVAL)*FRAME_INTERVAL;
        
    
        // Append Global feature corners
        global_features.push_back(corners);
        
    
        // Append cameras (rotation, translation, intrinsic parameters)
        std::vector<SFM::Vec9d> local_cameras(corners.size(), SFM::Vec9d(0, 0, 0, 0, 0, 0, f_init, cx_init, cy_init));
        
        
        // build local 3D points
        std::vector<cv::Point3d> local_Xs;
        ceres::Problem local_ba;
        for (auto idx = 0; idx<corners[RECONSTRUCT_INTERVAL].size(); idx++)
        {
            // for new 3D points
            if(idx < max_corners){
                local_Xs.push_back(cv::Point3d(0, 0, Z_init));
            }
        }
        // for all images in this set
        for (auto c_idx = 0; c_idx<corners.size(); c_idx++){
            // for every feature points
            for(auto idx = 0; idx<corners[c_idx].size(); idx++){
                // for new 3D points
                if(idx < max_corners){
                    SFM::addCostFunc6DOF(local_ba, local_Xs[idx], corners[c_idx][idx], local_cameras[c_idx], ba_loss_width); // Xs[idx+??]
                }
            }
        }
        ceres::Solver::Options local_options;
        local_options.linear_solver_type = ceres::DENSE_SCHUR;
        if (ba_num_iter > 0) local_options.max_num_iterations = ba_num_iter;
        //--------
        local_options.inner_iteration_tolerance = 1e-7;
        local_options.preconditioner_type = ceres::SCHUR_JACOBI;
        local_options.use_explicit_schur_complement = true;
        //--------
        local_options.num_threads = 8;
        local_options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary local_summary;
        local_summary.termination_type = ceres::NO_CONVERGENCE;
        ceres::Solve(local_options, &local_ba, &local_summary);
        std::cout << local_summary.FullReport() << std::endl;


        std::cout<<"-----------There are "<<max_corners<<" new 3D points-----------\n"<<std::endl;
        
        // append the new cameras
        cameras.push_back(local_cameras);
        
        
        // Append 3D points, global feature index
        
        std::unordered_map<int, int> local_feature_map;
        for (auto idx = 0; idx<corners[RECONSTRUCT_INTERVAL].size(); idx++)
        {
            // for new 3D points
            if(idx < max_corners){
                local_feature_map[idx] = int(Xs.size());
                // use pre-built 3D points
                Xs.push_back(local_Xs[idx]);
                Xs_count.push_back(1);
                Xs_rgb.push_back(color1.at<cv::Vec3b>(corners[RECONSTRUCT_INTERVAL][idx].y, corners[RECONSTRUCT_INTERVAL][idx].x));
            }
            // for existed 3D points
            else{
                int Xs_idx = global_feature_map.back()[good_corner_idx[idx - max_corners]];
                local_feature_map[idx] = Xs_idx;
                Xs_count[Xs_idx] += 1;
            }
        }
        global_feature_map.push_back(local_feature_map);
        
        
        
        // refine for each set
        if(count > INIT_FRAME+300){
            ceres::Problem ba;
            // for all images in this set
            for(auto set_idx=0; set_idx<global_features.size(); set_idx++){
                for (auto c_idx = 0; c_idx<global_features[set_idx].size(); c_idx++){
                    // for every feature points
                    for(auto idx = 0; idx<global_features[set_idx][c_idx].size(); idx++){
                        int Xs_idx = global_feature_map[set_idx][idx];
                        // for those 3d points shown more than two times
                        if(Xs_count[Xs_idx] > XS_THRESHOLD)
                            SFM::addCostFunc6DOF(ba, Xs[Xs_idx], global_features[set_idx][c_idx][idx], cameras[set_idx][c_idx], ba_loss_width); // Xs[idx+??]
                    }
                }
            }

            // 4) Optimize camera pose and 3D points together (bundle adjustment)
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_SCHUR;
            if (ba_num_iter > 0) options.max_num_iterations = ba_num_iter;
            //--------
            options.inner_iteration_tolerance = 1e-7;
            options.preconditioner_type = ceres::SCHUR_JACOBI;
            options.use_explicit_schur_complement = true;
            //--------
            options.num_threads = 8;
            options.minimizer_progress_to_stdout = true;
            ceres::Solver::Summary summary;
            summary.termination_type = ceres::NO_CONVERGENCE;
            ceres::Solve(options, &ba, &summary);
            std::cout << summary.FullReport() << std::endl;
            
            // save for instant camera pose estimation
            Xs_tmp.clear();
            for (auto idx = max_corners; idx<good_corners.size(); idx++)
            {
                int Xs_idx = global_feature_map[global_feature_map.size()-2][good_corner_idx[idx - max_corners]];
                Xs_tmp.push_back(Xs[Xs_idx]);
            }
        }
        std::cout<<"-----------There are total "<<Xs.size()<<" 3D points-----------\n"<<std::endl;
        
        
    }
    
    
    // delete those who show up less than 2 times
    int num_erased = 0;
    for(auto i=0; i<Xs_count.size(); i++){
        if(Xs_count[i] <= XS_THRESHOLD){
            Xs.erase(Xs.begin() + i - num_erased);
            num_erased++;
        }
    }
    std::cout<<"Final number of 3D points: "<<Xs.size()<<std::endl;
    

    for(size_t s = 0; s<cameras.size(); s++){
        printf("set #: %zd\n", s);
        for (size_t j = 0; j < cameras[s].size(); j++){
            printf("    3DV Tutorial: Camera %zd's (f, cx, cy) = (%.3f, %.1f, %.1f)\n", j, cameras[s][j][6], cameras[s][j][7], cameras[s][j][8]);
        }
    }
    
    // Store the 3D points to an XYZ file
    FILE* fpts = fopen("./probe_sfm_points_test.xyz", "wt");
//    FILE* fpts = fopen(pointsname, "wt");
    if (fpts == NULL) return -1;
    for (size_t i = 0; i < Xs.size(); i++)
    {
        if (Xs[i].z > -Z_limit && Xs[i].z < Z_limit)// && !is_noisy[i])
//        if (!is_noisy[i])
            fprintf(fpts, "%f %f %f %d %d %d\n", Xs[i].x, Xs[i].y, Xs[i].z, Xs_rgb[i][2], Xs_rgb[i][1], Xs_rgb[i][0]); // Format: x, y, z, R, G, B
    }
    fclose(fpts);

    // Store the camera poses to an XYZ file
    FILE* fcam = fopen("./probe_sfm_cameras_test.xyz", "wt");
    if (fcam == NULL) return -1;
    for(size_t s = 0; s<cameras.size(); s++){
        for (size_t j = 0; j < cameras[s].size(); j++)
        {
            cv::Vec3d rvec(cameras[s][j][0], cameras[s][j][1], cameras[s][j][2]), t(cameras[s][j][3], cameras[s][j][4], cameras[s][j][5]);
            cv::Matx33d R;
            cv::Rodrigues(rvec, R);
            cv::Vec3d p = -R.t() * t;
            fprintf(fcam, "%f %f %f %f %f %f\n", p[0], p[1], p[2], R.t()(0, 2), R.t()(1, 2), R.t()(2, 2)); // Format: x, y, z, n_x, n_y, n_z
        }
    }
    fclose(fcam);
    
    cap.release();
    
    std::cout << "Total time elapsed: " << (clock() - time_begin) * 1.0 / CLOCKS_PER_SEC << "seconds" << std::endl;
    
    return 0;
}

/*---------------------------------------------------------------------------*/



/* ---------------below are functions inherit from Jin's code---------------*/
cv::Mat pre_process(cv::Mat *resized_frame, cv::Mat *gray_pre_done){

    // resize and gray
    //    resize(frame, *resized_frame, Size(float(frame.cols)*scale_factor, float(frame.rows)*scale_factor));
    //    resize(frame, *resized_frame, Size(640, 480));
    cv::Mat gray;
    cvtColor(*resized_frame, gray, cv::COLOR_RGB2GRAY);

    // find blue-ish pizels and blur them
    cv::Mat mask, mask_inv;
    cv::Mat gray_not_blue;
    inRange(*resized_frame, cv::Scalar(0, 0, 0), cv::Scalar(255, 150, 150), mask);
    cv::Mat kernel = cv::Mat::ones(11, 11, CV_8U);
    // opening
    erode(mask, mask, kernel);
    dilate(mask, mask, kernel);
    bitwise_not(mask, mask_inv);
    bitwise_and(gray, gray, gray_not_blue, mask_inv);

    // CLAHE
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2);
    cv::Mat gray_notblue_clahe;
    clahe->apply(gray_not_blue, gray_notblue_clahe);

    // blur
    int blur_size = 5;
    cv::Mat gray_blur, gray_blue_blur;
    blur( gray, gray_blur, cv::Size( blur_size, blur_size ), cv::Point(-1,-1) );
    bitwise_and(gray_blur, gray_blur, gray_blue_blur, mask);
    add(gray_blue_blur, gray_notblue_clahe, *gray_pre_done);
//    *gray_pre_done = gray_notblue_clahe;

    return mask_inv;
}


void poc_pixel(cv::Mat p_img, cv::Mat q_img, std::vector<cv::Point2d> p_corners, std::vector<cv::Point2d> *p_corners_new, std::vector<cv::Point2d> *q_corners, std::vector<std::vector<cv::Point2d>> *corners, int *num_current_corners, std::vector<cv::Point2d> *last_good_corners, std::vector<int> *last_good_idx, std::vector<cv::Point3d> &Xs_tmp){

    // get img width and height

    int img_w = p_img.cols;
    int img_h = p_img.rows;


    // create image lists and corner lists

    //image lists
    cv::Mat p_img_list[POC_LAYERS+1];
    cv::Mat q_img_list[POC_LAYERS+1];
    p_img.copyTo(p_img_list[0]);
    q_img.copyTo(q_img_list[0]);

    //corner lists
    std::vector<cv::Point2d> p_corner_list[POC_LAYERS+1];
    p_corner_list[0] = p_corners;


    // create image pyramid and corresponding p corners

    int p_img_h, p_img_w;
    cv::Point2d corner_tmp;
    for(int i=0; i<POC_LAYERS; i++){

        //create image pyramid
        pyrDown(p_img, p_img);
        pyrDown(q_img, q_img);
        p_img.copyTo(p_img_list[i+1]);
        q_img.copyTo(q_img_list[i+1]);

        //get new image height and width
        p_img_h = p_img.rows;
        p_img_w = p_img.cols;

        //find corresponding p corners
        for(int j=0; j<p_corners.size(); j++){
            corner_tmp.x = (p_corners[j].x-float(img_w)/2.)/(pow(2, double(i+1))) + float(p_img_w)/2.;
            corner_tmp.y = (p_corners[j].y-float(img_h)/2.)/(pow(2, double(i+1))) + float(p_img_h)/2.;

            //sanity check
            if(corner_tmp.x<0){
                std::cout<<i<<" "<<corner_tmp.x<<std::endl;
                std::cout<<"warning while finding corners for image pyramid"<<std::endl;
                corner_tmp.x = 0;
            }
            if(corner_tmp.y<0){
                std::cout<<j<<" "<<corner_tmp.y<<std::endl;
                std::cout<<"warning while finding corners for image pyramid"<<std::endl;
                corner_tmp.y = 0;
            }

            // append corner
            p_corner_list[i+1].push_back(corner_tmp);

        }

    }


    // find q_corners

    //copy last layer of p_corner to q_corner
    std::vector<cv::Point2d> q_corner_list[POC_LAYERS+1];
    q_corner_list[POC_LAYERS] = p_corner_list[POC_LAYERS];

    // variables
    int p_x, p_y, q_x, q_y;
    double response=0;
    std::vector<int> del_list;

    // POC window size for pixel precision
    int win_left = (POC_BIG_WINDOW_SIZE-1)/2;
    int win_right = (POC_BIG_WINDOW_SIZE+1)/2;

    // for pixel alignment
    for(int i=0; i<POC_LAYERS; i++){

        // image crop tmp
        cv::Mat p_img_tmp(POC_BIG_WINDOW_SIZE, POC_BIG_WINDOW_SIZE , CV_8U);
        cv::Mat q_img_tmp(POC_BIG_WINDOW_SIZE, POC_BIG_WINDOW_SIZE , CV_8U);

        copyMakeBorder( p_img_list[POC_LAYERS-i-1], p_img, win_left, win_right, win_left, win_right, cv::BORDER_CONSTANT);
        copyMakeBorder( q_img_list[POC_LAYERS-i-1], q_img, win_left, win_right, win_left, win_right, cv::BORDER_CONSTANT);

        //starts from the smallest layer
        for(int j=0; j<p_corner_list[POC_LAYERS-i-1].size(); j++){

            p_x = p_corner_list[POC_LAYERS-i-1][j].x;
            p_y = p_corner_list[POC_LAYERS-i-1][j].y;
            q_x = (q_corner_list[POC_LAYERS-i][j].x - float(q_img_list[POC_LAYERS-i].cols)/2.)*2 + float(q_img_list[POC_LAYERS-i-1].cols)/2.;
            q_y = (q_corner_list[POC_LAYERS-i][j].y - float(q_img_list[POC_LAYERS-i].rows)/2.)*2 + float(q_img_list[POC_LAYERS-i-1].rows)/2.;

            // check if the corner is out of bound
            if(q_x<0)
                q_x=0;
            else if(q_x+POC_BIG_WINDOW_SIZE>=q_img.cols)
                q_x=q_img.cols-POC_BIG_WINDOW_SIZE;
            if(q_y<0)
                q_y=0;
            else if(q_y+POC_BIG_WINDOW_SIZE>=q_img.rows)
                q_y=q_img.rows-POC_BIG_WINDOW_SIZE;
            if(p_x<0)
                p_x=0;
            else if(p_x+POC_BIG_WINDOW_SIZE>=p_img.cols)
                p_x=p_img.cols-POC_BIG_WINDOW_SIZE;
            if(p_y<0)
                p_y=0;
            else if(p_y+POC_BIG_WINDOW_SIZE>=p_img.rows)
                p_y=p_img.rows-POC_BIG_WINDOW_SIZE;

            //crop
            p_img(cv::Rect(p_x,p_y,POC_BIG_WINDOW_SIZE,POC_BIG_WINDOW_SIZE)).copyTo(p_img_tmp);
            q_img(cv::Rect(q_x,q_y,POC_BIG_WINDOW_SIZE,POC_BIG_WINDOW_SIZE)).copyTo(q_img_tmp);
            p_img_tmp.convertTo(p_img_tmp, CV_32F);
            q_img_tmp.convertTo(q_img_tmp, CV_32F);

            // phase correlate
            cv::Mat hann;
            createHanningWindow(hann, cv::Size(q_img_tmp.cols, q_img_tmp.rows), CV_32F);
            corner_tmp = phaseCorrelate(p_img_tmp, q_img_tmp, hann, &response);


            // for those bad matches in largest scale, apply good features to track on this region
            if(response<0.9 && i==POC_LAYERS-1)
                del_list.push_back(j);

            // show corners
            if(response<0.9 && i==POC_LAYERS-1 && SHOW_POC_BAD_CORNERS){
                std::cout<<response<<std::endl;

                cv::Mat p_img_, q_img_;
                p_img_list[POC_LAYERS-i-1].copyTo(p_img_);
                q_img_list[POC_LAYERS-i-1].copyTo(q_img_);
                circle( p_img_, cv::Point2d(p_x, p_y), 5, 255, -1);
                circle( q_img_, cv::Point2d(q_x, q_y), 5, 255, -1);

                imshow("p", p_img_);
                imshow("q", q_img_);

                p_img_tmp.convertTo(p_img_tmp, CV_8U);
                q_img_tmp.convertTo(q_img_tmp, CV_8U);
                imshow("p patch", p_img_tmp);
                imshow("q patch", q_img_tmp);
                cv::waitKey();
            }
            else if(response>=0.9 && i==POC_LAYERS-1 && SHOW_POC_GOOD_CORNERS){
                std::cout<<response<<std::endl;

                cv::Mat p_img_, q_img_;
                p_img_list[POC_LAYERS-i-1].copyTo(p_img_);
                q_img_list[POC_LAYERS-i-1].copyTo(q_img_);
                circle( p_img_, cv::Point2d(p_x, p_y), 5, 255, -1);
                circle( q_img_, cv::Point2d(q_x, q_y), 5, 255, -1);

                imshow("p", p_img_);
                imshow("q", q_img_);

                p_img_tmp.convertTo(p_img_tmp, CV_8U);
                q_img_tmp.convertTo(q_img_tmp, CV_8U);
                imshow("p patch", p_img_tmp);
                imshow("q patch", q_img_tmp);
                cv::waitKey();
            }

            // append
            cv::Point2d tmp;
            tmp.x = round(q_x+corner_tmp.x); tmp.y = round(q_y+corner_tmp.y);
            q_corner_list[POC_LAYERS-i-1].push_back(tmp);
        }
    }
    // delete those corners too far away
    //    cout<<del_list.size()<<endl;
    for(int idx=0; idx<del_list.size(); idx++){
        p_corner_list[0].erase(p_corner_list[0].begin()+del_list[idx]-idx);
        q_corner_list[0].erase(q_corner_list[0].begin()+del_list[idx]-idx);
        for(int c=0; c<(*corners).size(); c++){
            (*corners)[c].erase((*corners)[c].begin()+del_list[idx]-idx);
        }
        // if delete points from previous corners
        if((*last_good_corners).size()!=0 && del_list[idx]-idx>=(*num_current_corners)){
            auto idx_tmp = (*last_good_corners).size()-Xs_tmp.size();
            (*last_good_corners).erase((*last_good_corners).begin()+del_list[idx]-idx-(*num_current_corners));
            (*last_good_idx).erase((*last_good_idx).begin()+del_list[idx]-idx-(*num_current_corners));
            if(!Xs_tmp.empty() && del_list[idx]-idx-(*num_current_corners)>idx_tmp)
                Xs_tmp.erase(Xs_tmp.begin()+del_list[idx]-idx-(*num_current_corners)-idx_tmp);
        }
        // if delete points from current corners
        else{
            (*num_current_corners) = (*num_current_corners)-1;
        }
    }

    // assign values to p_corners_new and q_corner
    *q_corners = q_corner_list[0];
    *p_corners_new = p_corner_list[0];
    std::cout<<"POC Pixel level done"<<std::endl;

    return;
}


void poc_sub_pixel(cv::Mat p_img, cv::Mat q_img, std::vector<cv::Point2d> *p_corners, std::vector<cv::Point2d> *q_corners, std::vector<std::vector<cv::Point2d>> *corners, int *num_current_corners, std::vector<cv::Point2d> *last_good_corners, std::vector<int> *last_good_idx, std::vector<cv::Point3d> &Xs_tmp){
    //    std::cout << std::setprecision(10) << std::fixed;

    // parameters
    int p_x, p_y, q_x, q_y;
//    double response=0;
    double response1=0, response2=0;

    // hanning window
    cv::Mat hann_p, hann_q;
    createHanningWindow(hann_p, cv::Size(POC_BIG_WINDOW_SIZE, POC_BIG_WINDOW_SIZE), CV_32F);


//    // image crop tmp
//    cv::Mat p_img_tmp(POC_BIG_WINDOW_SIZE, POC_BIG_WINDOW_SIZE , CV_8U);
//    cv::Mat q_img_tmp(POC_BIG_WINDOW_SIZE, POC_BIG_WINDOW_SIZE , CV_8U);

    // corner tmp
    cv::Point2d tmp;
    tmp.x=0; tmp.y=0;
    std::vector<cv::Point2d> corner_tmp((*q_corners).size());
    for(int i=0; i<(*q_corners).size();i++){
        corner_tmp[i].x=0; corner_tmp[i].y=0;
    }


    // window size for sub-pixel precision
    int win_left = (POC_BIG_WINDOW_SIZE-1)/2;
    int win_right = (POC_BIG_WINDOW_SIZE+1)/2;

    copyMakeBorder( p_img, p_img, win_left, win_right, win_left, win_right, cv::BORDER_CONSTANT);
    copyMakeBorder( q_img, q_img, win_left, win_right, win_left, win_right, cv::BORDER_CONSTANT);

    // declare the registration class
    FFTRegistration my_fmt(POC_BIG_WINDOW_SIZE, POC_BIG_WINDOW_SIZE);
    

    // iteratively update corner shift
    std::vector<int> del_list;
    for(int i=0; i<SUB_PIXEL_ITER; i++){

        // for all the corners
        for(int j=0; j<(*p_corners).size(); j++){

            //assign corner values to tmp
            p_x = (*p_corners)[j].x;
            p_y = (*p_corners)[j].y;
            q_x = (*q_corners)[j].x;
            q_y = (*q_corners)[j].y;

            // check if the corner is out of bound
            if(q_x<0)
                q_x=0;
            else if(q_x+POC_BIG_WINDOW_SIZE>q_img.cols)
                q_x=q_img.cols-POC_BIG_WINDOW_SIZE;
            if(q_y<0)
                q_y=0;
            else if(q_y+POC_BIG_WINDOW_SIZE>q_img.rows)
                q_y=q_img.rows-POC_BIG_WINDOW_SIZE;
            if(p_x<0)
                p_x=0;
            else if(p_x+POC_BIG_WINDOW_SIZE>p_img.cols)
                p_x=p_img.cols-POC_BIG_WINDOW_SIZE;
            if(p_y<0)
                p_y=0;
            else if(p_y+POC_BIG_WINDOW_SIZE>p_img.rows)
                p_y=p_img.rows-POC_BIG_WINDOW_SIZE;

            
            //crop and convert type
            cv::Mat p_img_tmp(POC_BIG_WINDOW_SIZE, POC_BIG_WINDOW_SIZE , CV_8U);
            cv::Mat q_img_tmp(POC_BIG_WINDOW_SIZE, POC_BIG_WINDOW_SIZE , CV_8U);
            p_img(cv::Rect(p_x,p_y,POC_BIG_WINDOW_SIZE,POC_BIG_WINDOW_SIZE)).copyTo(p_img_tmp);
            q_img(cv::Rect(q_x,q_y,POC_BIG_WINDOW_SIZE,POC_BIG_WINDOW_SIZE)).copyTo(q_img_tmp);
            
            
            
            // fmt registartion
            my_fmt.processImage_(p_img_tmp, q_img_tmp);
            tmp = my_fmt.registerImage(true, response1, response2);

            // phase correlate
//            p_img_tmp.convertTo(p_img_tmp, CV_32F);
//            q_img_tmp.convertTo(q_img_tmp, CV_32F);
//            createHanningWindow(hann_p, cv::Size(POC_BIG_WINDOW_SIZE, POC_BIG_WINDOW_SIZE), CV_32F);
//            cv::Mat warpGround = (cv::Mat_<float>(2,3) << 1, 0, -corner_tmp[j].x, 0, 1, -corner_tmp[j].y);
//            cv::Mat hann_q;
//            warpAffine(hann_p, hann_q, warpGround, cv::Size(hann_p.rows,hann_p.cols), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
//            p_img_tmp = p_img_tmp.mul(hann_p);
//            q_img_tmp = q_img_tmp.mul(hann_q);

//            tmp = phaseCorrelate(p_img_tmp, q_img_tmp, hann_p, &response);
//            std::cout << "x: " << tmp.x << ", y: " << tmp.y << std::endl;
//
//
//            // check converge and assign
//            if(j==0){
//                std::cout<<j<<"-th corner: "<<tmp.x<<" "<<tmp.y<<std::endl;
//                std::cout<<j<<"-th response "<<response<<std::endl;
//            }
            corner_tmp[j].x = float(tmp.x); corner_tmp[j].y = float(tmp.y);
//            corner_tmp[j].x = float(-transform_params[0]); corner_tmp[j].y = float(-transform_params[1]);


            // bad matches
            if((response1<0.9 || response2<0.9 || abs(corner_tmp[j].x)>1 || abs(corner_tmp[j].y)>1) && i==SUB_PIXEL_ITER-1){
                del_list.push_back(j);
            }


            // assign value when iteration is done
            if(i==SUB_PIXEL_ITER-1){
                (*q_corners)[j] = (*q_corners)[j] + corner_tmp[j];
            }

        }

    }

    // delete those corners too far away
    if(del_list.size()!=0){
        std::cout<<"bad corners in subpixel matching : "<<del_list.size()<<std::endl;
    }

    for(int idx=0; idx<del_list.size(); idx++){
        (*p_corners).erase((*p_corners).begin()+del_list[idx]-idx);
        (*q_corners).erase((*q_corners).begin()+del_list[idx]-idx);
        for(int c=0; c<(*corners).size(); c++){
            (*corners)[c].erase((*corners)[c].begin()+del_list[idx]-idx);
        }
        // if delete points from previous corners
        if((*last_good_corners).size()!=0 && del_list[idx]-idx>=(*num_current_corners)){
            auto idx_tmp = (*last_good_corners).size()-Xs_tmp.size();
            (*last_good_corners).erase((*last_good_corners).begin()+del_list[idx]-idx-(*num_current_corners));
            (*last_good_idx).erase((*last_good_idx).begin()+del_list[idx]-idx-(*num_current_corners));
            if(!Xs_tmp.empty() && del_list[idx]-idx-(*num_current_corners)>idx_tmp)
                Xs_tmp.erase(Xs_tmp.begin()+del_list[idx]-idx-(*num_current_corners)-idx_tmp);
        }
        // if delete points from current corners
        else{
            (*num_current_corners) = (*num_current_corners)-1;
        }
    }

    // return new_q_corners

    std::cout<<"POC Sub-Pixel level done"<<std::endl;

}
