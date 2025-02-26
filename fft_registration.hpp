#ifndef FFT_REGISTRATION_JIN
#define FFT_REGISTRATION_JIN

#include <complex>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

#include "opencv2/opencv.hpp"
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