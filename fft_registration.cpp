
#include "fft_registration.hpp"
#include "functions.hpp"

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

    if(src.depth() == CV_32F){
        _dst.create( src.rows, src.cols, CV_32FC1 );
    }
    else{
        _dst.create( src.rows, src.cols, CV_64FC1 );
    }

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
                dataDst[j/2] = (float)std::sqrt((double)dataSrc[j]*dataSrc[j] + (double)dataSrc[j+1]*dataSrc[j+1]);
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