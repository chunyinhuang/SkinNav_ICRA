#include "functions.hpp"
#include "fft_registration.hpp"


/* ---------------hyper parameters---------------*/


#define POC_LAYERS 5
#define SUB_PIXEL_ITER 5
#define POC_BIG_WINDOW_SIZE 71
#define POC_SMALL_WINDOW_SIZE 71


#define SHOW_POC_BAD_CORNERS 0
#define SHOW_POC_GOOD_CORNERS 0

/*---------------------------------------------------------------------------*/

/* ---------------helper functions---------------*/
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


/*---------------------------------------------------------------------------*/