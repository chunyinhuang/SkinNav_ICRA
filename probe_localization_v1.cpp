
//  Fourier-Based Feature Tracking, Probe Localization and Apriltag Detection
//  Version -- 1
//  Arthur: Jin 2020.10.20

/* ---------------below are parameters inherit from Jin's code---------------*/

#include "opencv2/opencv.hpp"
#include "ceres/ceres.h"


#include <math.h> // pow, round
#include <stdlib.h> // abs rand
#include <iostream>
#include <vector> // vector
#include <iomanip> // number of digits after decimal point
#include <algorithm> // random shuffle
#include <time.h> // timing

#include "bundle_adjustment.hpp"
#include "sfm.hpp"
#include "functions.hpp"
#include "fft_registration.hpp"

// include headers for apriltag
extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/tag36h11.h"
#include "apriltag/tag25h9.h"
#include "apriltag/tag16h5.h"
#include "apriltag/tagCircle21h7.h"
#include "apriltag/tagCircle49h12.h"
#include "apriltag/tagCustom48h12.h"
#include "apriltag/tagStandard41h12.h"
#include "apriltag/tagStandard52h13.h"
#include "apriltag/common/getopt.h"
#include "apriltag/common/homography.h"
#include "apriltag/apriltag_pose.h"
}



#define INIT_FRAME 100 //100
#define FRAME_COUNT 2700 //1800
#define FRAME_INTERVAL 20 // 4 16 20 remember to change if FPS=120
#define RECONSTRUCT_INTERVAL 4 // 3 1 4
//#define SCALE_FACTOR 1
#define GFT_CORNER_NUM 500
#define GFT_MIN_DIST 10
#define XS_THRESHOLD 2
#define ENABLE_SUB_POC 1
#define SHOW_FEATURE_TRACKING 0


/*---------------------------------------------------------------------------*/



/* --------------- Anatomy SLAM and Probe Localization ---------------*/


int main(int argc, char *argv[])
{

    // Command line inputs for Apriltag

    getopt_t *getopt = getopt_create();
    getopt_add_bool(getopt, 'h', "help", 0, "Show this help");
    getopt_add_bool(getopt, 'd', "debug", 1, "Enable debugging output (slow)");
    getopt_add_bool(getopt, 'q', "quiet", 0, "Reduce output");
    getopt_add_string(getopt, 'f', "family", "tagStandard41h12", "Tag family to use");
    getopt_add_int(getopt, 't', "threads", "1", "Use this many CPU threads");
    getopt_add_double(getopt, 'x', "decimate", "2.0", "Decimate input image by this factor");
    getopt_add_double(getopt, 'b', "blur", "0.0", "Apply low-pass blur to input");
    getopt_add_bool(getopt, '0', "refine-edges", 1, "Spend more time trying to align edges of tags");


    // Initialize tag detector with options

    apriltag_family_t *tf = NULL;
    const char *famname = getopt_get_string(getopt, "family");
    if (!strcmp(famname, "tag36h11")) {
        // tf = tag36h11_create();
        tf = tagStandard41h12_create();
    } else if (!strcmp(famname, "tag25h9")) {
        tf = tag25h9_create();
    } else if (!strcmp(famname, "tag16h5")) {
        tf = tag16h5_create();
    } else if (!strcmp(famname, "tagCircle21h7")) {
        tf = tagCircle21h7_create();
    } else if (!strcmp(famname, "tagCircle49h12")) {
        tf = tagCircle49h12_create();
    } else if (!strcmp(famname, "tagStandard41h12")) {
        tf = tagStandard41h12_create();
    } else if (!strcmp(famname, "tagStandard52h13")) {
        tf = tagStandard52h13_create();
    } else if (!strcmp(famname, "tagCustom48h12")) {
        tf = tagCustom48h12_create();
    } else {
        printf("Unrecognized tag family name. Use e.g. \"tag36h11\".\n");
        exit(-1);
    }


    // set up april tag detector

    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    td->quad_decimate = getopt_get_double(getopt, "decimate");
    td->quad_sigma = getopt_get_double(getopt, "blur");
    td->nthreads = getopt_get_int(getopt, "threads");
    td->debug = getopt_get_bool(getopt, "debug");
    td->refine_edges = getopt_get_bool(getopt, "refine-edges");

    
    clock_t time_begin = clock();


    // intrinsic mat
    /********* new calibration result ( first 1/3 +2 )
     Focal Length:          fc = [ 1610.51324   1613.05183 ] +/- [ 11.73317   11.55720 ]
     Principal point:       cc = [ 954.94718   548.71677 ] +/- [ 7.85361   9.57045 ]
     Skew:             alpha_c = [ 0.00000 ] +/- [ 0.00000  ]   => angle of pixel axes = 90.00000 +/- 0.00000 degrees
     Distortion:            kc = [ 0.18660   -0.63251   0.01228   0.00237  0.00000 ] +/- [ 0.02503   0.15011   0.00269   0.00211  0.00000 ]
     Pixel error:          err = [ 0.53526   0.37291 ]
     ***********/
    
    // settings for bundle adjustment

    double f_init =  1610.51324, cx_init = 954.94718, cy_init = 548.71677, Z_init = 0.5, Z_limit = 100, ba_loss_width = 9; // Negative 'loss_width' makes BA not to use a loss function.
    int min_inlier_num = 200, ba_num_iter = 200; // Negative 'ba_num_iter' uses the default value for BA minimization
    
    
    // video input

    char input_name[] = "./scan_tag_8.mov";
    cv::VideoCapture cap;
    cap.open(input_name + cv::CAP_ANY);
    std::cout<<cap.get(cv::CAP_PROP_FRAME_COUNT)<<std::endl;
    if(!cap.isOpened()){
        std::cout<<"INVALID VIDEO PATH!!"<<std::endl;
        return -1;
    }
    

    // initialize global variables

    std::vector<cv::Point2d> good_corners;                              // corner templates
    std::vector<std::unordered_map<int, int>> global_feature_map;       // global feature map
    std::vector<std::vector<std::vector<cv::Point2d>>> global_features; // global features
    std::vector<std::vector<SFM::Vec9d>> cameras;                       // global cameras
    std::vector<std::vector<cv::Point3d>> Apriltag_R;                   // global apriltag rotation matrix
    std::vector<std::vector<cv::Point3d>> Apriltag_t;                   // global apriltag translation matrix
    std::vector<std::vector<int>> apriltag_count;                       // global apriltag count
    std::vector<cv::Point3d> Xs_tmp;                                    // global 3D point templates
    std::vector<cv::Point3d> Xs;                                        // global 3D points
    std::vector<int> Xs_count;                                          // global 3D point counts per frame
    std::vector<cv::Vec3b> Xs_rgb;                                      // global 3D oint colors
    int count = INIT_FRAME;                                             // specify first frame
    

    while(cap.isOpened() && count<FRAME_COUNT)
    {

        //read frame0

        cv::Mat frame0, frame1;
        cap.set(cv::CAP_PROP_POS_FRAMES,count);
        cap.read(frame0);


        // local apriltag pose templates

        std::vector<cv::Point3d> local_Apriltag_R;
        std::vector<cv::Point3d> local_Apriltag_t;
        std::vector<int> local_apriltag_count;
        

        // Check if there is April tag

        cv::Mat gray;
        cvtColor(frame0, gray, cv::COLOR_RGB2GRAY);

        // Make an image_u8_t header for the Mat data
        image_u8_t im = { .width = gray.cols,
            .height = gray.rows,
            .stride = gray.cols,
            .buf = gray.data
        };
        zarray_t *detections = apriltag_detector_detect(td, &im);
        std::cout << zarray_size(detections) << " tags detected" << std::endl;


        // Draw detection outlines

        apriltag_detection_t *det;
        if (zarray_size(detections) != 0){
            zarray_get(detections, 0, &det);
            // First create an apriltag_detection_info_t struct using your known parameters.
            apriltag_detection_info_t info;
            info.det = det;
            info.tagsize = 0.022;
            info.fx = 1610.51324;
            info.fy = 1610.51324;
            info.cx = 954.94718;
            info.cy = 548.71677;
            // Then call estimate_tag_pose.
            apriltag_pose_t pose;
            double err = estimate_tag_pose(&info, &pose);
            local_Apriltag_R.push_back(cv::Point3d(pose.R->data[0], pose.R->data[1], pose.R->data[2]));
            local_Apriltag_t.push_back(cv::Point3d(pose.t->data[0], pose.t->data[1], pose.t->data[2]));
            local_apriltag_count.push_back(1);
        }
        else{
            local_Apriltag_R.push_back(cv::Point3d(0, 0, 0));
            local_Apriltag_t.push_back(cv::Point3d(0, 0, 0));
            local_apriltag_count.push_back(0);
        }



        // find good features on the first frame
        
        // pre process images
        cv::Mat color0 = cv::Mat::zeros(frame0.rows, frame0.cols, CV_8UC3);
        frame0.copyTo(color0);
        cv::Mat gray0 = cv::Mat::zeros(frame0.rows, frame0.cols, CV_8U);
        cv::Mat mask = pre_process(&color0, &gray0);

        // make tracking ROI smaller
        cv::Mat kernel = cv::Mat::ones(5, 5, CV_8U);
        cv::erode(mask, mask, kernel);

        // push back previously found good corners
        std::vector<int> good_corner_idx;
        for(int i=0; i<good_corners.size(); i++){
            good_corner_idx.push_back(i);
        }

        // add corner constraints to the mask (force corner distances to be > 5 pixels)
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

        // add corner constraints to the mask (force not to find corners close to apriltag)
        if (zarray_size(detections) != 0){
            cv::Mat kernel1 = cv::Mat::ones(11, 11, CV_8U);
            cv::Mat occlude = cv::Mat::zeros(frame0.rows, frame0.cols, CV_8U);
            for( int j = det->p[0][1]; j < det->p[2][1]; j++ ){
                for( int i = det->p[0][0]; i < det->p[2][0]; i++ ){
                    occlude.at<uchar>(j, i) = 255;
                }
            }
            dilate(occlude, occlude, kernel1);
            bitwise_and(occlude, mask, occlude);
            bitwise_xor(mask, occlude, mask);
        }
        apriltag_detections_destroy(detections);
        
        // good features to track for the first frame
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
        

        // add back last new corners (excluded in the preprocess step)

        if(good_corners.size()!=0){
            for( int i = 0; i < good_corners.size(); i++ )
            {
                corner0.push_back(good_corners[i]);
            }
        }
        
        
        // now lets go into video frames of the set (RECONSTRUCT_INTERVAL: numbers of frames per set)

        std::vector<std::vector<cv::Point2d>> corners;
        corners.push_back(corner0);
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


            // Check if there is April tag in frame 1

            cvtColor(frame1, gray, cv::COLOR_RGB2GRAY);

            // Make an image_u8_t header for the Mat data
            image_u8_t im = { .width = gray.cols,
                .height = gray.rows,
                .stride = gray.cols,
                .buf = gray.data
            };
            zarray_t *detections = apriltag_detector_detect(td, &im);
            std::cout << zarray_size(detections) << " tags detected" << std::endl;

            // Draw detection outlines
            if (zarray_size(detections) != 0){
                apriltag_detection_t *det;
                zarray_get(detections, 0, &det);
                // First create an apriltag_detection_info_t struct using your known parameters.
                apriltag_detection_info_t info;
                info.det = det;
                info.tagsize = 0.022;
                info.fx = 1610.51324;
                info.fy = 1610.51324;
                info.cx = 954.94718;
                info.cy = 548.71677;
                // Then call estimate_tag_pose.
                apriltag_pose_t pose;
                double err = estimate_tag_pose(&info, &pose);
                local_Apriltag_R.push_back(cv::Point3d(pose.R->data[0], pose.R->data[1], pose.R->data[2]));
                local_Apriltag_t.push_back(cv::Point3d(pose.t->data[0], pose.t->data[1], pose.t->data[2]));
                local_apriltag_count.push_back(1);
            }
            else{
                local_Apriltag_R.push_back(cv::Point3d(0, 0, 0));
                local_Apriltag_t.push_back(cv::Point3d(0, 0, 0));
                local_apriltag_count.push_back(0);
            }
            apriltag_detections_destroy(detections);


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
                    // mark survived features in Cyan
                    if(i < max_corners)
                        circle( outout, q_corner[i], 5, cv::Scalar(255, 255, 0), -1);
                    // mark newly tracked features in Blue
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
                bool dumb = cv::solvePnPRansac(Xs_tmp, q_corner_tmp, cameraMatrix, cv::noArray(), rvec, tvec);
                cv::Rodrigues(rvec, R);
                cv::Vec3d t(tvec[0], tvec[1], tvec[2]);
                cv::Vec3d p = -R.t() * t;
            }
            // ---------------------------------------
        }
        

        // Append Apriltag poses of the set
        Apriltag_R.push_back(local_Apriltag_R);
        Apriltag_t.push_back(local_Apriltag_t);
        apriltag_count.push_back(local_apriltag_count);

        
        // store the good corners of #(RECONSTRUCT_INTERVAL-1) for next frame intervals
        good_corners = corners[RECONSTRUCT_INTERVAL];
        
        // repeat the last frame for the next set
        count+=(RECONSTRUCT_INTERVAL)*FRAME_INTERVAL;
        
    
        // Append Global feature corners
        global_features.push_back(corners);
        
    
        // Append cameras (rotation, translation, intrinsic parameters)
        std::vector<SFM::Vec9d> local_cameras(corners.size(), SFM::Vec9d(0, 0, 0, 0, 0, 0, f_init, cx_init, cy_init));
        
        
        // build local 3D points (for this set)
        std::vector<cv::Point3d> local_Xs;
        ceres::Problem local_ba;


        // initialize local 3D points (MAY be redundant)

        // for all the survived points
        for (auto idx = 0; idx<corners[RECONSTRUCT_INTERVAL].size(); idx++)
        {
            // for new 3D points, initializa them at position (0, 0, Zinit)
            if(idx < max_corners){
                local_Xs.push_back(cv::Point3d(0, 0, Z_init));
            }
        }


        // put the points into reconstruction pipeline

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


        // Solve local (set) BA with Ceres Solver 

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
            // for new 3D points, add them to global Xs with locally computed values
            if(idx < max_corners){
                local_feature_map[idx] = int(Xs.size());
                // use pre-built 3D points
                Xs.push_back(local_Xs[idx]);
                Xs_count.push_back(1);
                Xs_rgb.push_back(color1.at<cv::Vec3b>(corners[RECONSTRUCT_INTERVAL][idx].y, corners[RECONSTRUCT_INTERVAL][idx].x));
            }
            // for existed 3D points, simply notify the system to add 1 to their number
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

            // Solve global BA with Ceres Solver 

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
    
    
    // delete points that show up less than XS_THRESHOLD (2) times

    int num_erased = 0;
    for(auto i=0; i<Xs_count.size(); i++){
        if(Xs_count[i] <= XS_THRESHOLD){
            Xs.erase(Xs.begin() + i - num_erased);
            num_erased++;
        }
    }
    std::cout<<"Final number of 3D points: "<<Xs.size()<<std::endl;
    

    // print results

    for(size_t s = 0; s<cameras.size(); s++){
        printf("set #: %zd\n", s);
        for (size_t j = 0; j < cameras[s].size(); j++){
            printf("    3DV Tutorial: Camera %zd's (f, cx, cy) = (%.3f, %.1f, %.1f)\n", j, cameras[s][j][6], cameras[s][j][7], cameras[s][j][8]);
        }
    }
    

    // Store the 3D points to an XYZ file

    FILE* fpts = fopen("./probe_sfm_points.xyz", "wt");
    if (fpts == NULL) return -1;
    for (size_t i = 0; i < Xs.size(); i++)
    {
        if (Xs[i].z > -Z_limit && Xs[i].z < Z_limit)
            fprintf(fpts, "%f %f %f %d %d %d\n", Xs[i].x, Xs[i].y, Xs[i].z, Xs_rgb[i][2], Xs_rgb[i][1], Xs_rgb[i][0]); // Format: x, y, z, R, G, B
    }
    fclose(fpts);


    // Store the camera and apriltag poses to an XYZ file

    FILE* fcam = fopen("./probe_sfm_cameras.xyz", "wt");
    FILE* fapril = fopen("./probe_localization_apriltags.xyz", "wt");
    if (fcam == NULL) return -1;
    if (fapril == NULL) return -1;
    for(size_t s = 0; s<cameras.size(); s++){
        for (size_t j = 0; j < cameras[s].size(); j++)
        {
            // camera poses
            cv::Vec3d rvec(cameras[s][j][0], cameras[s][j][1], cameras[s][j][2]), t(cameras[s][j][3], cameras[s][j][4], cameras[s][j][5]);
            cv::Matx33d R;
            cv::Rodrigues(rvec, R);
            cv::Vec3d p = -R.t() * t;
            fprintf(fcam, "%f %f %f %f %f %f\n", p[0], p[1], p[2], R.t()(0, 2), R.t()(1, 2), R.t()(2, 2)); // Format: x, y, z, n_x, n_y, n_z

            // apriltag poses
            std::cout<<apriltag_count[s][j]<<" ";
            if(apriltag_count[s][j]==1){
                cv::Vec3d t_a(t[0] - Apriltag_t[s][j].x, t[1] - Apriltag_t[s][j].y, t[2] - Apriltag_t[s][j].z);
                cv::Vec3d p_a = -R.t() * t_a;
                fprintf(fapril, "%f %f %f\n", p_a[0], p_a[1], p_a[2]); // Format: x, y, z
            }
        }
        std::cout<<std::endl;
    }
    fclose(fcam);
    fclose(fapril);
    

    // delete apriltag detector and video capture
    apriltag_detector_destroy(td);
    cap.release();

    // destroy apriltag UI 
    if (!strcmp(famname, "tag36h11")) {
        tag36h11_destroy(tf);
    } else if (!strcmp(famname, "tag25h9")) {
        tag25h9_destroy(tf);
    } else if (!strcmp(famname, "tag16h5")) {
        tag16h5_destroy(tf);
    } else if (!strcmp(famname, "tagCircle21h7")) {
        tagCircle21h7_destroy(tf);
    } else if (!strcmp(famname, "tagCircle49h12")) {
        tagCircle49h12_destroy(tf);
    } else if (!strcmp(famname, "tagStandard41h12")) {
        tagStandard41h12_destroy(tf);
    } else if (!strcmp(famname, "tagStandard52h13")) {
        tagStandard52h13_destroy(tf);
    } else if (!strcmp(famname, "tagCustom48h12")) {
        tagCustom48h12_destroy(tf);
    }
    getopt_destroy(getopt);
    
    std::cout << "Total time elapsed: " << (clock() - time_begin) * 1.0 / CLOCKS_PER_SEC << "seconds" << std::endl;
    
    return 0;
}


/*---------------------------------------------------------------------------*/

