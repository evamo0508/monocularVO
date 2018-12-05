/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include "myslam/common_include.h"
#include "myslam/map.h"

#include <opencv2/features2d/features2d.hpp>

namespace myslam 
{
class VisualOdometry
{
public:
    typedef shared_ptr<VisualOdometry> Ptr;
    enum VOState {
        INITIALIZING=-1,
        OK=0,
        LOST
    };
    
    VOState                        state_;                 // current VO status
    Map::Ptr                       map_;                   // map with all frames and map points
    vector<Point3d>                new_map_points_;        // for new traingulated points and clear for mappoints visualization
    vector<Point3d>                old_map_points_;        // for visualization of map points that are erased 
    Mat                            K;                      // camera intrinsics

    Frame::Ptr                     ref_;                   // reference key-frame 
    Frame::Ptr                     curr_;                  // current frame 
    
    cv::Ptr<cv::ORB>               orb_;                   // orb detector and computer 
    vector<cv::KeyPoint>           keypoints_ref_;         // keypoints in reference frame
    vector<cv::KeyPoint>           keypoints_curr_;        // keypoints in current frame
    Mat                            descriptors_ref_;       // descriptor in reference frame 
    Mat                            descriptors_curr_;      // descriptor in current frame 

    cv::BFMatcher                  matcher_bf_;            // bruteforce-hamming matcher
    vector<cv::DMatch>             matches_2frames_;       // matches for every 2 frames
    vector<cv::DMatch>             matches_2map_;          // matches between ref_ and map
    vector<MapPoint::Ptr>          match_3dpts_;           // matched 3d points 
    vector<int>                    match_2dkp_index_;      // matched 2d pixels (index of kp_curr)
    vector<int>                    match_2dkp_index_ref_;  // matched 2d pixels (index of kp_ref)
    
    SE3                            T_c_w_estimated_;       // the estimated pose of current frame 
    Mat                            R, t;                   // rotation matrix and translation
    int                            num_inliers_;           // number of inlier features in icp
    int                            num_lost_;              // number of lost times
    
    // parameters 
    int                            num_of_features_;       // number of features
    double                         scale_factor_;          // scale in image pyramid
    int                            level_pyramid_;         // number of pyramid levels
    float                          match_ratio_;           // ratio for selecting  good matches
    int                            max_num_lost_;          // max number of continuous lost times
    int                            min_inliers_;           // minimum inliers
    double                         key_frame_min_rot;      // minimal rotation of two key-frames
    double                         key_frame_min_trans;    // minimal translation of two key-frames
    double                         map_point_erase_ratio_; // remove map point ratio

public: 
    // functions 
    VisualOdometry();
    ~VisualOdometry();
    
    bool addFrame( Frame::Ptr&, Frame::Ptr = nullptr);      // add a new frame, frame2 for initialization only 
    
protected:  
    // inner operation 
    void findFeatureRef();
    void findFeatureCurr();
    void match2Frames();
    void match2Map();
    void poseEstimationEpipolar();
    void poseEstimationPnP(); 
    void triangulation();

    void addKeyFrame();
    void addMapPoints();
    bool checkEstimatedPose(); 
    bool checkKeyFrame();
    void optimizeMap();
    double getViewAngle( Frame::Ptr frame, MapPoint::Ptr point );
    Point2d pixel2cam ( const Point2d&);
};

}

#endif // VISUALODOMETRY_H
