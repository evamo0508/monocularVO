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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <algorithm>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"

namespace myslam
{

VisualOdometry::VisualOdometry() :
    state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 ), matcher_bf_( cv::NORM_HAMMING, true)
{
    num_of_features_       = Config::get<int> ( "number_of_features" );
    scale_factor_          = Config::get<double> ( "scale_factor" );
    level_pyramid_         = Config::get<int> ( "level_pyramid" );
    match_ratio_           = Config::get<float> ( "match_ratio" );
    max_num_lost_          = Config::get<float> ( "max_num_lost" );
    min_inliers_           = Config::get<int> ( "min_inliers" );
    key_frame_min_rot      = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans    = Config::get<double> ( "keyframe_translation" );
    map_point_erase_ratio_ = Config::get<double> ( "map_point_erase_ratio" );
    orb_                   = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
}

VisualOdometry::~VisualOdometry()
{

}

bool VisualOdometry::addFrame ( Frame::Ptr& frame, Frame::Ptr frame2 )
{
    switch ( state_ )
    {
    case INITIALIZING:
    {
        state_ = OK;
        ref_ = frame;
        curr_ = frame2;
        K = ( cv::Mat_<double> ( 3,3 ) <<
              ref_->camera_->fx_, 0, ref_->camera_->cx_,
              0, ref_->camera_->fy_, ref_->camera_->cy_,
              0,0,1 );
        findFeatureRef();
        findFeatureCurr();
        match2Frames();
        poseEstimationEpipolar(); // calculate R, t
        triangulation();          // new_map_points_ have been updated and are in camera ref_ coordinate       
        addKeyFrame();            // the first frame is a key-frame
        break;
    }
    case OK:
    {
        ref_ = curr_;
        keypoints_ref_ = keypoints_curr_;
        descriptors_ref_ = descriptors_curr_.clone();
        match_2dkp_index_ref_ = match_2dkp_index_;
        curr_ = frame;
        curr_->T_c_w_ = ref_->T_c_w_;
        findFeatureCurr();
        match2Frames();
        match2Map();
        poseEstimationPnP();      // calculate R,t (T_c_r)
        triangulation();
        
        if ( checkEstimatedPose()) // a good estimation
        {
            curr_->T_c_w_ = T_c_w_estimated_;
            optimizeMap();
            num_lost_ = 0;
            if ( checkKeyFrame()) addKeyFrame();
        }
        else // bad estimation due to various reasons
        {
            curr_ = ref_ ;
            keypoints_curr_ = keypoints_ref_;
            descriptors_curr_ = descriptors_ref_.clone();
            num_lost_++;
            if ( num_lost_ > max_num_lost_ ) state_ = LOST;
            return false;
        }
        break;
    }
    case LOST:
    {
        cout<<"vo has lost."<<endl;
        break;
    }
    }

    return true;
}

void VisualOdometry::findFeatureRef()
{ 
    Mat img = ref_->color_;
    orb_ -> detect ( img, keypoints_ref_ );
    orb_ -> compute ( img, keypoints_ref_, descriptors_ref_ );
}

void VisualOdometry::findFeatureCurr()
{ 
    Mat img = curr_->color_;
    orb_ -> detect ( img, keypoints_curr_ );
    orb_ -> compute ( img, keypoints_curr_, descriptors_curr_ );
}

void VisualOdometry::match2Frames()
{  
    vector<cv::DMatch> match;
    matcher_bf_.match ( descriptors_ref_, descriptors_curr_, match );
    cout << "matches size: " << match.size() << endl; 
     
    // filter matched features
    double min_dis = std::min_element (
                        match.begin(), match.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
                        { return m1.distance < m2.distance; } )->distance;
    cout << "min dis: " << min_dis << endl;

    matches_2frames_.clear();
    for ( cv::DMatch& m:match )
    {
        if ( m.distance < max<double> ( min_dis*match_ratio_, 35.0 ) ) matches_2frames_.push_back(m); 
    } 
    cout << "good matches between 2 frames: " << matches_2frames_.size() << endl;
}

void VisualOdometry::match2Map()
{
    // select the candidates in map 
    Mat desp_map;
    vector<MapPoint::Ptr> candidate;
    for ( auto& allpoints: map_->map_points_ )
    {
        MapPoint::Ptr& p = allpoints.second; // type: unordered_map<unsigned long, MapPoint::Ptr>
        if ( ref_->isInFrame(p->pos_) )      // check whether p is in ref frame, more likely to match with curr frame
        {
            // add to candidate 
            p->visible_times_++;
            candidate.push_back( p );
            desp_map.push_back( p->descriptor_ );
        } 
    }
    cout << "candidate size: " << candidate.size() << endl;
    
    matches_2map_.clear();
    matcher_bf_.match ( desp_map, descriptors_curr_, matches_2map_ );
    cout << "matches found between this frame & map: " << matches_2map_.size() << endl;
    
    // filter matched features
    float min_dis = std::min_element (
                        matches_2map_.begin(), matches_2map_.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
                        { return m1.distance < m2.distance; } )->distance;
    cout <<"min dist: " << min_dis << endl;
    
    match_3dpts_.clear();
    match_2dkp_index_.clear();
    for ( cv::DMatch& m:matches_2map_ )
    {
        if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
        {
            match_3dpts_.push_back( candidate[m.queryIdx] ); //mappoint of map candidate
            match_2dkp_index_.push_back( m.trainIdx );       //index of keypoints_curr_
        }
    }

    cout<<"good matches between ref frame and map: "<<match_3dpts_.size() <<endl;
}

void VisualOdometry::poseEstimationEpipolar ()
{
    Point2d principal_point (K.at<double> (0,2), K.at<double> (1,2)); 
    double focal_length = (K.at<double> (0,0) + K.at<double> (1,1)) / 2;   
    
    //transform matches points to vector<Point2f> 
    vector<Point2f> points1;
    vector<Point2f> points2;
    for (int i=0; i< (int)matches_2frames_.size(); i++)
    {
        points1.push_back(keypoints_ref_[matches_2frames_[i].queryIdx].pt);
        points2.push_back(keypoints_curr_[matches_2frames_[i].trainIdx].pt);
    }

    //compute essential matrix E
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point);
    cout << "essential matrix is " << endl << essential_matrix << endl;
    
    //recover R and t from essential matrix
    recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;
    Vector3d t_SE3;
    Matrix3d R_SE3;
    cv::cv2eigen(R, R_SE3);
    cv::cv2eigen(t, t_SE3);
    curr_->T_c_w_ = SE3(R_SE3, t_SE3);
}

void VisualOdometry::triangulation()
{  
    Mat T1 = (cv::Mat_<float> (3,4) <<
        1,0,0,0,
        0,1,0,0,
        0,0,1,0);
    Mat T2 = (cv::Mat_<float> (3,4) <<
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
    );

    vector<Point2f> pts_1, pts_2;
    for (cv::DMatch m:matches_2frames_)
    {
        //pixel coordinate -> camera coordinate
        pts_1.push_back(pixel2cam(keypoints_ref_[m.queryIdx].pt));
        pts_2.push_back(pixel2cam(keypoints_curr_[m.trainIdx].pt));
    }

    Mat pts_4d;
    triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);//the depth is camera T1's depth

    //transform to nonhomogeneous coordinate
    new_map_points_.clear();
    for (int i=0; i<pts_4d.cols; i++)
    {
        Mat norm = pts_4d.col(i);
        norm /= norm.at<float>(3,0); //w, a number generated simply for computing homogeneous matrix, need to normalize
        Point3d p(
            norm.at<float>(0,0),
            norm.at<float>(1,0),
            norm.at<float>(2,0)
        );
        new_map_points_.push_back(p);
        int x = cvRound(keypoints_ref_[matches_2frames_[i].queryIdx].pt.x);
        int y = cvRound(keypoints_ref_[matches_2frames_[i].queryIdx].pt.y);
        ref_->depth_.at<double>( y,x ) = p.z;
    }
}

void VisualOdometry::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<cv::Point3f> pts3d;                                                                                                               
    vector<cv::Point2f> pts2d;

    for ( int index:match_2dkp_index_ ) { pts2d.push_back ( keypoints_curr_[index].pt ); }
    for ( MapPoint::Ptr pt:match_3dpts_ ) { pts3d.push_back( pt->getPositionCV() ); }
    
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac ( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 20.0, 0.99, inliers );
    num_inliers_ = inliers.rows;
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    T_c_w_estimated_ = SE3 (
                           SO3      ( rvec.at<double> ( 0,0 ), rvec.at<double> ( 1,0 ), rvec.at<double> ( 2,0 ) ),
                           Vector3d ( tvec.at<double> ( 0,0 ), tvec.at<double> ( 1,0 ), tvec.at<double> ( 2,0 ) )
                           );
    
    // using bundle adjustment to optimize the pose
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block ( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
        T_c_w_estimated_.rotation_matrix(), T_c_w_estimated_.translation()
    ));
    optimizer.addVertex ( pose );

    // edges
    for ( int i=0; i<inliers.rows; i++ )
    {
        int index = inliers.at<int> ( i,0 );
        // 3D -> 2D projection
        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId ( i );
        edge->setVertex ( 0, pose );
        edge->camera_ = curr_->camera_.get();
        edge->point_ = Vector3d ( pts3d[index].x, pts3d[index].y, pts3d[index].z );
        edge->setMeasurement ( Vector2d ( pts2d[index].x, pts2d[index].y ) );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        optimizer.addEdge ( edge );
        // set the inlier map points 
        match_3dpts_[index]->matched_times_++;
    }

    optimizer.initializeOptimization();
    optimizer.optimize ( 10 );

    T_c_w_estimated_ = SE3 (
        pose->estimate().rotation(),
        pose->estimate().translation()
    );
    cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
    
    // calculate T_c_r from T_c_w_estimated_
    SE3 T_c_r = T_c_w_estimated_*ref_->T_c_w_.inverse();
    cv::eigen2cv(T_c_r.rotation_matrix(), R);
    cv::eigen2cv(T_c_r.translation(), t); 
}

void VisualOdometry::addKeyFrame()
{
    if (map_->keyframes_.empty())
    {
        //insert all match points & 1st frame into map
        for (int i=0; i<new_map_points_.size(); i++)
        {
            double d = new_map_points_[i].z;
            if (d<0) continue;
            int index = matches_2frames_[i].queryIdx;
            Vector3d p_world = ref_->camera_->pixel2world (
                Vector2d( keypoints_ref_[index].pt.x, keypoints_ref_[index].pt.y), 
                ref_->T_c_w_, d ); // ref_->T_c_w_ is still SE3()
            Vector3d n = p_world - ref_->getCamCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                p_world, n, descriptors_ref_.row(index).clone(), ref_.get());
            map_->insertMapPoint( map_point );
         }
         
         cout << "initialize map points: " << map_->map_points_.size() << endl;
         map_->insertKeyFrame ( ref_ );
         return;
    }
    map_->insertKeyFrame ( curr_ );
}

bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    
    // if the motion is too large, it is probably wrong
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    if ( d.norm() > 5.0 )
    {
        cout<<"reject because motion is too large: "<<d.norm() <<endl;
        return false;
    }
    return true;
}

void VisualOdometry::optimizeMap()
{
    // remove the hardly seen and no visible points 
    for ( auto iter = map_->map_points_.begin(); iter != map_->map_points_.end(); )
    {
        if ( !curr_->isInFrame(iter->second->pos_) )
        {
            MapPoint::Ptr& p = iter->second;
            Point3d point(p->pos_(0,0), p->pos_(1,0), p->pos_(2,0));
            old_map_points_.push_back(point);
            iter = map_->map_points_.erase(iter);
            continue;
        }
        
        float match_ratio = float(iter->second->matched_times_)/iter->second->visible_times_;
        if ( match_ratio < map_point_erase_ratio_ )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        
        double angle = getViewAngle( curr_, iter->second );
        if ( angle > M_PI/6. )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        
        iter++;
    }
     
    cout<<"map points size after delete: "<<map_->map_points_.size()<<endl;
    
    if ( match_2dkp_index_ref_.size() < 100 ) addMapPoints();
    
    if ( map_->map_points_.size() > 1000 ) { map_point_erase_ratio_ += 0.05; }
    else { map_point_erase_ratio_ = 0.1; }
    
    cout<<"map points size after update: "<<map_->map_points_.size()<<endl;
}


bool VisualOdometry::checkKeyFrame()
{
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void VisualOdometry::addMapPoints()
{
    // add the new map points into map
    vector<bool> matched(keypoints_ref_.size(), false); 
    
    for ( int index:match_2dkp_index_ref_ )
        matched[index] = true;
    
    for ( int i=0; i<keypoints_ref_.size(); i++ )
    {
        if ( matched[i] == true ) continue;
        double d = ref_->findDepth ( keypoints_ref_[i] );
        if ( d<0 ) continue;
        Vector3d p_world = ref_->camera_->pixel2world (
            Vector2d ( keypoints_ref_[i].pt.x, keypoints_ref_[i].pt.y ), 
            ref_->T_c_w_, d
        );
        Vector3d n = p_world - ref_->getCamCenter();
        n.normalize();
        MapPoint::Ptr map_point = MapPoint::createMapPoint(
            p_world, n, descriptors_ref_.row(i).clone(), ref_.get()
        );
        map_->insertMapPoint( map_point );
    }
}

double VisualOdometry::getViewAngle ( Frame::Ptr frame, MapPoint::Ptr point )
{
    Vector3d n = point->pos_ - frame->getCamCenter();
    n.normalize();
    return acos( n.transpose()*point->norm_ );
}

Point2d VisualOdometry::pixel2cam ( const Point2d& p)
{
    return Point2d
           (
                (p.x - K.at<double> (0,2))/K.at<double> (0,0), //(u-cx)/fx
                (p.y - K.at<double> (1,2))/K.at<double> (1,1)  //(v-cy)/fy
           );
}

}
