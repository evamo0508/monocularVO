// -------------- test the visual odometry -------------
#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

int main ( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"usage: run_vo parameter_file"<<endl;
        return 1;
    }

    myslam::Config::setParameterFile ( argv[1] );
    myslam::VisualOdometry::Ptr vo ( new myslam::VisualOdometry );

    string dataset_dir = myslam::Config::get<string> ( "dataset_dir" );
    cout<<"dataset: "<<dataset_dir<<endl;
    
    ifstream fin ( dataset_dir+"/associate.txt" );
    if ( !fin )
    {
        cout<<"please generate the associate file called associate.txt!"<<endl;
        return 1;
    }

    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while ( !fin.eof() )
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );

        if ( fin.good() == false )
            break;
    }
    cout<<"read total "<<rgb_files.size() <<" entries"<<endl;
    
    myslam::Camera::Ptr camera ( new myslam::Camera );
    vector< cv::Point3d> route;
    
    // visualization: camera
    cv::viz::Viz3d vis ( "Visual Odometry" );
    cv::viz::WCoordinateSystem world_coor ( 0.5 ), camera_coor ( 0.5 );
    cv::Point3d cam_pos ( 0, -1.0, -1.0 ), cam_focal_point ( 0,0,0 ), cam_y_dir ( 0,1,0 );
    cv::Affine3d cam_pose = cv::viz::makeCameraPose ( cam_pos, cam_focal_point, cam_y_dir );//search gluLookAt to understand the principle
    vis.setViewerPose ( cam_pose );
    world_coor.setRenderingProperty ( cv::viz::LINE_WIDTH, 2.0 );
    camera_coor.setRenderingProperty ( cv::viz::LINE_WIDTH, 1.0 );
    vis.showWidget ( "World", world_coor );
    vis.showWidget ( "Camera", camera_coor );
    
    for ( int i=0; i<rgb_files.size(); i++ )
    {
        cout<<"****** loop "<<i<<" ******"<<endl;
        boost::timer timer;
        Mat color, color2;
        myslam::Frame::Ptr frame, frame2;
        if (i==0)
        {
            //initialize: epipolar + triangulation
            color = cv::imread( rgb_files[i] );
            color2 = cv::imread( rgb_files[i+1] );
            if (color.data==nullptr || color2.data==nullptr) break;
            frame = myslam::Frame::createFrame();
            frame2 = myslam::Frame::createFrame();
            frame->camera_ = frame2->camera_ = camera;
            frame->color_ = color;
            frame2->color_ = color2;
            frame->time_stamp_ = rgb_times[i];
            frame2->time_stamp_ = rgb_times[i+1];
            frame->depth_ = Mat::zeros(color.rows, color.cols, cv::DataType<double>::type);
            frame2->depth_ = Mat::zeros(color2.rows, color2.cols, cv::DataType<double>::type);
            if(vo -> addFrame( frame, frame2)) cout << "Initialize successful" << endl;
        }
        else
        {
            // PnP + triangulation
            color = cv::imread ( rgb_files[i+1] );
            if ( color.data==nullptr ) break;
            frame = myslam::Frame::createFrame();
            frame->camera_ = camera;
            frame->color_ = color;
            frame->depth_ = Mat::zeros(color.rows, color.cols, cv::DataType<double>::type);
            frame->time_stamp_ = rgb_times[i+1];
            vo->addFrame ( frame );
        }
        cout<<"VO costs time: "<<timer.elapsed() <<endl;

        if ( vo->state_ == myslam::VisualOdometry::LOST ) 
        {
            cout << "VO has lost." << endl;
            break;
        }
        
        SE3 Twc = frame->T_c_w_.inverse();
        int scale = 100;
        
        // show the map and the camera pose
        cv::Affine3d M (
            cv::Affine3d::Mat3 (
                Twc.rotation_matrix() ( 0,0 ), Twc.rotation_matrix() ( 0,1 ), Twc.rotation_matrix() ( 0,2 ),
                Twc.rotation_matrix() ( 1,0 ), Twc.rotation_matrix() ( 1,1 ), Twc.rotation_matrix() ( 1,2 ),
                Twc.rotation_matrix() ( 2,0 ), Twc.rotation_matrix() ( 2,1 ), Twc.rotation_matrix() ( 2,2 )
            ),
            cv::Affine3d::Vec3 (
                Twc.translation() ( 0,0 )/scale, Twc.translation() ( 1,0 )/scale, Twc.translation() ( 2,0 )/scale
            )
        );
        route.push_back( cv::Point3d( Twc.translation() (0,0)/scale, Twc.translation() (1,0)/scale, Twc.translation() (2,0)/scale));
        
        Mat img_show = vo->ref_->color_.clone();
        vector<cv::Point3d> show_map_points_ = vo->old_map_points_;
        for (int i=0; i<show_map_points_.size(); i++) show_map_points_[i]/=scale;
        for ( auto& pt:vo->map_->map_points_ )
        {
            myslam::MapPoint::Ptr p = pt.second;
            cv::Point3d p3d =  cv::Point3d( p->pos_(0,0), p->pos_(1,0), p->pos_(2,0))/scale;
            show_map_points_.push_back(p3d);
            Vector2d pixel = frame->camera_->world2pixel ( p->pos_, vo->ref_->T_c_w_ );
            cv::circle ( img_show, cv::Point2f ( pixel ( 0,0 ),pixel ( 1,0 ) ), 5, cv::Scalar ( 0,255,0 ), 2 );
        }
        
        cv::imshow ( "image", img_show );
        cv::waitKey (1);
       
        //visualization: camera pose 
        vis.setWidgetPose ( "Camera", M );

        //visualization: point cloud 
        cv::viz::WCloud cloud_widget( show_map_points_, cv::viz::Color::red());
        cloud_widget.setRenderingProperty( cv::viz::POINT_SIZE, 5.0 );
        vis.showWidget ( "Cloud", cloud_widget );
        
        //visualization: route 
        cv::viz::WCloud route_widget( route, cv::viz::Color::green());
        route_widget.setRenderingProperty( cv::viz::POINT_SIZE, 2.0 );
        vis.showWidget ( "Route", route_widget );
        
        vis.spinOnce ( 1, false );
        
        cout<<endl;
    }

    return 0;
}
