--observations--
1. The world coordinate is probably the same as the first frame because curr_->T_c_w = SE3() for initialization. Also cv::viz::makeCameraPose has (0,0,0) as cam_focal_point.
2. The second frame is used to calculate the depth of frame 1 keypoints only. We don't record any information of frame 2 at all.
3. 2. is not true at all now. We need to calculate triangulation between each two frames to get the depth of new feature points.
4. First 2 frames: epipolar; others: PnP, however, depth with triangulation.
--problems--
1. The 1st & 2nd frame might not differ from each other enough for triangulation.
2. Need triangulation for every two frames, but the scale would be inconstant then?
