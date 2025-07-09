/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "utility/visualization.h"

using namespace Eigen;


// Global variables for data tracking
static PathData path;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
static double sum_of_path = 0;
static Vector3d last_path(0.0, 0.0, 0.0);
size_t pub_counter = 0;

// void registerPub(ros::NodeHandle &n)
// {
//     // printf("register publishers %s %s %s\n", IMU_TOPIC.c_str(), IMAGE0_TOPIC.c_str(), IMAGE1_TOPIC.c_str());

//     pub_imu = n.advertise<sensor_msgs::Imu>("/imu", 2000);
//     pub_img0 = n.advertise<sensor_msgs::Image>("/img0", 100);
//     pub_img1 = n.advertise<sensor_msgs::Image>("/img1", 100);
//     pub_gps = n.advertise<sensor_msgs::NavSatFix>("/gps", 100);

//     ROS_WARN("Publishing IMU on topic: %s", pub_imu.getTopic().c_str());
//     ROS_WARN("Publishing IMG0 on topic: %s", pub_img0.getTopic().c_str());
//     ROS_WARN("Publishing IMG1 on topic: %s", pub_img1.getTopic().c_str());

//     pub_latest_odometry = n.advertise<nav_msgs::Odometry>("/imu_propagate", 1000);
//     pub_path = n.advertise<nav_msgs::Path>("/path", 1000);
//     pub_odometry = n.advertise<nav_msgs::Odometry>("/odometry", 1000);
//     pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("/point_cloud", 1000);
//     pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("/margin_cloud", 1000);
//     pub_key_poses = n.advertise<visualization_msgs::Marker>("/key_poses", 1000);
//     pub_camera_pose = n.advertise<nav_msgs::Odometry>("/camera_pose", 1000);
//     pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("/camera_pose_visual", 1000);
//     pub_keyframe_pose = n.advertise<nav_msgs::Odometry>("/keyframe_pose", 1000);
//     pub_keyframe_point = n.advertise<sensor_msgs::PointCloud>("/keyframe_point", 1000);
//     pub_extrinsic = n.advertise<nav_msgs::Odometry>("/extrinsic", 1000);
//     pub_image_track = n.advertise<sensor_msgs::Image>("/image_track", 1000);

//     cameraposevisual.setScale(0.1);
//     cameraposevisual.setLineWidth(0.01);
// }

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t)
{
    // Create data structure that would have been published
    OdometryData data;
    data.position = P;
    data.orientation = Q;
    data.velocity = V;
    data.timestamp = t;
    
    // Data would be published to "/imu_propagate" topic
    // Implement your custom handler here
}

void pubImage0(cv::Mat &img0, double t)
{
    // Keep existing callback - this already looks ROS-free
    TimestampedImage frame = std::make_tuple(t, img0);
    estimator_node->img0_callback(frame);
    // This would have published to "/img0" topic
}

void pubImage1(cv::Mat &img1, double t)
{
    // Keep existing callback - this already looks ROS-free
    TimestampedImage frame = std::make_tuple(t, img1);
    estimator_node->img1_callback(frame);
    // This would have published to "/img1" topic
}

void pubImu(Eigen::Vector3d &acc, Eigen::Vector3d &gyr, double t)
{
    // Keep existing callback - this already looks ROS-free
    estimator_node->imu_callback(t, acc, gyr);
    // This would have published to "/imu" topic
}

void pubGps(double latitude, double longitude, double altitude, double posAccuracy, double t)
{
    // Create data structure that would have been published
    GpsData data;
    data.latitude = latitude;
    data.longitude = longitude;
    data.altitude = altitude;
    data.posAccuracy = posAccuracy;
    data.timestamp = t;
    
    // Data would be published to "/gps" topic
    // Implement your custom handler here
}


void pubOdometry(const Estimator &estimator, double t)
{
    if(estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;

    Eigen::Vector3d p = estimator.Ps[WINDOW_SIZE];
    Eigen::Quaterniond q(estimator.Rs[WINDOW_SIZE]);
    Eigen::Vector3d v = estimator.Vs[WINDOW_SIZE];

    // Keep existing callback - this already looks ROS-free
    posegraph_node.vio_callback(p, q, v, t);
    // globalopt_vio_callback(p, q, v, t));
    
    // Create data structure for odometry
    OdometryData odometry;
    odometry.position = p;
    odometry.orientation = q;
    odometry.velocity = v;
    odometry.timestamp = t;
    
    // Update path data (previously path.poses)
    path.positions.push_back(p);
    path.timestamps.push_back(t);
    
    // Data would have been published to:
    // - "/odometry" topic for odometry
    // - "/path" topic for path
    
    // Optional: Write result to file
    // std::ofstream foutC(VINS_RESULT_PATH, std::ios::app);
    // foutC.setf(std::ios::fixed, std::ios::floatfield);
    // foutC.precision(0);
    // foutC << t * 1e9 << ",";
    // foutC.precision(5);
    // foutC << p.x() << ","
    //       << p.y() << ","
    //       << p.z() << ","
    //       << q.w() << ","
    //       << q.x() << ","
    //       << q.y() << ","
    //       << q.z() << ","
    //       << v.x() << ","
    //       << v.y() << ","
    //       << v.z() << "," << std::endl;
    // foutC.close();
    
    // Debug output
    // printf("time: %f, t: %f %f %f q: %f %f %f %f \n", t, p.x(), p.y(), p.z(),
    //                                                q.w(), q.x(), q.y(), q.z());
}






void pubKeyPoses(const Estimator &estimator, double t)
{
    if (estimator.key_poses.size() == 0)
        return;
    
    // Create data structure for key poses
    KeyPosesData keyPosesData;
    keyPosesData.timestamp = t;
    keyPosesData.frame_id = "world";
    
    // Copy the key poses
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Vector3d correct_pose = estimator.key_poses[i];
        keyPosesData.poses.push_back(correct_pose);
    }
    
    // Data would have been published to "/key_poses" topic
    // Implement your custom handler here
}

void pubCameraPose(const Estimator &estimator, double t)
{
    int idx2 = WINDOW_SIZE - 1;

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        int i = idx2;
        Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);

        // Create data structure for camera pose
        CameraPoseData cameraPoseData;
        cameraPoseData.timestamp = t;
        cameraPoseData.frame_id = "world";
        cameraPoseData.positions.push_back(P);
        cameraPoseData.orientations.push_back(R);
        
        // Maintain visual representation for potential future visualization
        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        
        if(STEREO)
        {
            Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[1];
            Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[1]);
            cameraposevisual.add_pose(P, R);
            cameraPoseData.positions.push_back(P);
            cameraPoseData.orientations.push_back(R);
        }

        // Data would have been published to:
        // - "/camera_pose" topic for odometry
        // - "/camera_pose_visual" topic for visual markers
        // Implement your custom handler here
    }
}

void pubPointCloud(const Estimator &estimator, double t)
{
    // Create data structures for point clouds
    PointCloudData pointCloudData;
    pointCloudData.timestamp = t;
    pointCloudData.frame_id = "world";
    
    PointCloudData marginCloudData;
    marginCloudData.timestamp = t;
    marginCloudData.frame_id = "world";

    // Process main point cloud
    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        
        int imu_i = it_per_id.start_frame;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];
        
        pointCloudData.points.push_back(w_pts_i);
    }

    // Process margined point cloud
    for (auto &it_per_id : estimator.f_manager.feature)
    { 
        int used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 
            && it_per_id.solve_flag == 1)
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];
            
            marginCloudData.points.push_back(w_pts_i);
        }
    }
    
    // Data would have been published to:
    // - "/point_cloud" topic for main point cloud
    // - "/margin_cloud" topic for margined point cloud
    // Implement your custom handlers here
    
    // If you had a callback for margin points, you could call it like:
    // margin_point_callback(marginCloudData);
}

void pubTF(const Estimator &estimator, double t)
{
    if(estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    
    // Create transform data structures
    std::vector<TransformData> transforms;
    
    // Body frame transform
    TransformData bodyTransform;
    bodyTransform.translation = estimator.Ps[WINDOW_SIZE];
    bodyTransform.rotation = estimator.Rs[WINDOW_SIZE];
    bodyTransform.timestamp = t;
    bodyTransform.parent_frame = "world";
    bodyTransform.child_frame = "body";
    transforms.push_back(bodyTransform);
    
    // Camera frame transform
    TransformData cameraTransform;
    cameraTransform.translation = estimator.tic[0];
    cameraTransform.rotation = Quaterniond(estimator.ric[0]);
    cameraTransform.timestamp = t;
    cameraTransform.parent_frame = "body";
    cameraTransform.child_frame = "camera";
    transforms.push_back(cameraTransform);
    
    // Create extrinsic data
    ExtrinsicData extrinsicData;
    extrinsicData.position = estimator.tic[0];
    extrinsicData.orientation = Quaterniond(estimator.ric[0]);
    extrinsicData.timestamp = t;
    extrinsicData.frame_id = "world";
    
    // Data would have been published as:
    // - TF transforms from "world" to "body" and "body" to "camera"
    // - "/extrinsic" topic for extrinsic parameters
    // Implement your custom handlers here
    
    // If you had a callback for extrinsic data, you could call it like:
    // extrinsic_callback(extrinsicData);
}

void pubKeyframe(const Estimator &estimator)
{
    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR || estimator.marginalization_flag != 0)
        return;
    
    int i = WINDOW_SIZE - 2;
    Vector3d P = estimator.Ps[i];
    Quaterniond R = Quaterniond(estimator.Rs[i]);
    double timestamp = estimator.Headers[WINDOW_SIZE - 2];

    // Create keyframe pose data
    KeyframePoseData keyframePoseData;
    keyframePoseData.position = P;
    keyframePoseData.orientation = R;
    keyframePoseData.timestamp = timestamp;
    keyframePoseData.frame_id = "world";
    
    // Create keyframe point data
    KeyframePointData keyframePointData;
    keyframePointData.timestamp = timestamp;
    keyframePointData.frame_id = "world";
    
    // Process features for the point cloud
    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int frame_size = it_per_id.feature_per_frame.size();
        if(it_per_id.start_frame < WINDOW_SIZE - 2 && 
           it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && 
           it_per_id.solve_flag == 1)
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0])
                                  + estimator.Ps[imu_i];
            
            int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
            
            KeyframePointData::FeaturePoint feature;
            feature.position = w_pts_i;
            feature.point = it_per_id.feature_per_frame[imu_j].point;
            feature.uv = it_per_id.feature_per_frame[imu_j].uv;
            feature.feature_id = it_per_id.feature_id;
            
            keyframePointData.features.push_back(feature);
        }
    }

    // Data would have been published to:
    // - "/keyframe_pose" topic for keyframe pose
    // - "/keyframe_point" topic for keyframe points
    // Implement your custom handlers here
    
    // If you had callbacks, you could call them like:
    // pose_callback(keyframePoseData);
    // point_callback(keyframePointData);
}

void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;

    // Replace ROS debug with printf or std::cout
    printf("position: %f, %f, %f\n", 
           estimator.Ps[WINDOW_SIZE].x(), 
           estimator.Ps[WINDOW_SIZE].y(), 
           estimator.Ps[WINDOW_SIZE].z());
    printf("orientation: %f, %f, %f\n", 
           estimator.Vs[WINDOW_SIZE].x(), 
           estimator.Vs[WINDOW_SIZE].y(), 
           estimator.Vs[WINDOW_SIZE].z());
    
    if (ESTIMATE_EXTRINSIC)
    {
        cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            printf("extirnsic tic: %f, %f, %f\n", 
                   estimator.tic[i].x(),
                   estimator.tic[i].y(),
                   estimator.tic[i].z());
            
            Vector3d ypr = Utility::R2ypr(estimator.ric[i]);
            printf("extrinsic ric: %f, %f, %f\n", 
                   ypr.x(),
                   ypr.y(),
                   ypr.z());

            Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
            eigen_T.block<3, 3>(0, 0) = estimator.ric[i];
            eigen_T.block<3, 1>(0, 3) = estimator.tic[i];
            cv::Mat cv_T;
            cv::eigen2cv(eigen_T, cv_T);
            if(i == 0)
                fs << "body_T_cam0" << cv_T ;
            else
                fs << "body_T_cam1" << cv_T ;
        }
        fs.release();
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    printf("vo solver costs: %f ms\n", t);
    printf("average of time %f ms\n", sum_of_time / sum_of_calculation);

    sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
    last_path = estimator.Ps[WINDOW_SIZE];
    printf("sum of path %f\n", sum_of_path);
    if (ESTIMATE_TD)
        printf("td %f\n", estimator.td);
}