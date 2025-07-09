
#include "utility/visualization.h"


// Static Data
PathData path;
double sum_of_path = 0;
Vector3d last_path(0.0, 0.0, 0.0);

CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
size_t pub_counter = 0;


// Publishers: Main Data Publishers

void pubImage0(cv::Mat &img0, double t)
{
    // Orignially This would have published to "/img0" topic
    TimestampedImage frame;
    frame.image = img0.clone();
    frame.timestamp = t;
    
    estimator_node->img0_callback(frame);
    posegraph_node->image_callback(frame);
}

void pubImage1(cv::Mat &img1, double t)
{
    // Oringinally This would have published to "/img1" topic
    TimestampedImage frame;
    frame.image = img1.clone();
    frame.timestamp = t;

    estimator_node->img1_callback(frame);
}

void pubImu(Eigen::Vector3d &acc, Eigen::Vector3d &gyr, double t)
{
    // Orignially This would have published to "/imu" topic
    estimator_node->imu_callback(t, acc, gyr);
}

void pubGps(double latitude, double longitude, double altitude, double posAccuracy, double t)
{
    // Orignially Data would be published to "/gps" topic
    GpsData data;
    data.latitude = latitude;
    data.longitude = longitude;
    data.altitude = altitude;
    data.posAccuracy = posAccuracy;
    data.timestamp = t;

    globalopt_node->gps_callback(data);
}


// Publishers: Odometry Publishers

void pubOdometry(const Estimator &estimator, double t)
{

    if(estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;

    Eigen::Vector3d p = estimator.Ps[WINDOW_SIZE];
    Eigen::Quaterniond q(estimator.Rs[WINDOW_SIZE]);
    Eigen::Vector3d v = estimator.Vs[WINDOW_SIZE];
    
    OdometryData odometry;
    odometry.position = p;
    odometry.orientation = q;
    odometry.velocity = v;
    odometry.timestamp = t;
    
    // -------------- Placeholder: published to /odometry
    posegraph_node->vio_callback(odometry);
    globalopt_node->vio_callback(odometry);
    // --------------

    TimestampedPosition pose_stamped;
    pose_stamped.position = p;
    pose_stamped.timestamp = t;
    path.positions.push_back(pose_stamped);

    // -------------- Placeholder: published to path /path

    // --------------
    
    // Optional: Write result to file
    std::ofstream foutC(VINS_RESULT_PATH, std::ios::app);
    foutC.setf(std::ios::fixed, std::ios::floatfield);
    foutC.precision(0);
    foutC << t * 1e9 << ",";
    foutC.precision(5);
    foutC << p.x() << ","
          << p.y() << ","
          << p.z() << ","
          << q.w() << ","
          << q.x() << ","
          << q.y() << ","
          << q.z() << ","
          << v.x() << ","
          << v.y() << ","
          << v.z() << "," << std::endl;
    foutC.close();
    
    // Debug output
    printf("time: %f, t: %f %f %f q: %f %f %f %f \n", t, p.x(), p.y(), p.z(),
                                                   q.w(), q.x(), q.y(), q.z());
}


void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t)
{
    OdometryData data;
    data.position = P;
    data.orientation = Q;
    data.velocity = V;
    data.timestamp = t;

    // -------------- Placeholder: published to path /imu_propagate

    // --------------
}


void pubKeyPoses(const Estimator &estimator, double t)
{
    // Data would have been published to "/key_poses" topic
    // !!!!!!!!!!!!!: DISABLED FOR BEING VISUALIZATION
    if (estimator.key_poses.size() == 0)
        return;
    
    // KeyPosesData keyPosesData;
    // keyPosesData.timestamp = t;
    // keyPosesData.frame_id = "world";
    
    // for (int i = 0; i <= WINDOW_SIZE; i++)
    // {
    //     Vector3d correct_pose = estimator.key_poses[i];
    //     keyPosesData.poses.push_back(correct_pose);
    // }
}

void pubCameraPose(const Estimator &estimator, double t)
{
    // Data would have been published to:
    // - "/camera_pose" topic for odometry
    // - "/camera_pose_visual" topic for visual markers
    // !!!!!!!!!!!!!: DISABLED FOR BEING VISUALIZATION
    int idx2 = WINDOW_SIZE - 1;

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        int i = idx2;
        Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);

        // Create data structure for camera pose
        OdometryData data;
        data.position = P;
        data.orientation = R;
        data.timestamp = t;
        
        // -------------- Placeholder: published to path /camera_pose

        // --------------

        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        
        if(STEREO)
        {
            Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[1];
            Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[1]);
            cameraposevisual.add_pose(P, R);
        }

        // -------------- Placeholder: published to /camera_pose_visual for visual markers

        // --------------
    }
}


void pubPointCloud(const Estimator &estimator, double t)
{
    PointCloudData pointCloudData;
    pointCloudData.timestamp = t;
    pointCloudData.frame_id = "world";

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

    // Placeholder: Data would have been published to "/point_cloud" topic for main point cloud

    //

    PointCloudData marginCloudData;
    marginCloudData.timestamp = t;
    marginCloudData.frame_id = "world";
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

    // Placeholder: Data would have been published to "/margin_cloud" topic for margined point cloud
    posegraph_node->margin_point_callback(marginCloudData);
    //    
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
    extrinsicData.frame_id = "world";
    extrinsicData.position = estimator.tic[0];
    extrinsicData.orientation = Quaterniond(estimator.ric[0]);
    extrinsicData.timestamp = t;

    // -------------- Placeholder: published to /extrinsic for transforms
    // extrinsic_callback(extrinsicData);
    posegraph_node->extrinsic_callback(extrinsicData);
    // --------------

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
    OdometryData odometry;
    odometry.position = P;
    odometry.orientation = R;
    odometry.timestamp = timestamp;
    odometry.frame_id = "world";

    // -------------- Placeholder: Data would have been published to "/keyframe_pose" topic for keyframe pose
    posegraph_node->pose_callback(odometry);
    // --------------

    // Create keyframe point data
    KeyframePointData keyframePointData;
    keyframePointData.timestamp = timestamp;
    keyframePointData.frame_id = "world";
    
    // Process features for the point cloud
    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int frame_size = it_per_id.feature_per_frame.size();
        if(
            it_per_id.start_frame < WINDOW_SIZE - 2 && 
            it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && 
            it_per_id.solve_flag == 1
        )
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0])
                                  + estimator.Ps[imu_i];
            
            int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
            
            FeaturePoint feature;
            feature.position = w_pts_i;
            feature.point = it_per_id.feature_per_frame[imu_j].point;
            feature.uv = it_per_id.feature_per_frame[imu_j].uv;
            feature.feature_id = it_per_id.feature_id;
            
            keyframePointData.features.push_back(feature);
        }
    }

    // -------------- Placeholder: Data would have been published to "/keyframe_point" topic for keyframe points
    // keyframe_point_callback(keyframePointData);
    posegraph_node->point_callback(keyframePointData);
    // --------------
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