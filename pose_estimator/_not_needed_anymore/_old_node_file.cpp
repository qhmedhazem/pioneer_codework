// /*******************************************************
//  * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
//  * 
//  * This file is part of VINS.
//  * 
//  * Licensed under the GNU General Public License v3.0;
//  * you may not use this file except in compliance with the License.
//  *
//  * Author: Qin Tong (qintonguav@gmail.com)
//  *******************************************************/

// #include "estimator_node.h"

// EstimatorNode::EstimatorNode() {
//     sync_thread = std::thread(&EstimatorNode::sync_process, this);
//     sync_thread.detach();
// }

// EstimatorNode::~EstimatorNode() {}

// void EstimatorNode::img0_callback(const sensor_msgs::ImageConstPtr &img_msg) {
//     printf("received img0 at time %f\n", img_msg->header.stamp.toSec());
//     std::lock_guard<std::mutex> lock(m_buf);
//     img0_buf.push(img_msg);
// }

// void EstimatorNode::img1_callback(const sensor_msgs::ImageConstPtr &img_msg) {
//     printf("received img1 at time %f\n", img_msg->header.stamp.toSec());
//     std::lock_guard<std::mutex> lock(m_buf);
//     img1_buf.push(img_msg);
// }

// cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg) {
//     cv_bridge::CvImageConstPtr ptr;
//     if (img_msg->encoding == "8UC1") {
//         sensor_msgs::Image img;
//         img.header = img_msg->header;
//         img.height = img_msg->height;
//         img.width = img_msg->width;
//         img.is_bigendian = img_msg->is_bigendian;
//         img.step = img_msg->step;
//         img.data = img_msg->data;
//         img.encoding = "mono8";
//         ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
//     } else {
//         ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
//     }
//     return ptr->image.clone();
// }

// void EstimatorNode::sync_process() {
//     while (1) {
//         if (STEREO) {
//             cv::Mat image0, image1;
//             std_msgs::Header header;
//             double time = 0;
//             m_buf.lock();
//             if (!img0_buf.empty() && !img1_buf.empty()) {
//                 double time0 = img0_buf.front()->header.stamp.toSec();
//                 double time1 = img1_buf.front()->header.stamp.toSec();
//                 if (time0 < time1 - 0.003) {
//                     img0_buf.pop();
//                     printf("throw img0\n");
//                 } else if (time0 > time1 + 0.003) {
//                     img1_buf.pop();
//                     printf("throw img1\n");
//                 } else {
//                     time = img0_buf.front()->header.stamp.toSec();
//                     header = img0_buf.front()->header;
//                     image0 = getImageFromMsg(img0_buf.front());
//                     img0_buf.pop();
//                     image1 = getImageFromMsg(img1_buf.front());
//                     img1_buf.pop();
//                 }
//             }
//             m_buf.unlock();
//             if (!image0.empty())
//                 estimator.inputImage(time, image0, image1);
//         } else {
//             cv::Mat image;
//             std_msgs::Header header;
//             double time = 0;
//             m_buf.lock();
//             if (!img0_buf.empty()) {
//                 time = img0_buf.front()->header.stamp.toSec();
//                 header = img0_buf.front()->header;
//                 image = getImageFromMsg(img0_buf.front());
//                 img0_buf.pop();
//             }
//             m_buf.unlock();
//             if (!image.empty())
//                 estimator.inputImage(time, image);
//         }
//         std::chrono::milliseconds dura(2);
//         std::this_thread::sleep_for(dura);
//     }
// }

// void EstimatorNode::imu_callback(const sensor_msgs::ImuConstPtr &imu_msg) {
//     double t = imu_msg->header.stamp.toSec();
//     double dx = imu_msg->linear_acceleration.x;
//     double dy = imu_msg->linear_acceleration.y;
//     double dz = imu_msg->linear_acceleration.z;
//     double rx = imu_msg->angular_velocity.x;
//     double ry = imu_msg->angular_velocity.y;
//     double rz = imu_msg->angular_velocity.z;
//     Vector3d acc(dx, dy, dz);
//     Vector3d gyr(rx, ry, rz);
//     estimator.inputIMU(t, acc, gyr);
// }

// void EstimatorNode::feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg) {
//     map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
//     for (unsigned int i = 0; i < feature_msg->points.size(); i++) {
//         int feature_id = feature_msg->channels[0].values[i];
//         int camera_id = feature_msg->channels[1].values[i];
//         double x = feature_msg->points[i].x;
//         double y = feature_msg->points[i].y;
//         double z = feature_msg->points[i].z;
//         double p_u = feature_msg->channels[2].values[i];
//         double p_v = feature_msg->channels[3].values[i];
//         double velocity_x = feature_msg->channels[4].values[i];
//         double velocity_y = feature_msg->channels[5].values[i];
//         if (feature_msg->channels.size() > 5) {
//             double gx = feature_msg->channels[6].values[i];
//             double gy = feature_msg->channels[7].values[i];
//             double gz = feature_msg->channels[8].values[i];
//             pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
//         }
//         ROS_ASSERT(z == 1);
//         Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
//         xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
//         featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
//     }
//     double t = feature_msg->header.stamp.toSec();
//     estimator.inputFeature(t, featureFrame);
// }

// void EstimatorNode::restart_callback(const std_msgs::BoolConstPtr &restart_msg) {
//     if (restart_msg->data == true) {
//         ROS_WARN("restart the estimator!");
//         estimator.clearState();
//         estimator.setParameter();
//     }
// }

// void EstimatorNode::imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg) {
//     if (switch_msg->data == true) {
//         estimator.changeSensorType(1, STEREO);
//     } else {
//         estimator.changeSensorType(0, STEREO);
//     }
// }

// void EstimatorNode::cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg) {
//     if (switch_msg->data == true) {
//         estimator.changeSensorType(USE_IMU, 1);
//     } else {
//         estimator.changeSensorType(USE_IMU, 0);
//     }
// }

// EstimatorNode estimator_node;

// void init_estimator_node(std::string config_file, ros::NodeHandle &n) {
//     printf("config_file: %s\n", config_file.c_str());
//     readParameters(config_file);
//     estimator_node.estimator.setParameter();
//     #ifdef EIGEN_DONT_PARALLELIZE
//     ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
//     #endif
//     ROS_WARN("waiting for image and imu...");
//     ros::Subscriber sub_imu;
//     if (USE_IMU) {
//         sub_imu = n.subscribe("/imu", 2000, &EstimatorNode::imu_callback, &estimator_node, ros::TransportHints().tcpNoDelay());
//     }
//     ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2000, &EstimatorNode::feature_callback, &estimator_node);
//     ros::Subscriber sub_img0 = n.subscribe("/img0", 100, &EstimatorNode::img0_callback, &estimator_node);
//     ros::Subscriber sub_img1;
//     printf("Subscribed to IMU topic: %s", sub_imu.getTopic().c_str());
//     printf("Subscribed to IMG0 topic: %s", sub_img0.getTopic().c_str());
//     if (STEREO) {
//         sub_img1 = n.subscribe("/img1", 100, &EstimatorNode::img1_callback, &estimator_node);
//         printf("Subscribed to IMG1 topic: %s", sub_img1.getTopic().c_str());
//     }
//     ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, &EstimatorNode::restart_callback, &estimator_node);
//     ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, &EstimatorNode::imu_switch_callback, &estimator_node);
//     ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, &EstimatorNode::cam_switch_callback, &estimator_node);
//     printf("sync thread started!\n");
// }

// int main(int argc, char **argv) {
//     if (argc != 2) {
//         printf("please intput: rosrun vins vins_node [config file] \n"
//                "for example: rosrun vins vins_node "
//                "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
//         return 1;
//     }
//     ros::init(argc, argv, "vins_estimator");
//     ros::NodeHandle n("~");
//     init_estimator_node(argv[1], n);
//     ros::spin();
//     return 0;
// }