/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "estimator_node.h"

EstimatorNode::EstimatorNode() {}
EstimatorNode::~EstimatorNode() {}

void EstimatorNode::initialize() {
    if(initialized == false) {
        reset();
        sync_thread = std::thread(&EstimatorNode::sync_process, this);
        sync_thread.detach();
        initialized = true;
    }
}


void EstimatorNode::reset() {
    std::lock_guard<std::mutex> lock(m_buf);
    while (!img0_buf.empty()) img0_buf.pop();
    while (!img1_buf.empty()) img1_buf.pop();

    pts_gt.clear();
    estimator.setParameter();
    estimator.clearState();

    printf("EstimatorNode reset!\n");
}

void EstimatorNode::img0_callback(TimestampedImage img_msg) {
    printf("received img0 at time %f\n", img_msg.timestamp);
    std::lock_guard<std::mutex> lock(m_buf);
    img_msg.format();
    img0_buf.push(img_msg);
}

void EstimatorNode::img1_callback(TimestampedImage img_msg) {
    printf("received img1 at time %f\n", img_msg.timestamp);
    std::lock_guard<std::mutex> lock(m_buf);
    img_msg.format();
    img1_buf.push(img_msg);
}

void EstimatorNode::imu_callback(double t, Vector3d& acc, Vector3d& gyr) {
    estimator.inputIMU(t, acc, gyr);
}

void EstimatorNode::wait_for_measurements(double range_t0) {
    while (estimator.curTime >= (range_t0 + TD - EPS)) {
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

OdometryData EstimatorNode::get_pose_in_world_frame() {
    OdometryData odometry;
    odometry.frame_id = "world";
    odometry.timestamp = estimator.curTime;
    odometry.position = estimator.Ps[estimator.frame_count];
    odometry.orientation = estimator.Rs[estimator.frame_count];
    return odometry;
}

void EstimatorNode::restart_callback(bool restart_msg) {
    if (restart_msg == true) {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
}

void EstimatorNode::imu_switch_callback(bool switch_msg) {
    if (switch_msg == true) {
        estimator.changeSensorType(1, STEREO);
    } else {
        estimator.changeSensorType(0, STEREO);
    }
}

void EstimatorNode::cam_switch_callback(bool switch_msg) {
    if (switch_msg == true) {
        estimator.changeSensorType(USE_IMU, 1);
    } else {
        estimator.changeSensorType(USE_IMU, 0);
    }
}

cv::Mat EstimatorNode::get_track_image() {
    return estimator.featureTracker.getTrackImage();
}

void EstimatorNode::sync_process() {
    while (1) {
        if (STEREO) {
            m_buf.lock();
            if (!img0_buf.empty() && !img1_buf.empty()) {
                double time0 = img0_buf.front().timestamp;
                cv::Mat image0 = img0_buf.front().image;
                double time1 = img1_buf.front().timestamp;
                cv::Mat image1 = img1_buf.front().image;

                img0_buf.pop();
                img1_buf.pop();

                if (time0 < time1 - 0.003) {
                    // we can not handle this without the first image
                } else if (time0 > time1 + 0.003) {
                    estimator.inputImage(time0, image0, cv::Mat());
                } else {
                    estimator.inputImage(time0, image0, image1);
                }
            }
            m_buf.unlock();
        } else {
            m_buf.lock();
            if (!img0_buf.empty()) {
                double time = img0_buf.front().timestamp;
                cv::Mat image = img0_buf.front().image;
                estimator.inputImage(time, image);
                img0_buf.pop();
            }
            m_buf.unlock();
        }
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

std::shared_ptr<EstimatorNode> estimator_node = std::make_shared<EstimatorNode>();