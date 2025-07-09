
#ifndef ESTIMATOR_NODE_H
#define ESTIMATOR_NODE_H

#include <stdio.h>
#include <iostream>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator/estimator.h"
#include "parameters.h"
#include "messages.h"
#include "utility/visualization.h"



class EstimatorNode {
    public:
        EstimatorNode();
        ~EstimatorNode();

        // Settings
        bool initialized = false;
        void initialize();
        void reset();

        // Callback functions
        void img0_callback(TimestampedImage img_msg);
        void img1_callback(TimestampedImage img_msg);
        void imu_callback(double t, Vector3d& acc, Vector3d& gyr);
        void restart_callback(bool restart_msg);
        void imu_switch_callback(bool switch_msg); 
        void cam_switch_callback(bool switch_msg);
        
        // Getters
        void wait_for_measurements(double range_t0); 
        OdometryData get_pose_in_world_frame();
        cv::Mat get_track_image();
        
        // Entities
        Estimator estimator;
    private:
        // Handlers
        std::thread sync_thread;
        void sync_process();

        // Buffers
        std::mutex m_buf;
        std::queue<TimestampedImage> img0_buf;
        std::queue<TimestampedImage> img1_buf;
};

extern std::shared_ptr<EstimatorNode> estimator_node;

#endif // ESTIMATOR_NODE_H