
#ifndef POSEGRAPH_NODE_H
#define POSEGRAPH_NODE_H

#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


#include "pose_graph_essentials.h"
#include "utility/tic_toc.h"

#include "messages.h"
#include "parameters.h"

#define SKIP_FIRST_CNT 10
using namespace std;
using namespace Eigen;

class PoseGraphNode {
    public:
        PoseGraphNode();
        ~PoseGraphNode();

        // Settings
        bool initialized = false;
        void initialize();
        void reset();
        void new_sequence();
        void process();
        void command();

        // Callback functions
        void image_callback(TimestampedImage &image_msg);
        void vio_callback(OdometryData &vio_msg);
        void pose_callback(OdometryData &pose_msg);
        void extrinsic_callback(ExtrinsicData &pose_msg);
        void point_callback(KeyframePointData &point_msg);
        void margin_point_callback(PointCloudData &point_msg);


        // Getters
        void wait_for_measurements(double range_t0);
        void rectify(OdometryData &vio_msg);
        
        // Entities
        // CameraPoseVisualization cameraposevisual(1, 0, 0, 1);

    private:
        // State
        double SKIP_DIS = 0;
        int SKIP_CNT = 0;
        int sequence = 1;
        int frame_index  = 0;
        int skip_first_cnt = 0;
        int skip_cnt = 0;
        bool load_flag = 0;
        bool start_flag = 0;
        double last_image_time = -1;
        double last_process_time = -1;

        // Processes
        std::mutex m_buf;
        std::mutex m_process;
        std::thread measurement_process;
        std::thread keyboard_command_process;

        // Buffers
        queue<TimestampedImage> image_buf;
        queue<OdometryData> pose_buf;
        queue<KeyframePointData> point_buf;
        queue<OdometryData> odometry_buf;
};


extern std::shared_ptr<PoseGraphNode> posegraph_node;


// Callbacks
// void image_callback(const sensor_msgs::ImageConstPtr &image_msg);
// void point_callback(const sensor_msgs::PointCloudConstPtr &point_msg);
// void margin_point_callback(const sensor_msgs::PointCloudConstPtr &point_msg);
// void pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg);
// void vio_callback(const nav_msgs::Odometry::ConstPtr &pose_msg);
// void extrinsic_callback(const nav_msgs::Odometry::ConstPtr &pose_msg);

// Initialization
// void init_posegraph_node(std::string config_file, ros::NodeHandle &n);


#endif