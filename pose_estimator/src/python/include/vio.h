#pragma once

#include <string>
#include <memory>
#include <utility>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <thread>

#include "utility/visualization.h"
#include "parameters.h"

#include "estimator_node.h"
#include "pose_graph_node.h"
#include "globalopt_node.h"


class VINSWrapper {
public:
    VINSWrapper(std::string config_path);
    ~VINSWrapper();


    // Initalization, Deinitalization and settings
    bool initialize();
    bool reset();

    // Inputs
    bool input_image(double t, cv::Mat &img0);
    bool input_image_stereo(double t, cv::Mat &img0, cv::Mat &img1);
    bool input_imu(double t, Eigen::Vector3d &acc, Eigen::Vector3d &gyr);
    bool input_gps(double t, double latitude, double longitude, double altitude, double posAccuracy);

    // Waiters
    bool wait_for_vio(double range_t0);
    bool wait_for_optimization(double range_t0);

    // Getters
    std::pair<bool, cv::Mat> get_track_image();
    std::pair<bool, Eigen::Matrix4d> get_pose_in_world_frame();
    std::pair<bool, Eigen::Matrix4d> get_pose_in_world_frame_at(int idx);

    bool is_initialized = false;

private:
    // Configs
    std::string config_path;
};
