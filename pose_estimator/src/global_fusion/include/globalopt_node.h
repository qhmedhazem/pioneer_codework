
#pragma once

#include <iostream>
#include <thread>
#include <stdio.h>
#include <fstream>
#include <queue>
#include <mutex>

#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "ros/ros.h"

#include "globalopt.h"
#include "parameters.h"
#include "messages.h"


class GlobalOptimizerNode
{
  public:
    GlobalOptimizerNode();
    ~GlobalOptimizerNode();

    bool initialized = false;
    void initialize();

    // Callback
    void gps_callback(GpsData &gps_data);
    void vio_callback(OdometryData &pose_msg);
    
    // Data Getters
    void wait_for_measurements(double range0);
    OdometryData get_pose_in_world_frame();

    // Others
    void publish_car_model(double t, Eigen::Vector3d t_w_car, Eigen::Quaterniond q_w_car);

    // Entities
    GlobalOptimization globalEstimator;
  private:
    double last_vio_t = -1;
    std::mutex m_buf;
    std::queue<GpsData> gpsQueue;
};


extern std::shared_ptr<GlobalOptimizerNode> globalopt_node;