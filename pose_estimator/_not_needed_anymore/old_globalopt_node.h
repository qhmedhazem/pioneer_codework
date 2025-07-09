#include <iostream>
#include <thread>
#include <stdio.h>

#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <fstream>
#include <queue>
#include <mutex>

#include "ros/ros.h"
#include "globalopt.h"


extern GlobalOptimization globalEstimator;
extern ros::Publisher pub_global_odometry, pub_global_path, pub_car;
extern ros::Subscriber sub_GPS, sub_vio;

// Callbacks
void GPS_callback(const sensor_msgs::NavSatFixConstPtr &GPS_msg);
void globalopt_vio_callback(const nav_msgs::Odometry::ConstPtr &pose_msg);

// Initalization
void init_globalopt_node(std::string config_file, ros::NodeHandle &n);