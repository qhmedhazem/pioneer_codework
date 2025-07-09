/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>
#include <opencv2/opencv.hpp>

#include "utility/CameraPoseVisualization.h"
#include "messages.h"

#include "estimator/estimator.h"
#include "parameters.h"

#include "estimator_node.h"
#include "pose_graph_node.h"
#include "globalopt_node.h"

using namespace Eigen;


// Static data need to be accessed from anywheres
extern PathData path;
extern int IMAGE_ROW, IMAGE_COL;
extern CameraPoseVisualization cameraposevisual;
extern double sum_of_path;
extern Vector3d last_path;
extern size_t pub_counter;

// Core data input functions
void pubImu(Eigen::Vector3d &acc, Eigen::Vector3d &gyr, double t);
void pubImage0(cv::Mat &img0, double t);
void pubImage1(cv::Mat &img1, double t);
void pubGps(double latitude, double longitude, double altitude, double posAccuracy, double t);

// Output/visualization functions
void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t);
void pubOdometry(const Estimator &estimator, double t);
void pubKeyPoses(const Estimator &estimator, double t);
void pubCameraPose(const Estimator &estimator, double t);
void pubPointCloud(const Estimator &estimator, double t);
void pubTF(const Estimator &estimator, double t);
void pubKeyframe(const Estimator &estimator);
void printStatistics(const Estimator &estimator, double t);

// Functions not yet implemented - if needed, update these with ROS-free interfaces
// void pubInitialGuess(const Estimator &estimator, double t);
// void pubRelocalization(const Estimator &estimator);
// void pubCar(const Estimator & estimator, double t);
