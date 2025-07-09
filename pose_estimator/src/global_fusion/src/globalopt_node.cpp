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

#include "globalopt_node.h"


GlobalOptimizerNode::GlobalOptimizerNode() {}
GlobalOptimizerNode::~GlobalOptimizerNode() {}

void GlobalOptimizerNode::initialize() {
    if(initialized == true)
        return;

    initialized = true;
}

void GlobalOptimizerNode::publish_car_model(double t, Eigen::Vector3d t_w_car, Eigen::Quaterniond q_w_car) {
    // !!!!!!!!!!!! DISABLED AS IT IS VISUALIZATION
    //     visualization_msgs::MarkerArray markerArray_msg;
    //     visualization_msgs::Marker car_mesh;
    //     car_mesh.header.stamp = ros::Time(t);
    //     car_mesh.header.frame_id = "world";
    //     car_mesh.type = visualization_msgs::Marker::MESH_RESOURCE;
    //     car_mesh.action = visualization_msgs::Marker::ADD;
    //     car_mesh.id = 0;

    //     car_mesh.mesh_resource = "package://global_fusion/models/car.dae";

    //     Eigen::Matrix3d rot;
    //     rot << 0, 0, -1, 0, -1, 0, -1, 0, 0;
        
    //     Eigen::Quaterniond Q;
    //     Q = q_w_car * rot; 
    //     car_mesh.pose.position.x    = t_w_car.x();
    //     car_mesh.pose.position.y    = t_w_car.y();
    //     car_mesh.pose.position.z    = t_w_car.z();
    //     car_mesh.pose.orientation.w = Q.w();
    //     car_mesh.pose.orientation.x = Q.x();
    //     car_mesh.pose.orientation.y = Q.y();
    //     car_mesh.pose.orientation.z = Q.z();

    //     car_mesh.color.a = 1.0;
    //     car_mesh.color.r = 1.0;
    //     car_mesh.color.g = 0.0;
    //     car_mesh.color.b = 0.0;

    //     float major_scale = 2.0;

    //     car_mesh.scale.x = major_scale;
    //     car_mesh.scale.y = major_scale;
    //     car_mesh.scale.z = major_scale;
    //     markerArray_msg.markers.push_back(car_mesh);
    //     pub_car.publish(markerArray_msg);
}

void GlobalOptimizerNode::gps_callback(GpsData &gps_data) {
    m_buf.lock();
    gpsQueue.push(gps_data);
    m_buf.unlock();
}


void GlobalOptimizerNode::wait_for_measurements(double range_t0) {
    while (globalEstimator.lastTimestamp <= (range_t0 - EPS)) {
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

OdometryData GlobalOptimizerNode::get_pose_in_world_frame() {
    OdometryData data;
    data.frame_id = "world";
    globalEstimator.getGlobalOdom(data.timestamp, data.position, data.orientation);
    return data;
}

void GlobalOptimizerNode::vio_callback(OdometryData &pose_msg) {
    double t = pose_msg.timestamp;
    Eigen::Vector3d vio_t = pose_msg.position;
    Eigen::Quaterniond vio_q = pose_msg.orientation;
    globalEstimator.inputOdom(t, vio_t, vio_q);
    last_vio_t = t;
    

    m_buf.lock();
    while(!gpsQueue.empty())
    {
        GpsData GPS_msg = gpsQueue.front();
        double gps_t = GPS_msg.timestamp;
        printf("vio t: %f, gps t: %f \n", t, gps_t);
        // 10ms sync tolerance
        if(gps_t >= t - 0.01 && gps_t <= t + 0.01)
        {
            //printf("receive GPS with timestamp %f\n", GPS_msg->header.stamp.toSec());
            double latitude = GPS_msg.latitude;
            double longitude = GPS_msg.longitude;
            double altitude = GPS_msg.altitude;
            //int numSats = GPS_msg.status.service;
            double pos_accuracy = GPS_msg.posAccuracy;
            if(pos_accuracy <= 0)
                pos_accuracy = 1;
            //printf("receive covariance %lf \n", pos_accuracy);
            //if(GPS_msg->status.status > 8)
                globalEstimator.inputGPS(t, latitude, longitude, altitude, pos_accuracy);
            gpsQueue.pop();
            break;
        }
        else if(gps_t < t - 0.01)
            gpsQueue.pop();
        else if(gps_t > t + 0.01)
            break;
    }
    m_buf.unlock();

    double global_ts;
    Eigen::Vector3d global_t;
    Eigen:: Quaterniond global_q;
    globalEstimator.getGlobalOdom(global_ts, global_t, global_q);

    OdometryData odometry;
    odometry.frame_id = "world";
    odometry.timestamp = global_ts;
    odometry.position = global_t;
    odometry.orientation = global_q;

    // ---------- Placeholder: Data published orignially to /global_odometry
    // pub_global_odometry.publish(odometry);
    // ----------

    // ---------- Placeholder: Data published orignially to /global_path
    // pub_global_path.publish(*global_path);
    // ----------

    // ---------- Placeholder: Data published orignially to /car_model
    // publish_car_model(t, global_t, global_q);
    // ----------


    // write result to file
    // std::ofstream foutC("/home/tony-ws1/output/vio_global.csv", ios::app);
    // foutC.setf(ios::fixed, ios::floatfield);
    // foutC.precision(0);
    // foutC << pose_msg.timestamp * 1e9 << ",";
    // foutC.precision(5);
    // foutC << global_t.x() << ","
    //         << global_t.y() << ","
    //         << global_t.z() << ","
    //         << global_q.w() << ","
    //         << global_q.x() << ","
    //         << global_q.y() << ","
    //         << global_q.z() << endl;
    // foutC.close();
}

std::shared_ptr<GlobalOptimizerNode> globalopt_node = std::make_shared<GlobalOptimizerNode>();
