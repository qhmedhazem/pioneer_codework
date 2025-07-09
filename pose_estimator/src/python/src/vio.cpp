#include <ros/ros.h>

#include "vio.h"
#include "NDArrayConverter.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


VINSWrapper::VINSWrapper(std::string config_path)
    : config_path(std::move(config_path)) {}

VINSWrapper::~VINSWrapper() {}


/*
    Initialization
*/
bool VINSWrapper::initialize() {
        if (is_initialized) return false;

        // Reading Configuration
        readParameters(config_path);
        
        // Initalizing
        estimator_node->initialize();
        // posegraph_node->initialize();
        // globalopt_node->initialize();

        // Initalized
        is_initialized = true;
        return true;
}

bool VINSWrapper::reset() {
    if (!is_initialized) return false;
    estimator_node->reset();
    return true;
}


/*
    Inputs: Images
*/
bool VINSWrapper::input_image(double t, cv::Mat &img0) {
    if (!is_initialized || img0.empty()) return false;
    pubImage0(img0, t);
    return true;
}

bool VINSWrapper::input_image_stereo(double t, cv::Mat &img0, cv::Mat &img1) {
    if (!is_initialized || img0.empty() || img1.empty()) return false;
    pubImage0(img0, t);
    pubImage1(img1, t);
    return true;
}

bool VINSWrapper::wait_for_vio(double range_t0) {
    if (!is_initialized) return false;
    estimator_node->wait_for_measurements(range_t0);
    return true;
}


bool VINSWrapper::wait_for_optimization(double range_t0) {
    if (!is_initialized) return false;
    posegraph_node->wait_for_measurements(range_t0);
    return true;
}



/*
    Inputs: IMU
*/
bool VINSWrapper::input_imu(double t, Eigen::Vector3d &acc, Eigen::Vector3d &gyr) {
    if (!is_initialized) return false;
    pubImu(acc, gyr, t);
    return true;
}


/*
    Inputs: GPS
*/
bool VINSWrapper::input_gps(double t, double latitude, double longitude, double altitude, double posAccuracy) {
    if (!is_initialized) return false;
    pubGps(latitude, longitude, altitude, posAccuracy, t);
    return true;
}


/*
    Getters
*/
std::pair<bool, cv::Mat> VINSWrapper::get_track_image() {
    cv::Mat track_img;
    if (is_initialized) {
        track_img = estimator_node->estimator.featureTracker.getTrackImage();
    }
    return {!track_img.empty(), track_img.clone()};
}

std::pair<bool, Eigen::Matrix4d> VINSWrapper::get_pose_in_world_frame() {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
    if (!is_initialized) return {false, mat};

    OdometryData pose = estimator_node->get_pose_in_world_frame();
    printf("before rectify: %f %f %f\n", pose.position.x(), pose.position.y(), pose.position.z());
    posegraph_node->rectify(pose);
    printf("after rectify: %f %f %f\n", pose.position.x(), pose.position.y(), pose.position.z());
    mat = pose.toMatrix4d();

    return {true, mat};
}

std::pair<bool, Eigen::Matrix4d> VINSWrapper::get_pose_in_world_frame_at(int idx) {
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    if (!is_initialized) return {false, pose};
    estimator_node->estimator.getPoseInWorldFrame(idx, pose);
    return {true, pose};
}

PYBIND11_MODULE(vinspy, m) {
    NDArrayConverter::init_numpy();

    py::class_<VINSWrapper>(m, "Estimator")
        .def(py::init<const std::string&>())

        // Initalization, Deinitalization and settings
        .def("initialize", &VINSWrapper::initialize)
        .def("reset", &VINSWrapper::reset)
        // .def("change_sensor_type", &VINSWrapper::change_sensor_type, py::arg("use_imu"), py::arg("use_stereo"))
        
        // Inputs
            // Images
        .def("input_image", &VINSWrapper::input_image, py::arg("t"), py::arg("img0"))
        .def("input_image_stereo", &VINSWrapper::input_image_stereo, py::arg("t"), py::arg("img0"), py::arg("img1"))
            // IMU
        .def("input_imu", &VINSWrapper::input_imu)
            // GPS
        .def("input_gps", &VINSWrapper::input_gps)
            // Waiters
        .def("wait_for_vio", &VINSWrapper::wait_for_vio, py::arg("range_t0"))
        .def("wait_for_optimization", &VINSWrapper::wait_for_optimization, py::arg("range_t0"))


        // Getters
        .def("get_track_image", &VINSWrapper::get_track_image)
        .def("get_pose_in_world_frame", &VINSWrapper::get_pose_in_world_frame)
        .def("get_pose_in_world_frame_at", &VINSWrapper::get_pose_in_world_frame_at);
}