#include <ros/ros.h>

#include "vio.h"
#include "NDArrayConverter.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

bool is_initialized = false;

VINSWrapper::VINSWrapper(std::string config_path)
    : config_path(std::move(config_path)) {}

VINSWrapper::~VINSWrapper() {}

bool VINSWrapper::initialize() {
        if (is_initialized) return false;
        readParameters(config_path);

        int argc = 2;
        char* argv[2];
        argv[0] = const_cast<char*>("vinspy");
        argv[1] = const_cast<char*>(config_path.c_str());


        ros::init(argc, argv, "vinspy");
        ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
        ros::NodeHandle n("~");
        
        estimator_node->reset();
        init_posegraph_node(config_path, n);
        init_globalopt_node(config_path, n);
        is_initialized = true;
        
        static ros::AsyncSpinner spinner(2);
        spinner.start();
        printf("done starting spinner\n");

        return true;
}

bool VINSWrapper::reset() {
    if (!is_initialized) return false;
    estimator_node->reset();
    return true;
}

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

bool VINSWrapper::input_imu(double t, Eigen::Vector3d &acc, Eigen::Vector3d &gyr) {
    if (!is_initialized) return false;
    printf("from c++, sent imu at t: %f\n", t);
    pubImu(acc, gyr, t);
    return true;
}

std::pair<bool, cv::Mat> VINSWrapper::get_track_image() {
    if (!is_initialized) return {false, cv::Mat()};
    cv::Mat track_img = estimator_node->estimator.featureTracker.getTrackImage();
    if (track_img.empty())
        return {false, cv::Mat()};
    return {true, track_img.clone()};
}

std::pair<bool, Eigen::Matrix4d> VINSWrapper::get_pose_in_world_frame() {
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    if (!is_initialized) return {false, pose};
    estimator_node->estimator.getPoseInWorldFrame(pose);
    return {true, pose};
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
        .def("change_sensor_type", &VINSWrapper::change_sensor_type, py::arg("use_imu"), py::arg("use_stereo"))
        
        // Inputs
        .def("input_image", &VINSWrapper::input_image, py::arg("t"), py::arg("img0"))
        .def("input_image_stereo", &VINSWrapper::input_image_stereo, py::arg("t"), py::arg("img0"), py::arg("img1"))
        .def("input_imu", &VINSWrapper::input_imu)

        // Getters
        .def("get_track_image", &VINSWrapper::get_track_image)
        .def("get_pose_in_world_frame", &VINSWrapper::get_pose_in_world_frame)
        .def("get_pose_in_world_frame_at", &VINSWrapper::get_pose_in_world_frame_at);
}