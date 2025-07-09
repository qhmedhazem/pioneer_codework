#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <fstream>
#include <vector>
#include <string>
#include <tuple>

using namespace Eigen;


struct TimestampedImage {
    double timestamp;
    cv::Mat image;

    void format() {
        if (image.channels() == 1)
            return;  // already grayscale, nothing to do
        int conversion_code = (image.channels() == 3) ? cv::COLOR_BGR2GRAY :
                              (image.channels() == 4) ? cv::COLOR_BGRA2GRAY : -1;
        if (conversion_code == -1)
            throw std::runtime_error("Unsupported image format: must be 1, 3, or 4 channels.");
        cv::Mat gray_image;
        cv::cvtColor(image, gray_image, conversion_code);
        image = std::move(gray_image);
    }
};

// Path data structure 
struct TimestampedPosition {
    Vector3d position;
    double timestamp;
};


// Odometry data
struct OdometryData {
    Vector3d position;
    Quaterniond orientation;
    Vector3d velocity;
    std::string frame_id;
    double timestamp;

    Eigen::Matrix4d toMatrix4d() const {
        Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
        mat.block<3, 3>(0, 0) = orientation.toRotationMatrix();
        mat.block<3, 1>(0, 3) = position;
        return mat;
    }
};

struct GlobalPathData {
    std::vector<OdometryData> positions;
};

struct PathData {
    std::vector<TimestampedPosition> positions;
    std::vector<double> timestamps;
};



// GPS data
struct GpsData {
    double latitude;
    double longitude;
    double altitude;
    double posAccuracy;
    double timestamp;
};

// Key poses data
struct KeyPosesData {
    std::vector<Vector3d> poses;
    double timestamp;
    std::string frame_id;
};

// Camera pose data
// struct CameraPoseData {
//     std::vector<Vector3d> positions;
//     std::vector<Quaterniond> orientations;
//     double timestamp;
//     std::string frame_id;
// };

// Point cloud data
struct PointCloudData {
    std::vector<Vector3d> points;
    double timestamp;
    std::string frame_id;
};

// Transform data
struct TransformData {
    Vector3d translation;
    Quaterniond rotation;
    // The parent and child frames for the transform
    std::string parent_frame;
    std::string child_frame;
    // Timestamp for the transform
    double timestamp;
};

// Extrinsic data
struct ExtrinsicData {
    Vector3d position;
    Quaterniond orientation;
    double timestamp;
    std::string frame_id;
};


struct FeaturePoint {
        Vector3d position;   // 3D world position
        Vector3d point;      // Normalized camera coordinates
        Vector2d uv;         // Pixel coordinates
        int feature_id;
    };

// Keyframe point data
struct KeyframePointData {
    std::vector<FeaturePoint> features;
    double timestamp;
    std::string frame_id;
};
