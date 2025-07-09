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

#include "pose_graph_node.h"
#include "keyframe.h"
#include "pose_graph.h"


PoseGraph posegraph;
camodocal::CameraPtr m_camera;
Eigen::Vector3d tic;
Eigen::Matrix3d qic;
Eigen::Vector3d last_t(-100, -100, -100);


PoseGraphNode::PoseGraphNode() {}
PoseGraphNode::~PoseGraphNode() {}

void PoseGraphNode::initialize() {
    if(initialized == true)
        return;

    if (LOAD_PREVIOUS_POSE_GRAPH)
    {
        printf("load pose graph\n");
        m_process.lock();
        posegraph.loadPoseGraph();
        m_process.unlock();
        printf("load pose graph finish\n");
        load_flag = 1;
    }
    else
    {
        printf("no previous pose graph\n");
        load_flag = 1;
    }

    // printf("loading vocabulary file %s\n", VOCABULARY_FILE.c_str());
    // posegraph.loadVocabulary(VOCABULARY_FILE);
    m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(CAM_NAMES[0].c_str());

    // cameraposevisual.setScale(0.1);
    // cameraposevisual.setLineWidth(0.01);

    // measurement_process = std::thread(&PoseGraphNode::process, this);
    // measurement_process.detach();
    // keyboard_command_process = std::thread(&PoseGraphNode::command, this);
    // keyboard_command_process.detach();

    initialized = true;
}



void PoseGraphNode::reset() {}

void PoseGraphNode::new_sequence()
{
    sequence++;

    printf("new sequence\n");
    printf("sequence cnt %d \n", sequence);

    if (sequence > 5)
    {
        ROS_WARN("only support 5 sequences since it's boring to copy code for more sequences.");
        ROS_BREAK();
    }

    posegraph.posegraph_visualization->reset();
    posegraph.publish();

    m_buf.lock();
    while(!image_buf.empty())
        image_buf.pop();
    while(!point_buf.empty())
        point_buf.pop();
    while(!pose_buf.empty())
        pose_buf.pop();
    while(!odometry_buf.empty())
        odometry_buf.pop();
    m_buf.unlock();
}

void PoseGraphNode::image_callback(TimestampedImage &image_msg)
{
    //ROS_INFO("image_callback!");
    m_buf.lock();
    image_msg.format();
    image_buf.push(image_msg);
    m_buf.unlock();
    //printf(" image time %f \n", std::get<0>(image_msg));

    double time = image_msg.timestamp;

    if (last_image_time == -1)
        last_image_time = time;
    else if (time - last_image_time > 1.0 || time < last_image_time)
    {
        ROS_WARN("image discontinue! detect a new sequence!");
        new_sequence();
    }
    last_image_time = time;
}

void PoseGraphNode::vio_callback(OdometryData &vio_msg)
{
    Vector3d vio_t = vio_msg.position;
    Quaterniond vio_q = vio_msg.orientation;
    double t = vio_msg.timestamp;

    vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
    vio_q = posegraph.w_r_vio * vio_q;

    vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
    vio_q = posegraph.r_drift * vio_q;
    // pub_odometry_rect.publish(vio_t, vio_q, t);

    Vector3d vio_t_cam;
    Quaterniond vio_q_cam;
    vio_t_cam = vio_t + vio_q * tic;
    vio_q_cam = vio_q * qic;        

    // cameraposevisual.reset();
    // cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
    // cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg->header);
}


void PoseGraphNode::wait_for_measurements(double range_t0) {
    while (last_process_time >= (range_t0 - EPS)) {
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

void PoseGraphNode::rectify(
    OdometryData &data
) {
    Vector3d vio_t = data.position;
    Quaterniond vio_q = data.orientation;

    vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
    vio_q = posegraph.w_r_vio * vio_q;

    vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
    vio_q = posegraph.r_drift * vio_q;
    // pub_odometry_rect.publish(vio_t, vio_q, t);

    data.position = vio_t;
    data.orientation = vio_q;
}


void PoseGraphNode::pose_callback(OdometryData &pose_msg)
{
    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
}


void PoseGraphNode::extrinsic_callback(ExtrinsicData &pose_msg)
{
    m_process.lock();
    tic = pose_msg.position;
    qic = pose_msg.orientation.toRotationMatrix();
    m_process.unlock();
}


void PoseGraphNode::point_callback(KeyframePointData &point_msg)
{
    m_buf.lock();
    point_buf.push(point_msg);
    m_buf.unlock();
}


void PoseGraphNode::margin_point_callback(PointCloudData &point_msg)
{
    // !!!!!!!!!!!!!!!!DISABLE FOR VISUALIZATION
}



// OdometryData PoseGraphNode::get_pose_in_world_frame() {
//     OdometryData odometry;
//     odometry.frame_id = "world";
//     odometry.timestamp = estimator.curTime;
//     odometry.position = estimator.Ps[estimator.frame_count];
//     odometry.orientation = estimator.Qs[estimator.frame_count];

//     return odometry;
// }


void PoseGraphNode::process()
{
    while (true)
    {
        bool has_img_msg = false;
        bool has_point_msg = false;
        bool has_pose_msg = false;
        
        TimestampedImage image_msg;
        KeyframePointData point_msg;
        OdometryData pose_msg;

        // find out the messages with same time stamp
        m_buf.lock();
        if(!image_buf.empty() && !point_buf.empty() && !pose_buf.empty())
        {
            if (image_buf.front().timestamp > pose_buf.front().timestamp)
            {
                pose_buf.pop();
                printf("throw pose at beginning\n");
            }
            else if (image_buf.front().timestamp > point_buf.front().timestamp)
            {
                point_buf.pop();
                printf("throw point at beginning\n");
            }
            else if (image_buf.back().timestamp >= pose_buf.front().timestamp 
                && point_buf.back().timestamp >= pose_buf.front().timestamp)
            {
                pose_msg = pose_buf.front();
                has_pose_msg = true;
                pose_buf.pop();
                while (!pose_buf.empty())
                    pose_buf.pop();
                while (image_buf.front().timestamp < pose_msg.timestamp)
                    image_buf.pop();
                image_msg = image_buf.front();
                has_img_msg = true;
                image_buf.pop();

                while (point_buf.front().timestamp < pose_msg.timestamp)
                    point_buf.pop();
                point_msg = point_buf.front();
                has_point_msg = true;
                point_buf.pop();
            }
        }
        m_buf.unlock();

        if (has_pose_msg)
        {
            //printf(" pose time %f \n", pose_msg->header.timestamp);
            //printf(" point time %f \n", point_msg->header.timestamp);
            //printf(" image time %f \n", image_msg->header.timestamp);
            // skip fisrt few
            if (skip_first_cnt < SKIP_FIRST_CNT)
            {
                skip_first_cnt++;
                continue;
            }

            if (skip_cnt < SKIP_CNT)
            {
                skip_cnt++;
                continue;
            }
            else
            {
                skip_cnt = 0;
            }

            cv::Mat image = image_msg.image;
            // build keyframe
            Vector3d T = pose_msg.position;
            Matrix3d R = pose_msg.orientation.toRotationMatrix();
            if((T - last_t).norm() > SKIP_DIS)
            {
                vector<cv::Point3f> point_3d; 
                vector<cv::Point2f> point_2d_uv; 
                vector<cv::Point2f> point_2d_normal;
                vector<double> point_id;

                for (unsigned int i = 0; i < point_msg.features.size(); i++)
                {
                    cv::Point3f p_3d;
                    p_3d.x = point_msg.features[i].position.x();
                    p_3d.y = point_msg.features[i].position.y();
                    p_3d.z = point_msg.features[i].position.z();
                    point_3d.push_back(p_3d);

                    cv::Point2f p_2d_uv, p_2d_normal;
                    double p_id;
                    p_2d_normal.x = point_msg.features[i].point.x();
                    p_2d_normal.y = point_msg.features[i].point.y();
                    p_2d_uv.x = point_msg.features[i].uv.x();
                    p_2d_uv.y = point_msg.features[i].uv.y();
                    p_id = point_msg.features[i].feature_id;
                    point_2d_normal.push_back(p_2d_normal);
                    point_2d_uv.push_back(p_2d_uv);
                    point_id.push_back(p_id);

                    //printf("u %f, v %f \n", p_2d_uv.x, p_2d_uv.y);
                }

                KeyFrame* keyframe = new KeyFrame(pose_msg.timestamp, frame_index, T, R, image,
                                   point_3d, point_2d_uv, point_2d_normal, point_id, sequence);   
                m_process.lock();
                start_flag = 1;
                posegraph.addKeyFrame(keyframe, 1);
                m_process.unlock();
                frame_index++;
                last_t = T;
            }

            last_process_time = pose_msg.timestamp;
        }

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}



void PoseGraphNode::command()
{
    while(1)
    {
        char c = getchar();
        if (c == 's')
        {
            m_process.lock();
            posegraph.savePoseGraph();
            m_process.unlock();
            printf("save pose graph finish\nyou can set 'load_previous_pose_graph' to 1 in the config file to reuse it next time\n");
            printf("program shutting down...\n");
            ros::shutdown();
        }
        if (c == 'n')
            new_sequence();

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
};


std::shared_ptr<PoseGraphNode> posegraph_node = std::make_shared<PoseGraphNode>();





// void init_posegraph_node(std::string config_file, ros::NodeHandle &n)
// {
//     posegraph.registerPub(n);

//     printf("config_file: %s\n", config_file);
        
//     cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
//     if(!fsSettings.isOpened())
//     {
//         std::cerr << "ERROR: Wrong path to settings" << std::endl;
//     }
        
//         // cameraposevisual.setScale(0.1);
//         // cameraposevisual.setLineWidth(0.01);


//     fsSettings["pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;
//     fsSettings["output_path"] >> VINS_RESULT_PATH;
//     fsSettings["save_image"] >> DEBUG_IMAGE;
    
//     LOAD_PREVIOUS_POSE_GRAPH = fsSettings["load_previous_pose_graph"];
//     VINS_RESULT_PATH = VINS_RESULT_PATH + "/vio_loop.csv";
//     std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
//     fout.close();
    
//     int USE_IMU = fsSettings["imu"];
//     posegraph.setIMUFlag(USE_IMU);
//     fsSettings.release();
        

//     ros::Subscriber sub_vio = n.subscribe("/odometry", 2000, vio_callback);
//     ros::Subscriber sub_image = n.subscribe("/img0", 2000, image_callback);
//     ros::Subscriber sub_pose = n.subscribe("/keyframe_pose", 2000, pose_callback);
//     ros::Subscriber sub_extrinsic = n.subscribe("/extrinsic", 2000, extrinsic_callback);
//     ros::Subscriber sub_point = n.subscribe("/keyframe_point", 2000, point_callback);
//     ros::Subscriber sub_margin_point = n.subscribe("/margin_cloud", 2000, margin_point_callback);
    
//     pub_match_img = n.advertise<sensor_msgs::Image>("/match_image", 1000);
//     pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("/camera_pose_visual", 1000);
//     pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("/point_cloud_loop_rect", 1000);
//     pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("/margin_cloud_loop_rect", 1000);
//     pub_odometry_rect = n.advertise<nav_msgs::Odometry>("/odometry_rect", 1000);
    
//     measurement_process = std::thread(process);
//     measurement_process.detach();
//     keyboard_command_process = std::thread(command);
//     keyboard_command_process.detach();

// }

// int main(int argc, char **argv) {
//     if(argc != 2)
//     {
//         printf("please intput: rosrun loop_fusion loop_fusion_node [config file] \n"
//             "for example: rosrun loop_fusion loop_fusion_node "
//             "/home/tony-ws1/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
//             return 0;
//         }
        
//     ros::init(argc, argv, "loop_fusion");
//     ros::NodeHandle n("~");
//     init_posegraph_node(argv[1], n);
//     ros::spin();

//     return 0;
// }