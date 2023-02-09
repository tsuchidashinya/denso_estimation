#pragma once
#include <common_srvs/SensorService.h>
#include <common_srvs/VisualizeCloud.h>
#include <common_srvs/VisualizeImage.h>
#include <common_srvs/SemanticSegmentationService.h>
#include <common_srvs/ObjectDetectionService.h>
#include <common_srvs/AccuracyIouService.h>
#include <common_srvs/Hdf5OpenAccService.h>
#include <common_srvs/Hdf5OpenSensorDataService.h>
#include <gazebo_model_pkg/decide_object_position.hpp>
#include <gazebo_model_pkg/gazebo_model_move.hpp>
#include <tf_package/tf_function.hpp>
#include <util_package/util.hpp>
#include <data_transform_pkg/data_3D_to_2D.hpp>
#include <data_transform_pkg/data_2D_to_3D.hpp>
#include <data_transform_pkg/func_data_convertion.hpp>
#include <opencv2/opencv.hpp>
#include <accuracy_package/accuracy_util.hpp>
#include <sensor_package/cloud_process.hpp>


class EstimationClient
{
public:
    EstimationClient(ros::NodeHandle &);
    void main();
    void set_paramenter();
    XmlRpc::XmlRpcValue param_list;
    int the_number_of_execute_;

private:
    ros::NodeHandle nh_, pnh_;
    ros::ServiceClient sensor_client_, object_detect_client_, visualize_client_,
    cloud_network_client_, accuracy_client_, hdf5_client_, vis_image_client_;
    std::string sensor_service_name_, object_detect_service_name_, visualize_service_name_,
    cloud_network_service_name_, accuracy_service_name_, hdf5_service_name_, vis_image_service_name_;
    std::string hdf5_open_file_path_;
    std::string object_detect_checkpoint_path_, semantic_checkpoint_path_;
    std::string object_detect_mode_;
    std::string world_frame_, sensor_frame_;
    int counter_;
    TfFunction tf_func_;
    UtilMsgData util_msg_data_;
    DecidePosition decide_gazebo_object_;
    std::string estimation_name_;
    int semantic_class_num_;
};