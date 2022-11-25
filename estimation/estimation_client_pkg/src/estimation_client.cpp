#include <estimation_client_pkg/estimation_client.hpp>

EstimationClient::EstimationClient(ros::NodeHandle &nh) :
nh_(nh),
pnh_("~")
{
    set_paramenter();
    sensor_client_ = nh_.serviceClient<common_srvs::SensorService>(sensor_service_name_);
    object_detect_client_ = nh_.serviceClient<common_srvs::ObjectDetectionService>(object_detect_service_name_);
    cloud_network_client_ = nh_.serviceClient<common_srvs::SemanticSegmentationService>(cloud_network_service_name_);
    visualize_client_ = nh_.serviceClient<common_srvs::VisualizeCloud>(visualize_service_name_);
    accuracy_client_ = nh_.serviceClient<common_srvs::AccuracyIouService>(accuracy_service_name_);
    hdf5_client_ = nh_.serviceClient<common_srvs::Hdf5OpenService>(hdf5_service_name_);
}

void EstimationClient::set_paramenter()
{
    pnh_.getParam("estimation_main", param_list);
    visualize_service_name_ = static_cast<std::string>(param_list["visualize_service_name"]);
    sensor_service_name_ = static_cast<std::string>(param_list["sensor_service_name"]);
    object_detect_service_name_ = static_cast<std::string>(param_list["object_detect_service_name"]);
    cloud_network_service_name_ = static_cast<std::string>(param_list["cloud_network_service_name"]);
    accuracy_service_name_ = static_cast<std::string>(param_list["accuracy_service_name"]);
    hdf5_service_name_ = static_cast<std::string>(param_list["hdf5_service_name"]);
    the_number_of_execute_ = param_list["the_number_of_execute"];
}

void EstimationClient::main()
{
    DecidePosition decide_gazebo_object;
    GazeboMoveServer gazebo_model_move(nh_);
    std::vector<common_msgs::ObjectInfo> multi_object;
    for (int i = 0; i < 10; i++) {
        common_msgs::ObjectInfo object;
        object = decide_gazebo_object.make_object_info(i, "HV8");
        multi_object.push_back(object);
    }
    multi_object = decide_gazebo_object.get_remove_position(multi_object);
    gazebo_model_move.set_multi_gazebo_model(multi_object);
    multi_object = decide_gazebo_object.get_randam_place_position(multi_object);
    gazebo_model_move.set_multi_gazebo_model(multi_object);
    ros::Duration(0.5).sleep();

    common_srvs::SensorService sensor_srv;
    sensor_srv.request.counter = 1;
    Util::client_request(sensor_client_, sensor_srv, sensor_service_name_);
    common_msgs::CloudData sensor_cloud = sensor_srv.response.cloud_data;
    sensor_msgs::Image image = sensor_srv.response.image;

    common_srvs::ObjectDetectionService ob_detect_2d_srv;
    ob_detect_2d_srv.request.input_image = image;
    Util::client_request(object_detect_client_, ob_detect_2d_srv, object_detect_service_name_);
    std::vector<common_msgs::BoxPosition> box_pos = ob_detect_2d_srv.response.b_boxs;
    std::vector<float> cinfo_list = UtilMsgData::caminfo_to_floatlist(sensor_srv.response.camera_info);
    cv::Mat img = UtilMsgData::rosimg_to_cvimg(image, sensor_msgs::image_encodings::BGR8);
    Get3DBy2D get3d(cinfo_list, Util::get_image_size(img));
    std::vector<common_msgs::CloudData> cloud_multi = get3d.get_out_data(sensor_cloud, box_pos);
    common_srvs::SemanticSegmentationService semantic_srv;
    semantic_srv.request.input_data_multi = cloud_multi;
    Util::client_request(cloud_network_client_, semantic_srv, cloud_network_service_name_);

    common_srvs::VisualizeCloud visualize_srv;
    common_msgs::CloudData final_cloud;
    for (int i = 0; i < semantic_srv.response.output_data_multi.size(); i++) {
        final_cloud = UtilMsgData::concat_cloudmsg(final_cloud, semantic_srv.response.output_data_multi[i]);
    }
    visualize_srv.request.cloud_data_list.push_back(final_cloud);
    visualize_srv.request.topic_name_list.push_back("final_cloud");
    // visualize_srv.request.cloud_data_list = semantic_srv.response.output_data_multi;
    // for (int i = 0; i < semantic_srv.response.output_data_multi.size(); i++) {
    //     visualize_srv.request.topic_name_list.push_back("cloud_multi_" + std::to_string(i));
    // }
    Util::client_request(visualize_client_, visualize_srv, visualize_service_name_);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "estimation_client");
    ros::NodeHandle nh;
    EstimationClient estimation_client(nh);
    for (int i = 0; i < estimation_client.the_number_of_execute_; i++) {
        estimation_client.main();
        ros::Duration(0.1);
    }
}