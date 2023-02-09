#include <estimation_client_pkg/estimation_client.hpp>

EstimationClient::EstimationClient(ros::NodeHandle &nh) :
nh_(nh),
pnh_("~"),
counter_(0),
estimation_name_("esti")
{
    set_paramenter();
    object_detect_client_ = nh_.serviceClient<common_srvs::ObjectDetectionService>(object_detect_service_name_);
    cloud_network_client_ = nh_.serviceClient<common_srvs::SemanticSegmentationService>(cloud_network_service_name_);
    visualize_client_ = nh_.serviceClient<common_srvs::VisualizeCloud>(visualize_service_name_);
    vis_image_client_ = nh_.serviceClient<common_srvs::VisualizeImage>(vis_image_service_name_);
    accuracy_client_ = nh_.serviceClient<common_srvs::AccuracyIouService>(accuracy_service_name_);
    hdf5_client_ = nh_.serviceClient<common_srvs::Hdf5OpenSensorDataService>(hdf5_service_name_);
}

void EstimationClient::set_paramenter()
{
    pnh_.getParam("common_parameter", param_list);
    world_frame_ = static_cast<std::string>(param_list["world_frame"]);
    sensor_frame_ = static_cast<std::string>(param_list["sensor_frame"]);
    pnh_.getParam("acc_estimation_main", param_list);
    visualize_service_name_ = "visualize_cloud_service";
    vis_image_service_name_ = "visualize_image_service";
    object_detect_service_name_ = "object_detect_service";
    cloud_network_service_name_ = "network_semantic_service";
    accuracy_service_name_ = "accuracy_service";
    hdf5_service_name_ = "hdf5_open_sensor_data_service";
    hdf5_open_file_path_ = static_cast<std::string>(param_list["hdf5_open_file_path"]);
    object_detect_mode_ = static_cast<std::string>(param_list["object_detect_mode"]);
    object_detect_checkpoint_path_ = static_cast<std::string>(param_list["object_detect_checkpoint_path"]);
    semantic_class_num_ = static_cast<int>(param_list["semantic_class_num"]);
    semantic_checkpoint_path_ = static_cast<std::string>(param_list["semantic_checkpoint_path"]);
    estimation_name_ = static_cast<std::string>(param_list["estimation_name"]);
}

void EstimationClient::main()
{
    while (1) {
        common_srvs::Hdf5OpenSensorDataService hdf5_srv;
        hdf5_srv.request.index = counter_;
        hdf5_srv.request.hdf5_open_file_path = hdf5_open_file_path_;
        hdf5_srv.request.is_reload = 1;
        Util::client_request(hdf5_client_, hdf5_srv, hdf5_service_name_);
        sensor_msgs::Image image = hdf5_srv.response.image;
        common_msgs::CloudData cloud_data = hdf5_srv.response.cloud_data;
        
        common_srvs::ObjectDetectionService ob_detect_2d_srv;
        if (counter_ == 0) 
            ob_detect_2d_srv.request.reload = 1;
        else 
            ob_detect_2d_srv.request.reload = 0;
        ob_detect_2d_srv.request.model_mode = "ssd";
        ob_detect_2d_srv.request.input_image = hdf5_srv.response.image;
        ob_detect_2d_srv.request.checkpoints_path = object_detect_checkpoint_path_;
        Util::client_request(object_detect_client_, ob_detect_2d_srv, object_detect_service_name_);
        std::vector<common_msgs::BoxPosition> box_pos = ob_detect_2d_srv.response.b_boxs;
        // Util::message_show("box", box_pos.size());
        // Util::message_show("camera_info", hdf5_srv.response.camera_info.size());
        std::vector<float> cinfo_list = hdf5_srv.response.camera_info;
        // Util::message_show("58", "ok");
        cv::Mat img = UtilMsgData::rosimg_to_cvimg(image, sensor_msgs::image_encodings::BGR8);
        // Util::message_show("60", "ok");
        Data2Dto3D get3d(cinfo_list, Util::get_image_size(img));
        // Util::message_show("63", "ok");
        std::vector<common_msgs::CloudData> cloud_multi = get3d.get_out_data(cloud_data, box_pos);
        // Util::message_show("cloud_multi", cloud_multi.size());
        // Util::message_show("cloud_multi_1", cloud_multi[1].x.size());
        common_srvs::SemanticSegmentationService semantic_srv;
        if (counter_ == 0) 
            semantic_srv.request.reload = 1;
        else
            semantic_srv.request.reload = 0;
        semantic_srv.request.semantic_class_num = semantic_class_num_;
        semantic_srv.request.input_data_multi = cloud_multi;
        semantic_srv.request.checkpoints_path = semantic_checkpoint_path_;
        Util::client_request(cloud_network_client_, semantic_srv, cloud_network_service_name_);
        common_srvs::VisualizeCloud visualize_srv;
        cv::Mat draw_img = Data3Dto2D::draw_b_box(img, box_pos);
        sensor_msgs::Image out_img = UtilMsgData::cvimg_to_rosimg(draw_img, "bgr8");
        common_srvs::VisualizeImage vis_img_srv;
        vis_img_srv.request.image_list.push_back(out_img);
        vis_img_srv.request.topic_name_list.push_back(estimation_name_ + "_" + "hdf5_image_" + std::to_string(counter_));
        Util::client_request(vis_image_client_, vis_img_srv, vis_image_service_name_);
        common_msgs::CloudData final_cloud, all_color_cloud, all_sensor_color_cloud;
        for (int i = 0; i < semantic_srv.response.output_data_multi.size(); i++) {
            final_cloud = UtilMsgData::concat_cloudmsg(final_cloud, semantic_srv.response.output_data_multi[i]);
            visualize_srv.request.cloud_data_list.push_back(semantic_srv.response.output_data_multi[i]);
            visualize_srv.request.topic_name_list.push_back(estimation_name_ + "_" + std::to_string(counter_) + "_" + std::to_string(i));
            auto extract_cloud = UtilMsgData::extract_ins_cloudmsg(semantic_srv.response.output_data_multi[i], 1);
            extract_cloud = UtilMsgData::change_ins_cloudmsg(extract_cloud, 1, i + 1);
            all_color_cloud = UtilMsgData::concat_cloudmsg(all_color_cloud, extract_cloud);
            
        }
        auto sensor_cloud = UtilMsgData::draw_all_ins_cloudmsg(hdf5_srv.response.cloud_data, 0);
        all_sensor_color_cloud = UtilMsgData::concat_cloudmsg(sensor_cloud, all_color_cloud);
        CloudProcess cloud_process;
        cloud_process.set_crop_frame(sensor_frame_, world_frame_);
        all_sensor_color_cloud = cloud_process.cropbox_segmenter(all_sensor_color_cloud);
        visualize_srv.request.cloud_data_list.push_back(all_sensor_color_cloud);
        visualize_srv.request.topic_name_list.push_back("all_sensor_color_cloud" + std::to_string(counter_));
        visualize_srv.request.cloud_data_list.push_back(final_cloud);
        visualize_srv.request.topic_name_list.push_back(estimation_name_ + "_" + "final_cloud_" + std::to_string(counter_));
        Util::client_request(visualize_client_, visualize_srv, visualize_service_name_);
        ROS_INFO_STREAM("estimation " << counter_);
        if (counter_ >= hdf5_srv.response.data_size - 1) {
            ROS_INFO_STREAM("finish!!");
            break;
        }
        counter_++;
    }
    
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "estimation_client");
    ros::NodeHandle nh;
    EstimationClient estimation_client(nh);
    estimation_client.main();
    return 0;
}