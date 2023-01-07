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
    hdf5_client_ = nh_.serviceClient<common_srvs::Hdf5OpenAccService>(hdf5_service_name_);
}

void EstimationClient::set_paramenter()
{
    pnh_.getParam("acc_estimation_main", param_list);
    visualize_service_name_ = static_cast<std::string>(param_list["visualize_service_name"]);
    vis_image_service_name_ = static_cast<std::string>(param_list["visualize_image_service_name"]);
    object_detect_service_name_ = static_cast<std::string>(param_list["object_detect_service_name"]);
    cloud_network_service_name_ = static_cast<std::string>(param_list["cloud_network_service_name"]);
    accuracy_service_name_ = static_cast<std::string>(param_list["accuracy_service_name"]);
    hdf5_service_name_ = static_cast<std::string>(param_list["hdf5_service_name"]);
    hdf5_open_file_path_ = static_cast<std::string>(param_list["hdf5_open_file_path"]);
    ssd_checkpoint_path_ = static_cast<std::string>(param_list["ssd_checkpoint_path"]);
    semantic_checkpoint_path_ = static_cast<std::string>(param_list["semantic_checkpoint_path"]);
    estimation_name_ = static_cast<std::string>(param_list["estimation_name"]);
}

void EstimationClient::main()
{
    while (1) {
        common_srvs::Hdf5OpenAccService hdf5_srv;
        hdf5_srv.request.index = counter_;
        hdf5_srv.request.hdf5_open_file_path = hdf5_open_file_path_;
        Util::client_request(hdf5_client_, hdf5_srv, hdf5_service_name_);
        if (counter_ >= hdf5_srv.response.data_size - 1) {
            break;
        }
        sensor_msgs::Image image = hdf5_srv.response.image;
        common_msgs::CloudData cloud_data = hdf5_srv.response.cloud_data;
        for (int i = 2; i < 6; i++) {
            if (counter_ == 1 && (i == 3 || i == 5 || i == 4)) {
                cloud_data = UtilMsgData::change_ins_cloudmsg(cloud_data, i, 0);
            }
            else {
                cloud_data = UtilMsgData::change_ins_cloudmsg(cloud_data, i, 1);
            }
            if (counter_ > 1 && i > 2) {
                cloud_data = UtilMsgData::change_ins_cloudmsg(cloud_data, i, 1);
            }
        }
        common_srvs::ObjectDetectionService ob_detect_2d_srv;
        ob_detect_2d_srv.request.input_image = hdf5_srv.response.image;
        ob_detect_2d_srv.request.checkpoints_path = ssd_checkpoint_path_;
        Util::client_request(object_detect_client_, ob_detect_2d_srv, object_detect_service_name_);
        std::vector<common_msgs::BoxPosition> box_pos = ob_detect_2d_srv.response.b_boxs;
        std::vector<float> cinfo_list = hdf5_srv.response.camera_info;
        cv::Mat img = UtilMsgData::rosimg_to_cvimg(image, sensor_msgs::image_encodings::BGR8);
        Data2Dto3D get3d(cinfo_list, Util::get_image_size(img));
        std::vector<common_msgs::CloudData> cloud_multi = get3d.get_out_data(cloud_data, box_pos);
        common_srvs::SemanticSegmentationService semantic_srv;
        semantic_srv.request.input_data_multi = cloud_multi;
        semantic_srv.request.checkpoints_path = semantic_checkpoint_path_;
        Util::client_request(cloud_network_client_, semantic_srv, cloud_network_service_name_);

        common_srvs::VisualizeCloud visualize_srv;
        // visualize_srv.request.cloud_data_list.push_back(hdf5_srv.response.cloud_data);
        // visualize_srv.request.topic_name_list.push_back("hdf5_package");
        
        cv::Mat draw_img = Data3Dto2D::draw_b_box(img, box_pos);
        sensor_msgs::Image out_img = UtilMsgData::cvimg_to_rosimg(draw_img, "bgr8");
        common_srvs::VisualizeImage vis_img_srv;
        vis_img_srv.request.image_list.push_back(out_img);
        vis_img_srv.request.topic_name_list.push_back(estimation_name_ + "_" + "hdf5_image_" + std::to_string(counter_));
        Util::client_request(vis_image_client_, vis_img_srv, vis_image_service_name_);
        // visualize_srv.request.cloud_data_list = semantic_srv.response.output_data_multi;
        // for (int i = 0; i < semantic_srv.response.output_data_multi.size(); i++) {
        //     visualize_srv.request.topic_name_list.push_back("cloud_multi_" + std::to_string(i));
        // }
        common_msgs::CloudData final_cloud, final_color_cloud;
        for (int i = 0; i < semantic_srv.response.output_data_multi.size(); i++) {
            visualize_srv.request.cloud_data_list.push_back(semantic_srv.response.output_data_multi[i]);
            visualize_srv.request.topic_name_list.push_back(estimation_name_ + "_" + "cloud_multi_" + std::to_string(counter_) + "_" + std::to_string(i));
            final_cloud = UtilMsgData::concat_cloudmsg(final_cloud, semantic_srv.response.output_data_multi[i]);
        }
        visualize_srv.request.cloud_data_list.push_back(final_cloud);
        visualize_srv.request.topic_name_list.push_back(estimation_name_ + "_" + "final_cloud_" + std::to_string(counter_));
        final_color_cloud = UtilMsgData::extract_ins_cloudmsg(final_cloud, 1);
        visualize_srv.request.cloud_data_list.push_back(final_color_cloud);
        visualize_srv.request.topic_name_list.push_back(estimation_name_ + "_" + "final_cloud_color_" + std::to_string(counter_));
        visualize_srv.request.cloud_data_list.push_back(hdf5_srv.response.cloud_data);
        visualize_srv.request.topic_name_list.push_back(estimation_name_ + "_" + "sensor_data_" + std::to_string(counter_));
        Util::client_request(visualize_client_, visualize_srv, visualize_service_name_);

        common_srvs::AccuracyIouService accuracy_srv;
        accuracy_srv.request.estimation_cloud = final_cloud;
        accuracy_srv.request.instance = 1;
        accuracy_srv.request.ground_truth_cloud = cloud_data;
        accuracy_srv.request.ground_truth_cloud.tf_name = "acc" + std::to_string(counter_);
        ros::WallTime start = ros::WallTime::now();
        Util::client_request(accuracy_client_, accuracy_srv, accuracy_service_name_);
        ros::WallTime end = ros::WallTime::now();
        std::string time_str = std::to_string((end - start).toSec());
        Util::message_show("time: " + time_str + "  iou", accuracy_srv.response.iou_result);
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