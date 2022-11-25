#include <accuracy_package/accuracy_server.hpp>

AccuracyServer::AccuracyServer(ros::NodeHandle &nh) :
    nh_(nh),
    pnh_("~")
{
    set_parameter();
    server_ = nh_.advertiseService(acc_service_name_, &AccuracyServer::accuracy_service_callback, this);
}

void AccuracyServer::set_parameter()
{
    pnh_.getParam("accuracy_iou_service_name", acc_service_name_);
}

bool AccuracyServer::accuracy_service_callback(common_srvs::AccuracyIouService::Request &request, common_srvs::AccuracyIouService::Response &response)
{
    common_msgs::CloudData gt_cloud = request.ground_truth_cloud;
    common_msgs::CloudData esti_cloud = request.estimation_cloud;
    esti_cloud = UtilMsgData::extract_ins_cloudmsg(esti_cloud, request.instance);
    Util::message_show("esti_cloud_size", esti_cloud.x.size());
    common_msgs::CloudData gt_parts;
    // Util::message_show("", request.ground_truth_cloud.cloud_name);
    gt_parts = AccuracyUtil::extract_gt_parts(gt_cloud, esti_cloud);
    int gt_ins_quantity = AccuracyUtil::get_gt_the_instance_quantity(gt_cloud, request.instance);
    // Util::message_show("gt_quantity", gt_ins_quantity);
    response.iou_result = AccuracyUtil::calcurate_iou(gt_parts, gt_ins_quantity, request.instance);
    // Util::message_show("iou_show", response.iou_result);
    return true;
}