#pragma once
#include "accuracy_package/accuracy_util.hpp"
#include <common_srvs/AccuracyIouService.h>

class AccuracyServer
{
public:
    AccuracyServer(ros::NodeHandle &);
    XmlRpc::XmlRpcValue param_list;
private:
    bool accuracy_service_callback(common_srvs::AccuracyIouService::Request&, common_srvs::AccuracyIouService::Response&);
    void set_parameter();

    ros::NodeHandle nh_, pnh_;
    ros::ServiceServer server_;
    std::string acc_service_name_;
};