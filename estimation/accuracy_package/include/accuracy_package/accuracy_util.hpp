#pragma once
#include <util/util.hpp>
#include <util/util_msg_data.hpp>

class AccuracyUtil
{
public:
    static common_msgs::CloudData extract_gt_parts(common_msgs::CloudData, common_msgs::CloudData);
    static int get_gt_the_instance_quantity(common_msgs::CloudData, int);
    static float calcurate_iou(common_msgs::CloudData, int, int);
};