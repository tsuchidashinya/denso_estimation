#include <accuracy_package/accuracy_util.hpp>

common_msgs::CloudData AccuracyUtil::extract_gt_parts(common_msgs::CloudData gt_cloud, common_msgs::CloudData esti_cloud)
{
    common_msgs::CloudData gt_parts;
    for (int i = 0; i < esti_cloud.x.size(); i++) {
        for (int j = 0; j < gt_cloud.x.size(); j++) {
            if (esti_cloud.x[i] == gt_cloud.x[j] && esti_cloud.y[i] == gt_cloud.y[j] && esti_cloud.z[i] == gt_cloud.z[j]) {
                gt_parts.x.push_back(gt_cloud.x[j]);
                gt_parts.y.push_back(gt_cloud.y[j]);
                gt_parts.z.push_back(gt_cloud.z[j]);
                gt_parts.instance.push_back(gt_cloud.instance[j]);
                break;
            }
        }
    }
    return gt_parts;
}

int AccuracyUtil::get_gt_the_instance_quantity(common_msgs::CloudData gt_cloud, int instance)
{
    int gt_instance_quantity = 0;
    for (int i = 0; i < gt_cloud.x.size(); i++) {
        if (int(gt_cloud.instance[i]) == instance) {
            gt_instance_quantity++;
        }
    }
    return gt_instance_quantity;
}

/*
1: gt_parts
2: gt_ins_quantity
3: instance
*/
float AccuracyUtil::calcurate_iou(common_msgs::CloudData gt_parts, int gt_ins_quantity, int instance)
{
    float tp = 0, fp = 0, fn = 0;
    float iou;
    for (int i = 0; i < gt_parts.x.size(); i++) {
        if (int(gt_parts.instance[i]) == instance) {
            tp = tp + 1;
        }
        else {
            fp = fp + 1;
        }
    }
    fn = gt_ins_quantity - tp;
    // Util::message_show("tp", tp);
    // Util::message_show("fp", fp);
    // Util::message_show("fn", fn);
    // Util::message_show("gt_parts", gt_parts.x.size());
    iou = tp / (tp + fp + fn);
    return iou;
}