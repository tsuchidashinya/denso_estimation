#!/usr/bin/python3
import rospy
from common_srvs.srv import AccuracyIouService, AccuracyIouServiceResponse
from util import util_msg_data
from accuracy_package import accuracy_util

class AccuracyServer:
    def __init__(self):
        self.set_parameter()
        rospy.Service(self.accuracy_iou_service_name, AccuracyIouService, self.accuracy_iou_service_callback)
    
    def set_parameter(self):
        self.accuracy_iou_service_name = rospy.get_param("~accuracy_iou_service_name", "accuracy_iou_service")
    
    def accuracy_iou_service_callback(self, request):
        gt_cloud = request.ground_truth_cloud
        esti_cloud = util_msg_data.extract_ins_cloud_msg(request.estimation_cloud, request.instance)
        gt_part_cloud = accuracy_util.extract_ground_truth_parts(gt_cloud, esti_cloud)
        gt_ins_num = accuracy_util.get_gt_instance_num(gt_cloud, request.instance)
        response = AccuracyIouServiceResponse()
        response.iou_result = accuracy_util.calcurate_iou(gt_part_cloud, gt_ins_num, request.instance)
        return response

if __name__=='__main__':
    rospy.init_node('accuracy_server')
    acc_server = AccuracyServer()
    rospy.spin()
        