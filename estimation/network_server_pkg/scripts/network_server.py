#!/usr/bin/python3
from network.object_detection.yolov3 import test as yolo_run
from network.semantic_segmentation.pointnet_semantic import test as semantic_run
import rosparam
import rospy
from common_srvs.srv import NetworkCloudService, NetworkCloudServiceResponse, NetworkCloudServiceRequest
from common_srvs.srv import SemanticSegmentationService, SemanticSegmentationServiceResponse, SemanticSegmentationServiceRequest
from common_srvs.srv import ObjectDetectionService, ObjectDetectionServiceResponse, ObjectDetectionServiceRequest
from util import util_msg_data
from network.network_common import network_util

class NetworkServer:
    def __init__(self):
        self.set_parameter()
        self.network_initialize()
        rospy.Service(self.object_detect_service_name, ObjectDetectionService, self.object_detect_callback)
        rospy.Service(self.network_cloud_service_name, NetworkCloudService, self.network_cloud_callback)
        rospy.Service(self.network_semantic_service_name, SemanticSegmentationService, self.network_semantic_callback)
    
    def set_parameter(self):
        param_list = rosparam.get_param(rospy.get_name() + "/network_server")
        self.object_detect_service_name = param_list["object_detect_service_name"]
        self.network_cloud_service_name = param_list["network_cloud_service_name"]
        self.network_semantic_service_name = param_list["network_semantic_service_name"]
        self.yolo_checkpoints = param_list["yolov3"]["checkpoints"]
        self.yolo_config_path = param_list["yolov3"]["config_path"]
        self.semantic_class_num = param_list["semantic_pointnet"]["class_num"]
        self.semantic_checkpoints = param_list["semantic_pointnet"]["checkpoints"]
    
    def network_initialize(self):
        self.device = yolo_run.get_device()
        self.yolo_config_object = yolo_run.load_config(self.yolo_config_path)
        self.yolo_class_list = yolo_run.get_class_names(self.yolo_config_path)
        self.yolo_net = yolo_run.create_model(self.yolo_config_object, self.device)
        self.yolo_net = yolo_run.load_checkpoints(self.yolo_net, self.yolo_checkpoints, self.device)
        self.semantic_net = semantic_run.create_model(self.semantic_class_num, self.device)
        self.semantic_net = semantic_run.load_checkpoints(self.semantic_net, self.semantic_checkpoints, self.device)
        
    def object_detect_callback(self, request):
        img = util_msg_data.rosimg_to_npimg(request.input_image)
        detection = yolo_run.object_detection(img, self.yolo_net, self.yolo_config_object, self.device, self.yolo_class_list)
        boxes_pos = yolo_run.get_box_info(detection, img.shape[0], img.shape[1])
        response = ObjectDetectionServiceResponse()
        response.b_boxs = boxes_pos
        return response
    
    def network_cloud_callback(self, request):
        out_list = []
        for i in range(len(request.cloud_data_multi)):
            outdata = semantic_run.semantic_segmentation(self.semantic_net, request.cloud_data_multi[i], self.device)
            out_list.append(outdata)
        response = NetworkCloudServiceResponse()
        return response
    
    def network_semantic_callback(self, request):
        # request = SemanticSegmentationServiceRequest()
        out_list = []
        for i in range(len(request.input_data_multi)):
            np_input = util_msg_data.msgcloud_to_npcloud(request.input_data_multi[i])
            np_input, _ = util_msg_data.extract_mask_from_npcloud(np_input)
            np_input, _ = network_util.get_normalizedcloud(np_input)
            outdata = semantic_run.semantic_segmentation(self.semantic_net, np_input, self.device)
            outcloud = util_msg_data.npcloud_to_msgcloud(outdata)
            out_list.append(outcloud)
        response = SemanticSegmentationServiceResponse()
        response.output_data_multi = out_list
        return response
    

if __name__=='__main__':
    rospy.init_node('network_server')
    network = NetworkServer()
    rospy.spin()
        