#!/usr/bin/python3
from network.object_detection.yolov3 import test as yolo_run
from network.object_detection.ssd.demo import SSDEstimation
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
        self.ssd_config_path = param_list["ssd"]["config_path"]
        self.ssd_checkpoints_file = param_list["ssd"]["checkpoints_file_path"]
        self.ssd_score_threshold = param_list["ssd"]["threshold"]
        self.object_detect_mode = param_list["object_detect_mode"]
        self.semantic_class_num = param_list["semantic_pointnet"]["class_num"]
        self.semantic_checkpoints = param_list["semantic_pointnet"]["checkpoints"]
    
    def object_detection_initialize(self):
        self.device = yolo_run.get_device()
        self.ssd_network = SSDEstimation()
        self.ssd_network.setting_network(self.ssd_config_path, self.ssd_checkpoints_file, self.ssd_score_threshold, self.device)

    def semantic_initialize(self):
        # self.yolo_config_object = yolo_run.load_config(self.yolo_config_path)
        # self.yolo_class_list = yolo_run.get_class_names(self.yolo_config_path)
        # self.yolo_net = yolo_run.create_model(self.yolo_config_object, self.device)
        # self.yolo_net = yolo_run.load_checkpoints(self.yolo_net, self.yolo_checkpoints, self.device)
        self.semantic_net = semantic_run.create_model(self.semantic_class_num, self.device)
        self.semantic_net = semantic_run.load_checkpoints(self.semantic_net, self.semantic_checkpoints, self.device)
        
    def object_detect_callback(self, request):
        if self.ssd_checkpoints_file != request.checkpoints_path and request.checkpoints_path is not None:
            self.ssd_checkpoints_file = request.checkpoints_path
            self.object_detection_initialize()
        img = util_msg_data.rosimg_to_npimg(request.input_image)
        response = ObjectDetectionServiceResponse()
        if self.object_detect_mode == "yolo":
            detection = yolo_run.object_detection(img, self.yolo_net, self.yolo_config_object, self.device, self.yolo_class_list)
            boxes_pos = yolo_run.get_box_info(detection, img.shape[0], img.shape[1])
            response.b_boxs = boxes_pos
        elif self.object_detect_mode == "ssd":
            boxes, _, _ = self.ssd_network.object_detection(img, self.ssd_score_threshold)
            response.b_boxs = SSDEstimation.get_box_position(boxes)
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
        if self.semantic_checkpoints != request.checkpoints_path and request.checkpoints_path is not None:
            self.semantic_checkpoints = request.checkpoints_path
            self.semantic_initialize()
        out_list = []
        for i in range(len(request.input_data_multi)):
            np_input = util_msg_data.msgcloud_to_npcloud(request.input_data_multi[i])
            np_input, _ = util_msg_data.extract_mask_from_npcloud(np_input)
            np_input, offset = network_util.get_normalizedcloud(np_input)
            # print(offset.shape)
            outdata = semantic_run.semantic_segmentation(self.semantic_net, np_input, self.device)
            outcloud = util_msg_data.npcloud_to_msgcloud(outdata)
            outcloud.x += offset[0][0]
            outcloud.y += offset[0][1]
            outcloud.z += offset[0][2]
            out_list.append(outcloud)
        response = SemanticSegmentationServiceResponse()
        response.output_data_multi = out_list
        return response
    

if __name__=='__main__':
    rospy.init_node('network_server')
    network = NetworkServer()
    rospy.spin()
        