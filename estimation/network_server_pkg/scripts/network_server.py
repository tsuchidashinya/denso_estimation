#!/usr/bin/python3
from network.object_detection.yolov3 import test as yolo_run
import rosparam
import rospy
from common_srvs.srv import NetworkCloudService, NetworkCloudServiceResponse, NetworkCloudServiceRequest
from common_srvs.srv import ObjectDetectionService, ObjectDetectionServiceResponse, ObjectDetectionServiceRequest
from util import util_msg_data

class NetworkServer:
    def __init__(self):
        self.set_parameter()
        rospy.Service(self.object_detect_service_name, ObjectDetectionService, self.object_detect_callback)
        rospy.Service(self.network_cloud_service_name, NetworkCloudService, self.network_cloud_callback)
    
    def set_parameter(self):
        param_list = rosparam.get_param(rospy.get_name() + "/network_server")
        self.object_detect_service_name = param_list["object_detect_service_name"]
        self.network_cloud_service_name = param_list["network_cloud_service_name"]
        self.yolo_checkpoints = param_list["yolov3"]["checkpoints"]
        self.yolo_config_path = param_list["yolov3"]["config_path"]
    
    def network_initialize(self):
        self.device = yolo_run.get_device()
        self.yolo_config_object = yolo_run.load_config(self.yolo_config_path)
        self.yolo_class_list = yolo_run.get_class_names(self.yolo_config_path)
        self.yolo_net = yolo_run.create_model(self.yolo_config_object, self.device)
        
    def object_detect_callback(self, request):
        # request = ObjectDetectionServiceRequest()
        img = util_msg_data.rosimg_to_npimg(request.input_image)
        detections = yolo_run.object_detection(img, self.yolo_net, self.yolo_config_object)
    
    def network_cloud_callback(self, request):
        request = NetworkCloudServiceRequest()
        
        
        