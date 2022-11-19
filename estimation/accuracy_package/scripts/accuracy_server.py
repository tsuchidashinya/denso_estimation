#!/usr/bin/python3
from network.object_detection.yolov3 import test as yolo_run
from network.semantic_segmentation.pointnet_semantic import test as semantic_run
import rosparam
import rospy
from common_srvs.srv import NetworkCloudService, NetworkCloudServiceResponse, NetworkCloudServiceRequest
from common_srvs.srv import ObjectDetectionService, ObjectDetectionServiceResponse, ObjectDetectionServiceRequest
from util import util_msg_data

class AccuracyServer:
    def __init__(self):
        self.set_parameter()
        rospy.Service(self.object_detect_service_name, ObjectDetectionService, self.object_detect_callback)
        rospy.Service(self.network_cloud_service_name, NetworkCloudService, self.network_cloud_callback)