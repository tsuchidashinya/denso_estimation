#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
import torch
import argparse
import models.POINTNET as POINTNET
import data.util as util
from util import hdf5_function

def get_net_output(model, input_data, device):
    x_data = torch.from_numpy(input_data)
    x_data = x_data.float()
    x_data = x_data.transpose(2, 1)
    x_data = x_data.to(device)
    pred = model(x_data)
    pred = pred.to('cpu').detach().numpy().copy()
    return pred

def load_checkpoints(net, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, device)
    net.load_state_dict(state_dict, strict=False)
    return net

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(device):
    net = POINTNET.PointNetPose(3, 9)
    return net.to(device)

    
def pose_estimation(net, input_data, device):
    input_data = input_data[np.newaxis, :, :]
    y_pre = get_net_output(net, input_data, device)
    y_pos = y_pre[0][0:3]
    rot = Rotation.from_matrix(np.array(y_pre[0][3:12]).reshape(3, 3))
    y_euler = rot.as_quat()
    y = np.r_[y_pos, y_euler]
    est_pose = PoseStamped()
    est_pose.pose.position.x = y[0]
    est_pose.pose.position.y = y[1]
    est_pose.pose.position.z = y[2]
    est_pose.pose.orientation.x = y[3]
    est_pose.pose.orientation.y = y[4]
    est_pose.pose.orientation.z = y[5]
    est_pose.pose.orientation.w = y[6]
    return est_pose

if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_path', help='Dataset root directory path')
    parser.add_argument('--checkpoints', default='weights/', help='Directory for saving checkpoint models')
    args = parser.parse_args()
    hdf5_object = hdf5_function.open_readed_hdf5(args.dataset_path)
    input_data = util.get_input_data_from_hdf5(hdf5_object, 3)
    input_data = np.array(input_data)
    device = get_device()
    net = create_model(device)
    net = load_checkpoints(net, args.checkpoints, device)
    input_data, _ = util.getNormalizedPcd_nodown(input_data)
    est_pose = pose_estimation(net, input_data, device)
    print(est_pose)


