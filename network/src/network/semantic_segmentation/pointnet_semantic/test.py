#!/usr/bin/env python3
import numpy as np
import torch
import argparse
from network.network_common import network_util
from network.semantic_segmentation.pointnet_semantic.data import segmentation_dataset
from hdf5_package import hdf5_function
from network.semantic_segmentation.pointnet_semantic.model import POINTNET_SEMANTIC
from util import util_msg_data


def get_net_output(net, input_data, device):
    x_data = torch.from_numpy(input_data)
    x_data = x_data.float()
    x_data = x_data.transpose(2, 1)
    x_data = x_data.to(device)
    pred, _ = net(x_data)
    pred = pred.contiguous().cpu().data.max(2)[1].numpy()
    return pred

def create_model(class_num, device):
    net = POINTNET_SEMANTIC.PointNetSemantic(class_num)
    net = net.to(device)
    return net

def load_checkpoints(net, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, device)
    net.load_state_dict(state_dict, strict=False)
    return net

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def semantic_segmentation(net, input_data ,device):
    x = input_data[np.newaxis, :, :]
    y_pre = get_net_output(net, x, device)
    y = y_pre.T
    outdata = np.hstack([input_data, y])
    return outdata

if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_path', help='Dataset root directory path')
    parser.add_argument('--checkpoints', default='weights/', help='Directory for saving checkpoint models')
    parser.add_argument('--num_instance_classes', default=2, type=int)
    args = parser.parse_args()
    hdf5_object = hdf5_function.open_readed_hdf5(args.dataset_path)
    input_data = segmentation_dataset.get_input_data_from_hdf5(hdf5_object, 3)
    input_data = np.array(input_data)
    print(input_data.shape)
    device = get_device()
    net = create_model(args.num_instance_classes, device)
    net = load_checkpoints(net, args.checkpoints, device)
    input_data, _ = network_util.get_normalizedcloud(input_data)
    out_data = semantic_segmentation(net, input_data, device)
    # input_data = util_msg_data.npcloud_to_msgcloud(input_data)
    # print(util_msg_data.get_instance_dict(input_data))
    out_data_cloud = util_msg_data.npcloud_to_msgcloud(out_data)
    print(util_msg_data.get_instance_dict(out_data_cloud))
    # cloud_data = util_msg_data.npcloud_to_msgcloud(out_data)
    # extract_data = util_msg_data.extract_ins_cloud_msg(cloud_data, 0)
    # extract_np_data = util_msg_data.msgcloud_to_npcloud(extract_data)
    # print(np.array(extract_np_data).shape)
    
