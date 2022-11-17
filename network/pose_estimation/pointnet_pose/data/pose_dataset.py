import torch.utils.data
import numpy as np
from network_common import network_util
import torch
from tqdm import tqdm
from util import hdf5_function, util_python


def make_pose_data(hdf5_object, start_index):
    x_data = []
    y_data = []
    for i in tqdm(range(hdf5_function.get_len_hdf5(hdf5_object))):
        pcl_data = hdf5_object["data_" + str(start_index + i)]['pcl'][()]
        pose_data = hdf5_object["data_" + str(start_index + i)]['pose'][()]
        pose_data = util_python.conv_quat2mat(pose_data)
        x_data.append(pcl_data)
        y_data.append(pose_data)
    return x_data, y_data

def get_input_data_from_hdf5(hdf5_object, index):
    pcl_data = hdf5_object["data_" + str(index)]['pcl'][()]
    return pcl_data


class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, start_index):
        super(PoseDataset, self).__init__()
        self.hdf5_object = hdf5_function.open_readed_hdf5(data_path)
        self.x_data, self.y_data = make_pose_data(self.hdf5_object, start_index)
        
    def __getitem__(self, index):
        pcd_data = self.x_data[index]
        x_data, pcd_offset = network_util.get_normalizedcloud(pcd_data)
        y_data = self.y_data[index]
        y_pos = y_data[0:3] - pcd_offset[0]
        y_rot = y_data[3:]
        y_data = np.concatenate([y_pos, y_rot])
        meta = {}
        meta["x_data"] = x_data
        meta["y_data"] = y_data
        return meta

    def __len__(self):
        return len(self.x_data)