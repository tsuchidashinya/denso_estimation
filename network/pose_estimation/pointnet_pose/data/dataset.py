import torch.utils.data

import numpy as np
import data.data_util as data_util
import torch
from util import hdf5_function


class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super(PoseDataset, self).__init__()
        self.hdf5_object = hdf5_function.open_readed_hdf5(data_path)
        self.x_data, self.y_data = data_util.make_pose_data(self.hdf5_object)
        
    def __getitem__(self, index):
        pcd_data = self.x_data[index]
        x_data, pcd_offset = data_util.getNormalizedPcd_nodown(pcd_data)
        y_data = self.y_data[index]
        y_pos = y_data[0:3] - pcd_offset[0]
        y_rot = y_data[3:]
        y_data = np.concatenate([y_pos, y_rot])
        meta = {}
        print("x_data", x_data.shape)
        meta["x_data"] = x_data
        meta["y_data"] = y_data
        return meta

    def __len__(self):
        return self.len_size