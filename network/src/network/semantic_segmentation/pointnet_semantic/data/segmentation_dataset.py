import torch.utils.data
import numpy as np
from network.network_common import network_util
import torch
from tqdm import tqdm
from hdf5_package import hdf5_function


def make_segmentation_data(hdf5_object, start_index):
    x_data = []
    for i in tqdm(range(hdf5_function.get_len_hdf5(hdf5_object))):
        pcl_data = hdf5_object["data_" + str(start_index + i)]['Points'][()]
        mask_data = hdf5_object["data_" + str(start_index + i)]['masks'][()]
        # print(pcl_data.shape)
        # print(mask_data.shape)
        if pcl_data.shape[0] > 0 and pcl_data.shape[0] == mask_data.shape[0]:
            concat_data = np.hstack([pcl_data, mask_data])
            x_data.append(concat_data)
    return x_data

def get_input_data_from_hdf5(hdf5_object, index):
    pcl_data = hdf5_object["data_" + str(index)]['Points'][()]
    return pcl_data

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, start_index):
        super(SegmentationDataset, self).__init__()
        self.hdf5_object = hdf5_function.open_readed_hdf5(data_path)
        self.x_data = make_segmentation_data(self.hdf5_object, start_index)
        
    def __getitem__(self, index):
        pcd_data = self.x_data[index]
        input_data, _ = network_util.get_normalizedcloud_segmentation(pcd_data)
        x_data = input_data[:,:3]
        y_data = input_data[:,3]
        meta = {}
        meta["x_data"] = x_data
        meta["y_data"] = y_data
        return meta

    def __len__(self):
        return len(self.x_data)