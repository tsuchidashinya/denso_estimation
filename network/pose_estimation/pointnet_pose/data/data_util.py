from tqdm import tqdm
from util import hdf5_function, util_python
import numpy as np
from torch.utils.data.dataset import Subset


def DivideTrainValDataset(dataset):
    n_samples = 10000
    train_size = int(n_samples * 0.95)
    train_dataset_indices = list(range(0, train_size))
    val_dataset_indices = list(range(train_size, n_samples))
    train_dataset = Subset(dataset, train_dataset_indices)
    val_dataset = Subset(dataset, val_dataset_indices)
    return train_dataset, val_dataset

def collate_fn(batch):
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        meta.update({key:np.array([d[key] for d in batch])})

def make_pose_data(hdf5_object):
    x_data = []
    y_data = []
    for i in tqdm(range(hdf5_function.get_len_hdf5(hdf5_object))):
        pcl_data = hdf5_object["data_" + str(i+1)]['pcl'][()]
        pose_data = hdf5_object["data_" + str(i+1)]['pose'][()]
        pose_data = util_python.conv_quat2mat(pose_data)
        x_data.append(pcl_data)
        y_data.append(pose_data)
    return x_data, y_data

def getNormalizedPcd_nodown(np_cloud):
    pcd_offset = np.expand_dims(np.mean(np_cloud, axis=0), 0)
    pcd_data = np_cloud - pcd_offset  #original
    return pcd_data, pcd_offset