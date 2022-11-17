from tqdm import tqdm
from util import hdf5_function, util_python
import numpy as np
from torch.utils.data.dataset import Subset


def print_current_losses(self, phase, epoch, i, losses):
    """ prints train loss on terminal / file """
    message = '(phase: %s, epoch: %d, iters: %d) loss: %.3f ' %(phase, epoch, i, losses)
    print(message)

def DivideTrainValDataset(dataset):
    n_samples = len(dataset)
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
    return meta

def get_normalizedcloud(input_cloud):
    cloud_offset = np.expand_dims(np.mean(input_cloud, axis=0), 0)
    cloud_data = input_cloud - cloud_offset  #original
    return cloud_data, cloud_offset

def get_normalizedcloud_segmentation(input_cloud):
    cloud_offset = np.expand_dims(np.mean(input_cloud[:,:3], axis=0),0)
    pre_cloud = input_cloud[:,:3] - cloud_offset
    mask_data = input_cloud[:,3]
    mask_data = np.expand_dims(mask_data, 1)
    cloud_data = np.hstack([pre_cloud, mask_data])
    return cloud_data, cloud_offset[0]