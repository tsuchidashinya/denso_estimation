import numpy as np
from torch.utils.data.dataset import Subset
import torch


def print_current_losses(phase, epoch, losses):
    """ prints train loss on terminal / file """
    message = '(phase: %s, epoch: %d) loss: %.3f ' %(phase, epoch, losses)
    print(message)

def DivideTrainValDataset(dataset):
    n_samples = len(dataset)
    train_size = int(n_samples * 1)
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

def batch_quat_to_rotmat(q, out=None):
    batchsize = q.size(0)
    if out is None:
        out = torch.FloatTensor(batchsize, 3, 3)
    # 2 / squared quaternion 2-norm
    s = 2/torch.sum(q.pow(2), 1)
    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))
    out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)
    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)
    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)
    return out
