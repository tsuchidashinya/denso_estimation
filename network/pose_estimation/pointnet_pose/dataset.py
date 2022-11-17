import torch.utils.data
from torch.utils.data.dataset import Subset
import numpy as np
from tqdm import tqdm
import h5py
import torch 
import argparse


def TrainValDataset(opt):
    print(opt.dataroot)
    dataset = PoseData(opt)
    n_samples = len(dataset)
    train_size = int(n_samples * 0.95)

    subset1_indices = list(range(0, train_size))
    subset2_indices = list(range(train_size, n_samples))

    subset1 = Subset(dataset, subset1_indices) #set train_data and index(対応付け)
    subset2 = Subset(dataset, subset2_indices)
    return subset1, subset2

def collate_fn(batch):
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        meta.update({key:np.array([d[key] for d in batch])})
        
        


class PCD_Loader(torch.utils.data.Dataset):
    def __init__(self, dir_name, dataset_model, dataset_size, dataset_number, opt):
        super(PCD_Loader, self).__init__(dir_name, dataset_model, dataset_size, dataset_number)

    def load_hdf5(self):
        for i in range(self.dataset_number):
            path = self.find_h5py_filenames(self.dir)[i] #get file_name
            dir_path = self.dir+"/"+path #get path
            self.hdf5_file = h5py.File(dir_path, "r")
            for n in tqdm(range(0, self.dataset_size[i])):
                pcl_data = self.hdf5_file["data_" + str(n + 1)]['pcl'][()]
                pose_data = self.hdf5_file["data_" + str(n + 1)]['pose'][()]
                pose_data = self.conv_quat2mat(pose_data)   #7to12
                self.x_data.append(pcl_data)
                self.y_data.append(pose_data)

    def get_pcd_data(self, index):
        pcd_data = self.x_data[index]
        x_data, pcd_offset = getNormalizedPcd_nodown(pcd_data)
        y_data = self.y_data[index]
        y_pos = y_data[0:3] - pcd_offset[0]
        y_rot = y_data[3:]
        y_data = np.concatenate([y_pos, y_rot])
        return x_data, y_data

class PoseData(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.resolution = opt.resolution
        self.hdf5_data = PCD_Loader(self.dataroot, self.dataset_model, self.size, self.dataset_number, opt)
        self.hdf5_data.load_hdf5()

    def __getitem__(self, index):
        meta = {}
        x_data, y_data = self.hdf5_data.get_pcd_data(index)
        
        meta["x_data"] = x_data
        meta["y_data"] = y_data
        return meta
            


    def __len__(self):
        return self.len_size