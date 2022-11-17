import torch.utils.data
from torch.utils.data.dataset import Subset

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