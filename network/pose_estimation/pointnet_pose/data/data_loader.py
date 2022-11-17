import data.data_util as data_util
import torch.utils.data

class TrainDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset= dataset
        self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=data_util.collate_fn)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for _, data in enumerate(self.dataloader):
            print("yeild", data)
            yield data
