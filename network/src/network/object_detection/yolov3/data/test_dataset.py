import cv2
import numpy as np
import torch
from torchvision import transforms as transforms
from network.object_detection.yolov3.data import data_util


class ImageFolder(torch.utils.data.Dataset):
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, path, img_size):
        super().__init__()
        if path.is_dir():
            self.img_paths = self.get_img_paths(path)
        else:
            if not path.exists():
                raise FileExistsError(f"image path {path} does not exist.")
            self.img_paths = [path]
        self.img_size = img_size

    def __getitem__(self, index):
        path = str(self.img_paths[index])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, pad_info = data_util.letterbox(img, self.img_size)
        img = transforms.ToTensor()(img)

        return img, pad_info, path

    def __len__(self):
        return len(self.img_paths)

    def get_img_paths(self, img_dir):
        img_paths = [
            x for x in img_dir.iterdir() if x.suffix in ImageFolder.IMG_EXTENSIONS
        ]
        img_paths.sort()

        return img_paths


class ImageList(torch.utils.data.Dataset):
    def __init__(self, imgs, img_size):
        super().__init__()
        if isinstance(imgs, np.ndarray):
            self.imgs = [imgs]
        elif isinstance(imgs, list):
            self.imgs = imgs
        else:
            raise ValueError(
                f"imgs type {type(imgs)} should be ndarray or list of ndarray."
            )
        self.img_size = img_size

    def __getitem__(self, index):
        img = self.imgs[index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, pad_info = data_util.letterbox(img, self.img_size)
        img = transforms.ToTensor()(img)
        return img, pad_info

    def __len__(self):
        return len(self.imgs)
