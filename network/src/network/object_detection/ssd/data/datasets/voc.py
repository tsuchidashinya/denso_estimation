import os
import torch.utils.data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from util import util
from network.object_detection.ssd.structures.container import Container


class VOCDataset(torch.utils.data.Dataset):
    class_names = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, data_dir, split, transform=None, target_transform=None, keep_difficult=False):
        """Dataset for VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        image_sets_file = os.path.join(self.data_dir, "ImageSets", "Main", "%s.txt" % self.split)
        # image_sets_dir = os.path.join(self.data_dir, "images")
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        # self.ids = VOCDataset._read_image_list(image_sets_dir)
        self.keep_difficult = keep_difficult

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        return image, targets, index

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)
        # return image_id, self._get_annotation_denso(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    @staticmethod
    def _read_image_list(image_sets_dir):
        # ids = []
        # with open(image_sets_file) as f:
        #     for line in f:
        #         ids.append(line.rstrip())
        ids = os.listdir(image_sets_dir)
        return ids

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % image_id)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _get_annotation_denso(self, image_id):
        annotation_file_path = os.path.join(self.data_dir, "labels", "%s.txt" % image_id)
        lines = open(annotation_file_path).read().splitlines()
        boxes = []
        labels = []
        is_difficult = []
        for line in lines:
            class_name, *coords = line.split()
            coords = list(map(float, coords))
            x1, y1, x2, y2 = coords
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult.append(0)
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))


    def get_img_info(self, index):
        img_id = self.ids[index]
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % img_id)
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "JPEGImages", "%s.jpg" % image_id)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image


class VOCDatasetDenso(torch.utils.data.Dataset):
    class_names = ('__background__',
                   'HV8_occuluder')

    def __init__(self, data_dir, transform=None, target_transform=None, keep_difficult=False):
        """Dataset for VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        # image_sets_file = os.path.join(self.data_dir, "ImageSets", "Main", "%s.txt" % self.split)
        image_sets_dir = os.path.join(self.data_dir, "images")
        # self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.ids = VOCDataset._read_image_list(image_sets_dir)
        self.keep_difficult = keep_difficult

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation_denso(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        return image, targets, index

    def get_annotation(self, index):
        image_id = self.ids[index]
        # return image_id, self._get_annotation(image_id)
        return image_id, self._get_annotation_denso(image_id)

    def __len__(self):
        return len(self.ids)


    @staticmethod
    def _read_image_list(image_sets_dir):
        # ids = []
        # with open(image_sets_file) as f:
        #     for line in f:
        #         ids.append(line.rstrip())
        ids = os.listdir(image_sets_dir)
        return ids

    

    def _get_annotation_denso(self, image_id):
        image_id = util.exclude_ext_str(image_id)
        annotation_file_path = os.path.join(self.data_dir, "labels", "%s.txt" % image_id)
        if os.path.exists(annotation_file_path):
            lines = open(annotation_file_path).read().splitlines()
            boxes = []
            labels = []
            is_difficult = []
            for line in lines:
                class_name, *coords = line.split()
                coords = list(map(float, coords))
                x1, y1, x2, y2 = coords
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_dict[class_name])
                is_difficult.append(0)
            return (np.array(boxes, dtype=np.float32),
                    np.array(labels, dtype=np.int64),
                    np.array(is_difficult, dtype=np.uint8))
        else:
            print(annotation_file_path)
        


    def get_img_info(self, index):
        img_id = self.ids[index]
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % img_id)
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def _read_image(self, image_id):
        image_id = util.exclude_ext_str(image_id)
        # image_file = os.path.join(self.data_dir, "JPEGImages", "%s.jpg" % image_id)
        image_file = os.path.join(self.data_dir, "images", "%s.jpg" % image_id)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image