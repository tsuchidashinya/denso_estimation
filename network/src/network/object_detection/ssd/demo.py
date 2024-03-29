import glob
import os
import time
from pathlib import Path
import torch
from PIL import Image
from vizer.draw import draw_boxes

from network.object_detection.ssd.config import cfg
from network.object_detection.ssd.data.datasets import COCODataset, VOCDataset, VOCDatasetDenso
from util_package import util
import argparse
import numpy as np

from network.object_detection.ssd.data.transforms import build_transforms
from network.object_detection.ssd.modeling.detector import build_detection_model
from network.object_detection.ssd.utils import mkdir
from network.object_detection.ssd.utils.checkpoint import CheckPointer

from common_msgs.msg import BoxPosition


class SSDEstimation:
    def __init__(self):
        pass
    
    def setting_network(self, config_path, checkpoint_file, score_threshold, device):
        self.config = SSDEstimation.load_config(config_path)
        self.device = device
        self.model = SSDEstimation.create_model(self.config, self.device)
        check_dir, check_file = util.devide_dir_and_file(checkpoint_file)
        self.model = SSDEstimation.load_checkpoints(self.model, check_dir, check_file)
        self.transform = build_transforms(self.config, is_train=False)
        self.cpu_device = torch.device("cpu")
        self.score_threshold = score_threshold

    @staticmethod
    def load_checkpoints(model, checkpoint_dir, checkpoint_file):
        checkpointer = CheckPointer(model, save_dir=checkpoint_dir)
        path = os.path.join(checkpoint_dir, checkpoint_file)
        checkpointer.load(path, use_latest=False)
        return model
    
    @staticmethod
    def load_config(config_path):
        cfg.merge_from_file(config_path)
        cfg.freeze()
        return cfg

    @staticmethod
    def create_model(config, device):
        model = build_detection_model(config)
        model = model.to(device=device).eval()
        return model

    @staticmethod
    def get_class_names():
        return VOCDatasetDenso.class_names

    @staticmethod
    def draw_b_box(image, boxes, labels, scores, class_names):
        drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
        return drawn_image

    @torch.no_grad()
    def object_detection(self, image, score_threshold):
        transforms = build_transforms(cfg, is_train=False)
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        result = self.model(images.to(self.device))[0]
        result = result.resize((width, height)).to(self.cpu_device).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']
        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        return boxes, labels, scores
    
    @staticmethod
    def get_box_position(boxes):
        box_pos_list = []
        for i in range(boxes.shape[0]):
            box_pos = BoxPosition()
            box_pos.x_one = boxes[i][0]
            box_pos.y_one = boxes[i][1]
            box_pos.x_two = boxes[i][2]
            box_pos.y_two = boxes[i][3]
            box_pos_list.append(box_pos)
        return box_pos_list

@torch.no_grad()
def run_demo(cfg, args, ckpt, score_threshold, images_dir, output_dir, dataset_type):
    if dataset_type == "voc":
        # class_names = VOCDataset.class_names
        class_names = VOCDatasetDenso.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    else:
        raise NotImplementedError('Not implemented now.')
    device = torch.device(cfg.MODEL.DEVICE)

    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=args.checkpoints_dir)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()
    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)

        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        result = model(images.to(device))[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).to(cpu_device).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))

        drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
        Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))


def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--checkpoints_dir", type=Path)
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument("--images_dir", default='demo', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='demo/result', type=str, help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="voc", type=str, help='Specify dataset type. Currently support voc and coco.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    # print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # print("Loaded configuration file {}".format(args.config_file))
    # with open(args.config_file, "r") as cf:
    #     config_str = "\n" + cf.read()
        # print(config_str)
    # print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
            args=args,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()
