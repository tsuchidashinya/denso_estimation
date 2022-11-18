import torch
import torch.utils.data
import os
from model import YOLO
import yaml
import argparse
from data import data_util
from pathlib import Path
import cv2
from data import test_dataset
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import numpy as np


font_path = str(Path(__file__).parent / "font/ipag.ttc")

def get_text_color(color):
    value = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
    return "black" if value > 128 else "white"

def get_net_output(net, input,  pad_info, config_object, class_list):
    conf_threshhold = config_object["test"]["conf_threshold"]
    nms_threshhold = config_object["test"]["nms_threshold"]
    with torch.no_grad():
        outputs = net(input)
        outputs = data_util.postprocess(outputs, conf_threshhold, nms_threshhold, pad_info)
        detections = [data_util.output_to_dict(x, class_list) for x in outputs]
        return detections
    
def load_checkpoints(net, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, device)
    net.load_state_dict(state_dict, strict=False)
    return net

def get_class_names(config_path):
    config_path = Path(config_path)
    config_object = load_config(config_path)
    path = os.path.join(os.path.dirname(config_path), config_object["model"]["class_names"])
    with open(path) as f:
        class_names = [x.strip() for x in f.read().splitlines()]
        return class_names

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(config_model, device):
    net = YOLO.YOLOv3(config_model["model"])
    return net.to(device)

def load_config(config_path):
    with open(config_path) as f:
        config_object = yaml.safe_load(f)
    return config_object

def object_detection(input_data, net, config):
    pad_img, pad_info = data_util.letterbox(input_data, config["test"]["img_size"])
    pad_info = [x.to(device) for x in pad_info]
    outputs = get_net_output(net, pad_img, config["test"]["conf_threshold"], config["test"]["nms_threshold"], pad_info)
    return outputs

def draw_boxes(img, boxes, n_classes):
    draw_mae = img
    draw = ImageDraw.Draw(img, mode="RGBA")
    cmap = plt.cm.get_cmap("hsv", n_classes)
    fontsize = max(3, int(0.01 * min(img.size)))
    font = ImageFont.truetype(font_path, size=fontsize)
    for box in boxes:
        # 矩形を画像の範囲内にクリップする。
        x1 = int(np.clip(box["x1"], 0, img.size[0] - 1))
        y1 = int(np.clip(box["y1"], 0, img.size[1] - 1))
        x2 = int(np.clip(box["x2"], 0, img.size[0] - 1))
        y2 = int(np.clip(box["y2"], 0, img.size[1] - 1))
        caption = box["class_name"]
        if caption == 'traffic light':
            print("traffic light")
            img_2 = np.asarray(draw_mae)
            img_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR)
            cv2.imshow("img_2", img_2)
            img_crop = draw_mae.crop((x1-10, y1-10, x2+10, y2+10))
            img_3 = np.asarray(img_crop)
            img_3 = cv2.cvtColor(img_3, cv2.COLOR_RGB2BGR)
            cv2.imshow("img_3", img_3)
            cv2.waitKey(1)
        if "confidence" in box:
            caption += f" {box['confidence']:.0%}"
        color = tuple(cmap(box["class_id"], bytes=True))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_size = draw.textsize(caption, font=font)
        text_origin = np.array([x1, y1])
        text_color = get_text_color(color)
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + text_size - 1)], fill=color
        )
        draw.text(text_origin, caption, fill=text_color, font=font)


if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_path', help='Dataset root directory path')
    parser.add_argument('--checkpoints', default='weights/', help='Directory for saving checkpoint models')
    parser.add_argument('--num_instance_classes', default=2, type=int)
    args = parser.parse_args()
    device = get_device()
    config_object = load_config(args.config_path)
    class_list = get_class_names(args.config_path)
    img_size = config_object["test"]["img_size"]
    net = create_model(config_object, device)
    net = load_checkpoints(net, args.checkpoints, device)
    dataset = test_dataset.ImageFolder(args.dataset_path, img_size)
    dataloader = torch.utils.data.DataLoader(dataset, config_object["test"]["batch_size"])
    detections, image_paths = [], []
    for inputs, pad_infos, paths in dataloader:
        inputs = inputs.to(device)
        pad_infos = [x.to(device) for x in pad_infos]
        detections += get_net_output(net, inputs, pad_infos, config_object, class_list)
        image_paths += paths
    for detection, image_path in zip(detections, image_paths):
        for box in detection:
            print(
                f"{box['class_name']} {box['confidence']:.0%} "
                f"({box['x1']:.0f}, {box['y1']:.0f}, {box['x2']:.0f}, {box['y2']:.0f})"
            )
        img = Image.open(image_path)
        draw_boxes(img, detection, len(class_list))
        img.save(args.output / Path(image_path).name)
        
