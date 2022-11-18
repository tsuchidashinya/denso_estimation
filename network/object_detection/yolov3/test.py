import torch
import os
from model import YOLO
import yaml
import argparse
from data import data_util
from pathlib import Path

def get_net_output(net, pad_img, conf_threshhold, nms_threshhold, pad_info):
    with torch.no_grad():
        outputs = net(pad_img)
        outputs = data_util.postprocess(outputs, conf_threshhold, nms_threshhold, pad_info)
        outputs = data_util.output_to_dict(outputs, )
        return outputs

def load_checkpoints(net, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, device)
    net.load_state_dict(state_dict, strict=False)
    return net

def get_class_names(config_path):
    config_path = Path(config_path)
    config_object = load_config(config_path)
    path = os.path.join(config_path, config_object["model"]["class_names"])
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

def object_detection(net, config, input_data):
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_path', help='Dataset root directory path')
    parser.add_argument('--checkpoints', default='weights/', help='Directory for saving checkpoint models')
    parser.add_argument('--num_instance_classes', default=2, type=int)
    args = parser.parse_args()