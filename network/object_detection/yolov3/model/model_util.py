import numpy as np
import torch

def conv_initializer(param):
    out_ch, in_ch, h, w = param.shape
    fan_in = h * w * in_ch
    scale = np.sqrt(2 / fan_in)
    w = scale * torch.randn_like(param)
    return w

def parse_conv_block(module, weights, offset, initialize):
    conv, bn, leakey = module
    params = [
        bn.bias,
        bn.weight,
        bn.running_mean,
        bn.running_var,
        conv.weight,
    ]
    for param in params:
        if initialize:
            if param is bn.weight:
                w = torch.ones_like(param)
            elif param is conv.weight:
                w = conv_initializer(param)
            else:
                w = torch.zeros_like(param)
        else:
            param_len = param.numel()
            w = torch.from_numpy(weights[offset : offset + param_len]).view_as(param)
            offset += param_len

        param.data.copy_(w)
    return offset

def parse_yolo_block(module, weights, offset, initialize):
    conv = module._modules["conv"]
    for param in [conv.bias, conv.weight]:
        if initialize:
            if param is conv.bias:
                w = torch.zeros_like(param)
            else:
                w = conv_initializer(param)
        else:
            param_len = param.numel()
            w = torch.from_numpy(weights[offset : offset + param_len]).view_as(param)
            offset += param_len
        param.data.copy_(w)
    return offset

def parse_yolo_weights(model, weights_path):
    with open(weights_path, "rb") as f:
        f.read(20)
        weights = np.fromfile(f, dtype=np.float32)
    offset = 0
    initialize = False

    for module in model.module_list:
        if module._get_name() == "Sequential":
            offset = parse_conv_block(module, weights, offset, initialize)
        elif module._get_name() == "resblock":
            for resblocks in module._modules["module_list"]:
                for resblock in resblocks:
                    offset = parse_conv_block(resblock, weights, offset, initialize)
        elif module._get_name() == "YOLOLayer":
            offset = parse_yolo_block(module, weights, offset, initialize)
        initialize = offset >= len(weights)