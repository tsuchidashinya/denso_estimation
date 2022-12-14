import numpy as np
import cv2
import torch
from torchvision import ops as ops
import os


def encode_bboxes(bboxes, pad_info):
    scale_x, scale_y, dx, dy = pad_info
    bboxes *= np.array([scale_x, scale_y, scale_x, scale_y])
    bboxes[:, 0] += dx
    bboxes[:, 1] += dy
    return bboxes

def coco_to_yolo(bboxes):
    x, y, w, h = np.split(bboxes, 4, axis=-1)
    cx, cy = x + w / 2, y + h / 2
    bboxes = np.concatenate((cx, cy, w, h), axis=-1)
    return bboxes

def letterbox(img, img_size, jitter=0, random_placing=False):
    org_h, org_w, _ = img.shape
    if jitter:
        dw = jitter * org_w
        dh = jitter * org_h
        new_aspect = (org_w + np.random.uniform(low=-dw, high=dw)) / (
            org_h + np.random.uniform(low=-dh, high=dh)
        )
    else:
        new_aspect = org_w / org_h

    if new_aspect < 1:
        new_w = int(img_size * new_aspect)
        new_h = img_size
    else:
        new_w = img_size
        new_h = int(img_size / new_aspect)

    if random_placing:
        dx = int(np.random.uniform(img_size - new_w))
        dy = int(np.random.uniform(img_size - new_h))
    else:
        dx = (img_size - new_w) // 2
        dy = (img_size - new_h) // 2
    img = cv2.resize(img, (new_w, new_h))
    pad_img = np.full((img_size, img_size, 3), 127, dtype=np.uint8)
    pad_img[dy : dy + new_h, dx : dx + new_w, :] = img
    scale_x = np.float32(new_w / org_w)
    scale_y = np.float32(new_h / org_h)
    pad_info = (scale_x, scale_y, dx, dy)
    return pad_img, pad_info

def filter_boxes(output, conf_threshold, iou_threshold):
    keep_rows, keep_cols = (
        (output[:, 5:] * output[:, 4:5] >= conf_threshold).nonzero().T
    )
    if not keep_rows.nelement():
        return []

    conf_filtered = torch.cat(
        (
            output[keep_rows, :5],
            output[keep_rows, 5 + keep_cols].unsqueeze(1),
            keep_cols.float().unsqueeze(1),
        ),
        1,
    )
    nms_filtered = []
    detected_classes = conf_filtered[:, 6].unique()
    for c in detected_classes:
        detections_class = conf_filtered[conf_filtered[:, 6] == c]
        keep_indices = ops.nms(
            detections_class[:, :4],
            detections_class[:, 4] * detections_class[:, 5],
            iou_threshold,
        )
        detections_class = detections_class[keep_indices]
        nms_filtered.append(detections_class)
    nms_filtered = torch.cat(nms_filtered)
    return nms_filtered

def decode_bboxes(bboxes, info_img):
    scale_x, scale_y, dx, dy = info_img
    bboxes -= torch.stack([dx, dy, dx, dy])
    bboxes /= torch.stack([scale_x, scale_y, scale_x, scale_y])
    return bboxes

def yolo_to_pascalvoc(bboxes):
    cx, cy, w, h = torch.chunk(bboxes, 4, dim=1)
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    bboxes = torch.cat((x1, y1, x2, y2), dim=1)
    return bboxes

def postprocess(outputs, conf_threshold, iou_threshold, pad_info):
    decoded = []
    for output, *pad_info in zip(outputs, *pad_info):
        output[:, :4] = yolo_to_pascalvoc(output[:, :4])
        output = filter_boxes(output, conf_threshold, iou_threshold)
        if len(output):
            output[:, :4] = decode_bboxes(output[:, :4], pad_info)
        decoded.append(output)
    return decoded

def output_to_dict(output, class_names):
    detection = []
    for x1, y1, x2, y2, obj_conf, class_conf, label in output:
        bbox = {
            "confidence": float(obj_conf * class_conf),
            "class_id": int(label),
            "class_name": class_names[int(label)],
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        }
        detection.append(bbox)
    return detection

def change_label_file(label_dir):
    file_list = os.listdir(label_dir)
    for file in file_list:
        file_path = os.path.join(label_dir, file)
        f = open(file_path, "r")
        textline_list = f.readlines()
        text_list = []
        for textline in textline_list:
            text_list.append(textline.split())
        fw = open(file_path, "w")
        for textline_list in text_list:
            for i in range(len(textline_list)):
                if i == 0:
                    fw.write("HV8_occuluder ")
                elif i == 1 or i == 3:
                    fw.write(textline_list[i] + " ")
                elif i == 2:
                    fw.write(textline_list[4] + " ")
                elif i == 4:
                    fw.write(textline_list[2] + "\n")

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    args = parser.parse_args()
    change_label_file(args.folder)
