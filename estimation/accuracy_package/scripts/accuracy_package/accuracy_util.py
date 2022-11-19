import numpy as np
from util import util_msg_data

def extract_ground_truth_parts(ground_truth_cloud, esti_cloud):
    gt_parts = np.empty((0, 4))
    for i in range(esti_cloud.shape[0]):
        for j in range(ground_truth_cloud.shape[0]):
            if esti_cloud[i][0] == ground_truth_cloud[j][0] and esti_cloud[i][1] == ground_truth_cloud[j][1] and esti_cloud[i][2] == ground_truth_cloud[j][2]:
                gt_parts = np.append(gt_parts, np.expand_dims(ground_truth_cloud[j], 0), axis=0)
                break
    return gt_parts

def calcurate_iou(ground_truth_parts, instance):
    tp = 0
    fp = 0
    fn = 0
    for i in range(ground_truth_parts.shape[0]):
        if ground_truth_parts[i][3] == instance:
            tp += 1
        else:
            fp += 1
    fn = ground_truth_parts.shape[0] - tp
    iou = tp / (tp + fp + fn)
    return iou