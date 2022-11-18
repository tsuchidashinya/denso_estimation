#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/object_detection/yolov3/train.py \
--dataset_dir /home/ericlab/tsuchida/2022_05/annotation/YOLO/5_7_22_25_48 \
--weights /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/object_detection/yolov3/weights/darknet53.conv.74 \
--config_path /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/object_detection/yolov3/config/yolov3_denso.yaml \
--checkpoints /home/ericlab/tsuchida/2022_11/checkpoints/try/yolo \
--num_epoch 200 \
--save_epoch_freq 20 
