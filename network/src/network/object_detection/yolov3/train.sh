#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/yolov3/train.py \
--dataset_dir /home/ericlab/tsuchida/2022_11/annotation/YOLO/retry_1000 \
--weights /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/yolov3/weights/darknet53.conv.74 \
--config_path /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/yolov3/config/yolov3_denso.yaml \
--checkpoints /home/ericlab/tsuchida/2022_11/checkpoints/Yolo \
--num_epoch 100 \
--save_epoch_freq 20 
