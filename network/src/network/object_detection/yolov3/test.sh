#!/usr/bin/env bash

python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/yolov3/test.py \
--dataset_path /home/ericlab/tsuchida/2022_11/annotation/YOLO/try_10/images \
--checkpoints /home/ericlab/tsuchida/2022_11/checkpoints/Yolo/180.pth \
--config_path /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/yolov3/config/yolov3_denso.yaml \
--output_dir /home/ericlab/tsuchida/2022_11/annotation/YOLO/try_10/output