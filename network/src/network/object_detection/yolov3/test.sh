#!/usr/bin/env bash

python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/object_detection/yolov3/test.py \
--dataset_path /home/ericlab/tsuchida/2022_05/annotation/YOLO/5_7_16_49_randomizer/input \
--checkpoints /home/ericlab/tsuchida/2022_05/checkpoints/YOLO/5_7_16_49_randomizer/yolov3_final.pth \
--config_path /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/object_detection/yolov3/config/yolov3_denso.yaml \
--output_dir /home/ericlab/tsuchida/2022_05/annotation/YOLO/5_7_16_49_randomizer/output