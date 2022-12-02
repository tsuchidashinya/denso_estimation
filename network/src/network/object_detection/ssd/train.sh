#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/train.py \
--dataset_dir /home/ericlab/tsuchida/2022_11/annotation/YOLO/ishiyama_demo_dataset \
--weights /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/yolov3/weights/darknet53.conv.74 \
--config-file /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/config/yaml/vgg_ssd512_voc0712.yaml \
--checkpoints /home/ericlab/tsuchida/2022_11/checkpoints/SSD/ishiyama \
# --num_epoch 100 \
# --save_step 20 