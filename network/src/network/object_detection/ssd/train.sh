#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/train.py \
--dataset_dir /home/ericlab/tsuchida/2023_01/annotation/ObjectDetection/SSD/HV8/full_randomizer \
--config-file /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/config/yaml/vgg_ssd512_voc0712.yaml \
--checkpoints /home/ericlab/tsuchida/2023_01/checkpoints/SSD/HV8/full_randomizer_1000 \
--save_step 1000
# --num_epoch 100 \
# --save_step 20 