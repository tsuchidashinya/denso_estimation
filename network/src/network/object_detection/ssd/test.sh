#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/test.py \
--dataset_dir /home/ericlab/tsuchida/2023_01/annotation/ObjectDetection/SSD/HV8/test \
--config-file /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/config/yaml/vgg_ssd512_voc0712.yaml \
--checkpoints_dir /home/ericlab/tsuchida/2022_12/checkpoints/SSD/full_randomize/HV8 \
--output_dir /home/ericlab/tsuchida/2023_01/output
# --num_epoch 100 \
# --save_step 20 