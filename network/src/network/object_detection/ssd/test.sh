#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/test.py \
--dataset_dir /home/ericlab/tsuchida/2022_07/annotation/real_data_HV8 \
--config-file /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/config/yaml/vgg_ssd512_voc0712.yaml \
--checkpoints_dir /home/ericlab/tsuchida/2022_12/checkpoints/SSD/latest_method \
--output_dir /home/ericlab/tsuchida/2022_11/annotation/YOLO/object_bbox/output
# --num_epoch 100 \
# --save_step 20 