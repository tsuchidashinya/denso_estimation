#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/train.py \
--dataset_dir /home/ericlab/tsuchida/2023_02/annotation/ObjectDetection/ssd/t_pipe_full_radious_0_04 \
--config-file /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/config/yaml/vgg_ssd512_voc0712.yaml \
--checkpoints /home/ericlab/tsuchida/2023_02/checkpoints/SSD/t_pipe/t_pipe_full_radious_0_04_3000 \
--save_step 1000
# --num_epoch 100 \
# --save_step 20 