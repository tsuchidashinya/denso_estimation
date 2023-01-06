#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/train.py \
--dataset_dir /home/ericlab/tsuchida/2022_12/annotation/ObjectDetection/SSD/t_pipe/full_randomizer \
--config-file /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/config/yaml/vgg_ssd512_voc0712.yaml \
--checkpoints /home/ericlab/tsuchida/2022_12/checkpoints/SSD/full_randomize/t_pipe \
# --num_epoch 100 \
# --save_step 20 