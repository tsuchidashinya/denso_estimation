#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/semantic_segmentation/pointnet_semantic/train.py \
--dataset_path /home/ericlab/tsuchida/2022_12/annotation/Semseg/t_pipe/sensor_b_box_3000_noize.hdf5 \
--checkpoints /home/ericlab/tsuchida/2023_02/checkpoints/Semseg/t_pipe/3000_propose \
--batch_size 1 \
--num_epoch 200 \
--save_epoch_freq 20 \
--lr 0.0001 \
--num_instance_classes 2 \
--start_index 0 
