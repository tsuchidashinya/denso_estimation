#!/usr/bin/env bash

python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/pose_estimation/pointnet_pose/train.py \
--dataset_path /home/ericlab/Downloads/HV8_2000_-pi_-3pi4_3pi4_pi.hdf5 \
--checkpoints /home/ericlab/tsuchida/2022_11/checkpoints/try \
--batch_size 1 \
--num_epoch 200 \
--lr 0.0001 \
--save_epoch_freq 10 \
--start_index 1