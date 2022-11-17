#!/usr/bin/env bash

python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/pose_estimation/pointnet_pose/train.py \
--dataset_path /home/ericlab/tsuchida/2022_05/annotation/pose/5_11_9_59_54/t_pipe_10000.hdf5 \
--checkpoints /home/ericlab/tsuchida/2022_11/checkpoints/try \
--batch_size 1 \
--num_epoch 200 \
--lr 0.0001 \
--save_epoch_freq 10