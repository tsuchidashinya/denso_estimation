#!/usr/bin/env bash

python3 /home/ericlab/ros_package/integrate_ws/src/integrate_pkgs/networks/raugh_recognition/PointNetPose/train.py \
--dataroot /home/ericlab/tsuchida/2022_05/annotation/pose/5_11_12_59_48_random \
--dataset_model t_pipe_10000.hdf5 \
--checkpoints_dir /home/ericlab/tsuchida/2022_05/checkpoints/pose/random \
--resolution 1024 \
--phase train \
--dataset_mode pose_estimation \
--batch_size 1 \
--num_epoch 200 \
--max_dataset_size 10000 \
--arch PointNetPose \
--print_freq 1000 \
--save_latest_freq 20000 \
--save_epoch_freq 5 \
--run_test_freq 1 \
--gpu_ids 0 \
--gpu_num 1 \
--num_threads 0 \
--serial_batches False \
--verbose_plot True \
--lr 0.0001 \
--checkpoints_human_swich tsuchida_raugh \
--dataroot_swich front \
--local_checkpoints_dir /home/ericlab/DENSO_results/raugh_recognition/checkpoint \
--tensorboardX_results_directory /home/ericlab/ros_package/integrate_ws/src/networks/raugh_recognition/PointNetPose/tensorboardX \
--tensorboardX_results_directory_switch tsuchida_raugh/0628 \