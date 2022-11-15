#!/usr/bin/env bash

python3 /home/ericlab/ros_package/integrate_ws/src/integrate_pkgs/networks/raugh_recognition/pointnet_pose/train.py \
--dataroot /home/ericlab/M2/Annotation/train/annotation/pose/5_9_23_22_0 \
--dataset_model HV8_10000.hdf5 \
--checkpoints_dir /home/ericlab/M2/train/Pose/range_change \
--resolution 1024 \
--phase train \
--dataset_mode pose_estimation \
--batch_size 1 \
--num_epoch 200 \
--max_dataset_size 10000 \
--arch PointNet_Pose \
--print_freq 1000 \
--save_latest_freq 50000 \
--save_epoch_freq 10 \
--run_test_freq 1 \
--gpu_ids 0 \
--gpu_num 1 \
--num_threads 0 \
--serial_batches False \
--verbose_plot True \
--lr 0.0001 \
--checkpoints_human_swich tsuchida_raugh \
--dataroot_swich front \
--local_checkpoints_dir /home/ericlab/MEGA/DENSO/HV8/ラフ認識/ラフ認識 \
--tensorboardX_results_directory /home/ericlab/M2/train/Pose/tensorboards \
--tensorboardX_results_directory_switch 0614 \
