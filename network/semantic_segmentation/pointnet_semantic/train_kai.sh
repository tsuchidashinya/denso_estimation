#!/usr/bin/env bash

python3 /home/fukitani/rospackage_last/integrate_ws/src/integrate_pkgs/networks/semantic_segmentation/pointnet_semantic/train.py \
--dataroot /home/fukitani/annotation/4_seg/t_pipe/hdf5_data \
--dataset_model t_pipe_1000_new.hdf5 \
--dataset_mode semantic_segmentation \
--checkpoints_dir /home/fukitani/annotation/4_seg/t_pipe/checkpoints \
--resolution 8192 \
--phase train \
--process_swich object_segment \
--batch_size 1 \
--num_epoch 200 \
--max_dataset_size 999 \
--arch PointNet_Segmentation \
--print_freq 10 \
--save_latest_freq 999 \
--save_epoch_freq 10 \
--run_test_freq 1 \
--gpu_ids 0 \
--gpu_num 1 \
--num_threads 0 \
--serial_batches False \
--verbose_plot True \
--lr 0.0001 \
--is_train True \
--checkpoints_human_swich ishiyama \
--dataroot_swich semantic_occlusion \
--local_checkpoints_dir /home/fukitani/annotation/4_seg/t_pipe/checkpoints/local \
--tensorboardX_results_directory /home/fukitani/annotation/4_seg/t_pipe/tensorboardX \
--tensorboardX_results_directory_switch ishiyama/0915_semantic \
--instance_number 2 \
--semantic_number 3
