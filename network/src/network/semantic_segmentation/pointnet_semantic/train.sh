#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/semantic_segmentation/pointnet_semantic/train.py \
--dataset_path /home/ericlab/Downloads/sensor_b_box_2000_noize_max_improve.hdf5 \
--checkpoints /home/ericlab/tsuchida/2023_01/checkpoints/Semseg/HV8/sensor_b_box_2000_noize_max_improve \
--batch_size 1 \
--num_epoch 200 \
--save_epoch_freq 20 \
--lr 0.0001 \
--num_instance_classes 2 \
--start_index 0 
