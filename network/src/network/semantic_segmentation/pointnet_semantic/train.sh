#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/semantic_segmentation/pointnet_semantic/train.py \
--dataset_path /home/ericlab/tsuchida/2023_01/annotation/Semseg/HV8/data_1000/1000_propose.hdf5 \
--checkpoints /home/ericlab/tsuchida/2023_01/checkpoints/Semseg/HV8/1000_propose \
--batch_size 1 \
--num_epoch 200 \
--save_epoch_freq 20 \
--lr 0.0001 \
--num_instance_classes 2 \
--start_index 0 
