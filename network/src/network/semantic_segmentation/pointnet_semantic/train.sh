#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/semantic_segmentation/pointnet_semantic/train.py \
--dataset_path /home/ericlab/tsuchida/2022_11/annotation/segmentation/ano_execute1443/ano_execute_1123145816.hdf5 \
--checkpoints /home/ericlab/tsuchida/2022_11/checkpoints/SemSeg/semantic_11_23 \
--batch_size 1 \
--num_epoch 200 \
--save_epoch_freq 20 \
--lr 0.0001 \
--num_instance_classes 2 \
--start_index 1
