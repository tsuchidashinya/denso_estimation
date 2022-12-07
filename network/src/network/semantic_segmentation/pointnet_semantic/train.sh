#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/semantic_segmentation/pointnet_semantic/train.py \
--dataset_path /home/ericlab/tsuchida/2022_12/annotation/Semseg/multi_object_kai/kai3228/kai_1.hdf5 \
--checkpoints /home/ericlab/tsuchida/2022_12/checkpoints/Semseg/multi_object_kai_2000 \
--batch_size 1 \
--num_epoch 200 \
--save_epoch_freq 20 \
--lr 0.0001 \
--num_instance_classes 2 \
--start_index 1
