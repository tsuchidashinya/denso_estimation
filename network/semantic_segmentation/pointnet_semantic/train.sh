#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/semantic_segmentation/pointnet_semantic/train.py \
--dataset_path /home/ericlab/tsuchida/2022_05/annotation/SemSeg/5_8_2_6_39_default/t_pipe_1000.hdf5 \
--checkpoints /home/ericlab/tsuchida/2022_11/checkpoints/try/semseg \
--batch_size 1 \
--num_epoch 200 \
--save_epoch_freq 20 \
--lr 0.0001 \
--num_instance_classes 3 \
--start_index 1
