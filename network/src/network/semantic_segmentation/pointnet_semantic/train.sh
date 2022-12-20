#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/semantic_segmentation/pointnet_semantic/train.py \
--dataset_path /home/ericlab/tsuchida/2022_12/annotation/Semseg/data_num_experiment/data_4000/sensor_b_box_4000.hdf5 \
--checkpoints /home/ericlab/tsuchida/2022_12/checkpoints/Semseg/実験1/data_4000 \
--batch_size 1 \
--num_epoch 120 \
--save_epoch_freq 20 \
--lr 0.0001 \
--num_instance_classes 2 \
--start_index 1 \
--weights /home/ericlab/Downloads/data_4000/latest.pth
