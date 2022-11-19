#!/usr/bin/env bash

python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/semantic_segmentation/pointnet_semantic/test.py \
--dataset_path /home/ericlab/tsuchida/2022_05/annotation/SemSeg/5_8_2_6_39_default/t_pipe_1000.hdf5 \
--checkpoints /home/ericlab/tsuchida/2022_11/checkpoints/try/semseg/180.pth \
--num_instance_classes 3