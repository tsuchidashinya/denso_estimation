#!/usr/bin/env bash

python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/semantic_segmentation/pointnet_semantic/test.py \
--dataset_path /home/ericlab/tsuchida/2022_11/annotation/segmentation/11_15_9_32_59/honban_11_15_13_39_21.hdf5 \
--checkpoints /home/ericlab/tsuchida/2022_11/checkpoints/SemSeg/semantic_segmentation/ishiyama/PointNet_Segmentation/honban_11_15_13_39_21/latest_net.pth \
--num_instance_classes 3