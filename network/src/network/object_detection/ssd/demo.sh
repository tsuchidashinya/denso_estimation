#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/demo.py \
--images_dir /home/ericlab/tsuchida/2022_11/annotation/YOLO/retry_1000/inputs \
--config-file /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/config/yaml/vgg_ssd512_voc0712.yaml \
--checkpoints_dir /home/ericlab/tsuchida/2022_11/checkpoints/SSD \
--output_dir /home/ericlab/tsuchida/2022_11/annotation/YOLO/retry_1000/output \
--score_threshold 0.9
# --num_epoch 100 \
# --save_step 20 