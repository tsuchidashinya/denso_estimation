#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/demo.py \
--images_dir /home/ericlab/tsuchida/2022_07/annotation/real_data_HV8/input \
--config-file /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/config/yaml/vgg_ssd512_voc0712.yaml \
--checkpoints_dir /home/ericlab/tsuchida/2022_12/checkpoints/SSD/full_randomize/HV8 \
--output_dir /home/ericlab/tsuchida/2022_07/annotation/real_data_HV8/output_1 \
--score_threshold 0.5
# --num_epoch 100 \
# --save_step 20 