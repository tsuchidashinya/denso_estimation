#!/usr/bin/env bash
python3 /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/demo.py \
--images_dir /home/ericlab/tsuchida/2023_01/annotation/ObjectDetection/SSD/HV8/test/images \
--config-file /home/ericlab/tsuchida/ros_package/study_ws/src/denso_estimation/network/src/network/object_detection/ssd/config/yaml/vgg_ssd512_voc0712.yaml \
--checkpoints_dir /home/ericlab/tsuchida/2023_01/checkpoints/SSD/HV8/full_randomizer_1000 \
--output_dir /home/ericlab/tsuchida/2023_01/output \
--score_threshold 0.7
# --num_epoch 100 \
# --save_step 20 

# --checkpoints_dir /home/ericlab/tsuchida/2023_01/checkpoints/SSD/HV8/full_randomizer_3000 \
# --checkpoints_dir /home/ericlab/tsuchida/2022_12/checkpoints/SSD/full_randomize/HV8 \