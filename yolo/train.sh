#!/usr/bin/env bash

# https://github.com/CosmiQ/simrdwn#2-train

cp training_list.txt xview.pbtxt xview.tfrecord /raid/simrdwn/data


yolt() {
    # xView small_car search
    python /raid/simrdwn/core/simrdwn.py \
        --framework yolt \
        --mode train \
        --outname dense_cowc \
        --yolt_object_labels_str small_car \
        --yolt_cfg_file yolt.cfg  \
        --weight_dir /simrdwn/yolt/input_weights \
        --weight_file yolov2.weights \
        --yolt_train_images_list_file training_list.txt \
        --label_map_path /raid/simrdwn/data/xview.pbtxt \
        --max_batches 30000 \
        --batch_size 64 \
        --subdivisions 16 \
        --gpu 0
}

ssd() {
    python /raid/simrdwn/core/simrdwn.py \
        --framework ssd \
        --mode train \
        --outname inception_v2_cowc \
        --label_map_path /raid/simrdwn/data/xview.pbtxt \
        --tf_cfg_train_file /raid/simrdwn/configs/_orig/ssd_inception_v2_simrdwn.config \
        --train_tf_record /raid/simrdwn/data/xview.tfrecord \
        --max_batches 30000 \
        --batch_size 16 \
        --gpu 0
}

"$@"
