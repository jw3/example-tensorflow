#!/usr/bin/env bash

# https://github.com/CosmiQ/simrdwn#1b-create-tfrecord

python /raid/simrdwn/core/preprocess_tfrecords.py \
    --pbtxt_filename /raid/xview_yolo.pbtxt \
    --image_list_file /raid/training_list.txt \
    --outfile /raid/simrdwn/data/xview.tfrecord \
    --val_frac 0.0
