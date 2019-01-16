#!/usr/bin/env bash

# https://github.com/CosmiQ/simrdwn#1b-create-tfrecord

readonly raiddir="${RAIDDIR:raid}"

python /${raiddir}/simrdwn/core/preprocess_tfrecords.py \
    --pbtxt_filename /${raiddir}/xview_yolo.pbtxt \
    --image_list_file /${raiddir}/training_list.txt \
    --outfile /${raiddir}/simrdwn/data/xview.tfrecord \
    --val_frac 0.0
