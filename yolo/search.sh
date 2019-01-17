#!/usr/bin/env bash

# https://github.com/CosmiQ/simrdwn#3-test

readonly usage='search.sh model_path weights'
readonly model_path="${1?${usage}}"
readonly weight_file="${2?${usage}}"

if [[ ! -d ${model_path} ]]; then
  echo "invalid model path"
  exit 1
fi

if [[ ! -f ${model_path}/${weight_file} ]]; then
  echo "invalid weight file"
  exit 1
fi

python /raid/simrdwn/core/simrdwn.py \
	--framework yolt \
	--mode valid \
	--outname dense_cowc \
	--yolt_object_labels_str small_car \
	--train_model_path ${model_path} \
	--weight_file ${weight_file} \
	--yolt_cfg_file yolt.cfg \
	--valid_testims_dir cowc/Utah_AGRC  \
	--use_tfrecords 0 \
	--min_retain_prob=0.15 \
	--keep_valid_slices 0 \
	--slice_overlap 0.1 \
	--slice_sizes_str 544 \
	--valid_slice_sep __ \
	--plot_thresh_str 0.2 \
	--valid_make_legend_and_title 0 \
	--edge_buffer_valid 1 \
	--valid_box_rescale_frac 1 \
	--alpha_scaling 1 \
	--show_labels 0
