#!/usr/bin/env bash

readonly root_path="${1:-${PWD}}"

mkdir -p ${root_path}/train/images \
         ${root_path}/train/labels \
         ${root_path}/val/images   \
         ${root_path}/val/labels
