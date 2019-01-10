#!/usr/bin/env bash

readonly root_path="${BASE:-${PWD}}"


clean() {
  rm ${root_path}/train/images/* \
     ${root_path}/train/labels/* \
     ${root_path}/val/images/*   \
     ${root_path}/val/labels/*
}

make() {
  mkdir -p ${root_path}/train/images \
           ${root_path}/train/labels \
           ${root_path}/val/images   \
           ${root_path}/val/labels
}

"$@"
