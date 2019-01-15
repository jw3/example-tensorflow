#!/usr/bin/env bash

readonly root_path="${BASE:-${PWD}}"


clean() {
  rm ${root_path}/images/* \
     ${root_path}/labels/*
}

make() {
  mkdir -p ${root_path}/images \
           ${root_path}/labels
}

"$@"
