#!/usr/bin/env bash

readonly raid=$PWD
readonly weights=/usr/local/src/yolov2.weights

if [[ ! -f ${weights} ]]; then
  curl -o ${weights} https://pjreddie.com/media/files/yolov2.weights
fi

git clone https://github.com/CosmiQ/simrdwn ${raid}/simrdwn

cp ${weights} ${raid}/simrdwn/yolt/input_weights

cd ${raid}/simrdwn/docker

nvidia-docker build --no-cache --build-arg http_proxy --build-arg https_proxy -t simrdwn .

nvidia-docker run -d --name simrdwn_train -v ${raid}:/raid simrdwn

docker exec -w /raid/simrdwn/yolt simrdwn make
