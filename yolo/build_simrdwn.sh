#!/usr/bin/env bash

readonly raid=$PWD
readonly weights=/usr/local/src/yolov2.weights

if [[ ! -f ${weights} ]]; then
  curl -o ${weights} https://pjreddie.com/media/files/yolov2.weights
fi

if [[ ! -d ${raid}/simrdwn ]]; then
  git clone https://github.com/CosmiQ/simrdwn ${raid}/simrdwn
fi

cp ${weights} ${raid}/simrdwn/yolt/input_weights

cd ${raid}/simrdwn/docker

nvidia-docker build --no-cache --build-arg http_proxy --build-arg https_proxy -t simrdwn .

nvidia-docker run -d --name simrdwn_train -v ${raid}:/raid simrdwn bash -c '''while true; do sleep 1000; done'''

docker exec simrdwn_train git config --global user.email 'foo@bar.baz'
docker exec -w /opt/tensorflow-models simrdwn_train bash -c '''git am < /raid/0001-fix-exporter.patch'''

docker exec -w /raid/simrdwn/yolt simrdwn_train make -j8

docker kill simrdwn_train
docker commit --change='CMD []' simrdwn_train simrdwn:train
docker rm simrdwn_train

echo "image simrdwn:train is ready!  run it with:"
echo "nvidia-docker run -d --name simrdwn_train -v $PWD:/raid simrdwn:train /raid/train.sh"
