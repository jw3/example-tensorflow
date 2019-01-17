Deep Learning experiments
===

### Frameworks
- [xView](xview)
- [YOLOv3 with SIMRDWN](yolo)
- [KITTI](kitti)
- [DetectNet](detectnet)

### Datasets
- [Playing Cards](cards)




### xView Dataset

- http://xviewdataset.org
- https://github.com/DIUx-xView
- https://challenge.xviewdataset.org/baseline
- https://challenge.xviewdataset.org/tutorial
- https://github.com/ultralytics/xview-docker
- https://github.com/PlatformStories/train-cnn-classifier
- https://medium.com/picterra/the-xview-dataset-and-baseline-results-5ab4a1d0f47f

### Segmenting xView data

Sometimes json is easier to work with when partitioned

`jq '[.features[] | select(.properties.image_id == "100.tif")]' xView_train.geojson`

`for f in *.tif; do echo $f; jq --arg f "$f" '[.features[] | select(.properties.image_id == $f)]' ../xView_train.geojson > $f.geojson; done`


### nvidia docker
- install nvidia-docker
  - https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)#prerequisites
- install cuda
  - https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
  - https://developer.nvidia.com/cuda-downloads
- register for nvidia registry
  - https://ngc.nvidia.com/registry
- run containers with `nvidia-docker`
  - `nvidia-docker run --rm nvidia/cuda nvidia-smi`
- debug issues with
  - `nvidia-container-cli -k -d /dev/tty info`


