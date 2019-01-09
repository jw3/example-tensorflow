NVIDIA Docker install
===


https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)

1. nvidia.github.io/nvidia-docker/
2. sudo apt-get install nvidia-docker2
3. sudo pkill -SIGHUP dockerd
4. docker run --runtime nvidia --rm -it --entrypoint bash nvcr.io/nvidia/digits:18.12-tensorflow
5. [test.py](test.py)


```
2019-01-09 19:59:31.615167: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:957] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-01-09 19:59:31.615698: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties: 
name: Tesla P100-PCIE-16GB major: 6 minor: 0 memoryClockRate(GHz): 1.3285
pciBusID: 0000:00:06.0
totalMemory: 15.90GiB freeMemory: 15.61GiB
2019-01-09 19:59:31.615738: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-01-09 19:59:31.991821: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-09 19:59:31.991886: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2019-01-09 19:59:31.991900: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2019-01-09 19:59:31.992298: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/device:GPU:0 with 15129 MB memory) -> physical GPU (device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:00:06.0, compute capability: 6.0)
True
2019-01-09 19:59:31.995257: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-01-09 19:59:31.995300: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-09 19:59:31.995315: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2019-01-09 19:59:31.995324: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2019-01-09 19:59:31.995667: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 15129 MB memory) -> physical GPU (device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:00:06.0, compute capability: 6.0)
Hello, TensorFlow!
42
```
