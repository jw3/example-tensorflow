installation from source
===

https://www.tensorflow.org/install/source

docker
===

docker pull tensorflow/tensorflow:nightly-devel-gpu-py3

docker run --runtime=nvidia -it -w /tensorflow -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" \
    tensorflow/tensorflow:nightly-devel-gpu-py3 bash


but that doesnt get custome CUDA versions, only the official supported version (9.0 in latest)

or
===

Install custom CUDA and compile from source on host


https://github.com/bazelbuild/bazel/releases/download/0.21.0/bazel-0.21.0-installer-linux-x86_64.sh

`bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package`
- remove `--config=cuda` if not using CUDA

`./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg`

`pip install /tmp/tensorflow_pkg/*.whl`


### requirements:
- bazel 0.15.0
- pip: distutils
