installation from source
===

`bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package`
- remove `--config=cuda` if not using CUDA

`./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg`

`pip install /tmp/tensorflow_pkg/*.whl`


### requirements:
- bazel

### reference
- https://www.tensorflow.org/install/source
