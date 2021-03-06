DetectNet
===

1. mind the feature size; 50x50 to 400x400
2. only train a single class at a time, but include `dontcare`
3. be sure that the image sizes are correct across the network
    - `param_str`s
    - image sizes
4. lower the image sizes in `detectnet_groundtruth_param` for both train and val
5. adjust the stride also with `#4` for smaller features
6. use `Adam` solver
7. learning rate around 1e-05
8. Play around with mean subraction
9. Kitti formatted data
    - `class 0 0 0 xmin ymin xmax ymax 0 0 0 0 0 0 0`


### DIGITS configuration


#### with minio

Have to adjust the boto library configuration to use the non-subdomain s3 urls

Mount this as `/etc/boto.cfg`

``` 
[s3]
calling_format=boto.s3.connection.OrdinaryCallingFormat
```

#### image/label formats

- https://github.com/NVIDIA/DIGITS/issues/992
- https://github.com/NVIDIA/DIGITS/blob/master/docs/ImageFolderFormat.md
  - PNG, JPEG, BMP and PPM
