### yolo via simrdwn

I have distilled the yolo training process for xview data into something straight forward (I think).

From scratch, from this directory, with these scripts, just for xview 18 `small_car`

1. python yolo_from_xview.py /data/train_images/ /data/xView_train.geojson -c 18
2. ./rewrite_class.sh
3. ./build_simrdwn

train with the command provided by (3), something like
`nvidia-docker run -d --name simrdwn_train -v $PWD:/raid simrdwn:train /raid/train.sh`

You will see man iterations of

```
Batch Num: 25918 / 30000
25918: 0.666785, 0.598179 avg, 0.000100 rate, 6.682340 seconds, 1658752 images
Loaded: 0.000130 seconds
Batch Num: 25919 / 30000
Region Avg IOU: 0.491327, Class: 1.000000, Obj: 0.367586, No Obj: 0.001777, Avg Recall: 0.526316,  count: 19
Region Avg IOU: 0.753546, Class: 1.000000, Obj: 0.360975, No Obj: 0.001953, Avg Recall: 1.000000,  count: 4
Region Avg IOU: 0.521300, Class: 1.000000, Obj: 0.259653, No Obj: 0.001712, Avg Recall: 0.545455,  count: 11
Region Avg IOU: 0.460492, Class: 1.000000, Obj: 0.236518, No Obj: 0.002219, Avg Recall: 0.458333,  count: 24
Region Avg IOU: 0.632915, Class: 1.000000, Obj: 0.275407, No Obj: 0.001583, Avg Recall: 1.000000,  count: 6
Region Avg IOU: 0.401253, Class: 1.000000, Obj: 0.267250, No Obj: 0.001914, Avg Recall: 0.400000,  count: 10
Region Avg IOU: 0.584713, Class: 1.000000, Obj: 0.256432, No Obj: 0.002101, Avg Recall: 0.666667,  count: 3
Region Avg IOU: 0.621623, Class: 1.000000, Obj: 0.482377, No Obj: 0.001159, Avg Recall: 1.000000,  count: 3
Region Avg IOU: 0.544773, Class: 1.000000, Obj: 0.299269, No Obj: 0.002008, Avg Recall: 0.571429,  count: 7
Region Avg IOU: 0.462302, Class: 1.000000, Obj: 0.126225, No Obj: 0.001966, Avg Recall: 0.461538,  count: 13
Region Avg IOU: 0.531724, Class: 1.000000, Obj: 0.636916, No Obj: 0.001620, Avg Recall: 1.000000,  count: 1
Region Avg IOU: 0.448713, Class: 1.000000, Obj: 0.212568, No Obj: 0.002187, Avg Recall: 0.458333,  count: 24
Region Avg IOU: 0.672977, Class: 1.000000, Obj: 0.314061, No Obj: 0.002007, Avg Recall: 0.947368,  count: 19
Region Avg IOU: 0.492697, Class: 1.000000, Obj: 0.264764, No Obj: 0.001827, Avg Recall: 0.636364,  count: 11
Region Avg IOU: 0.329678, Class: 1.000000, Obj: 0.238931, No Obj: 0.001086, Avg Recall: 0.000000,  count: 5
Region Avg IOU: 0.345780, Class: 1.000000, Obj: 0.103434, No Obj: 0.001800, Avg Recall: 0.142857,  count: 7
Region Avg IOU: 0.325542, Class: 1.000000, Obj: 0.240831, No Obj: 0.002172, Avg Recall: 0.194444,  count: 36
Region Avg IOU: 0.571721, Class: 1.000000, Obj: 0.278781, No Obj: 0.001695, Avg Recall: 0.800000,  count: 5
Region Avg IOU: 0.453695, Class: 1.000000, Obj: 0.165323, No Obj: 0.002085, Avg Recall: 0.500000,  count: 18
Region Avg IOU: 0.505694, Class: 1.000000, Obj: 0.211362, No Obj: 0.002246, Avg Recall: 0.514286,  count: 35
Region Avg IOU: 0.590093, Class: 1.000000, Obj: 0.295269, No Obj: 0.002395, Avg Recall: 0.857143,  count: 21
Region Avg IOU: 0.578225, Class: 1.000000, Obj: 0.391345, No Obj: 0.001444, Avg Recall: 0.750000,  count: 4
Region Avg IOU: 0.461878, Class: 1.000000, Obj: 0.227462, No Obj: 0.002926, Avg Recall: 0.531250,  count: 64
Region Avg IOU: 0.771214, Class: 1.000000, Obj: 0.456792, No Obj: 0.001616, Avg Recall: 1.000000,  count: 1
Region Avg IOU: 0.380890, Class: 1.000000, Obj: 0.209506, No Obj: 0.001474, Avg Recall: 0.222222,  count: 9
Region Avg IOU: 0.580550, Class: 1.000000, Obj: 0.313905, No Obj: 0.001906, Avg Recall: 0.750000,  count: 4
Region Avg IOU: 0.446581, Class: 1.000000, Obj: 0.286102, No Obj: 0.002190, Avg Recall: 0.538462,  count: 13
Region Avg IOU: 0.474542, Class: 1.000000, Obj: 0.177982, No Obj: 0.001795, Avg Recall: 0.583333,  count: 12
Region Avg IOU: 0.493808, Class: 1.000000, Obj: 0.332142, No Obj: 0.002118, Avg Recall: 0.714286,  count: 7
Region Avg IOU: 0.497783, Class: 1.000000, Obj: 0.177512, No Obj: 0.001027, Avg Recall: 0.600000,  count: 5
Region Avg IOU: 0.438349, Class: 1.000000, Obj: 0.284358, No Obj: 0.002042, Avg Recall: 0.428571,  count: 14
Region Avg IOU: 0.546726, Class: 1.000000, Obj: 0.280005, No Obj: 0.002004, Avg Recall: 0.656250,  count: 32
Region Avg IOU: 0.554709, Class: 1.000000, Obj: 0.272547, No Obj: 0.002465, Avg Recall: 0.750000,  count: 24
Region Avg IOU: 0.522666, Class: 1.000000, Obj: 0.244105, No Obj: 0.002352, Avg Recall: 0.714286,  count: 14
Region Avg IOU: 0.302965, Class: 1.000000, Obj: 0.113253, No Obj: 0.001954, Avg Recall: 0.000000,  count: 2
Region Avg IOU: 0.590350, Class: 1.000000, Obj: 0.340184, No Obj: 0.001862, Avg Recall: 0.714286,  count: 7
Region Avg IOU: 0.416393, Class: 1.000000, Obj: 0.248653, No Obj: 0.001913, Avg Recall: 0.500000,  count: 4
```

The function for the output is [here](https://github.com/pjreddie/darknet/blob/master/src/region_layer.c#L320),
 
- Region Avg IOU: `avg_iou/count`
- Class: `avg_cat/class_count`
- Obj: `avg_obj/count`
- No Obj: `avg_anyobj/(l.w*l.h*l.n*l.batch)`
- Avg Recall: `recall/count`
- count: `count`

Havent come across a UI for this yet, though they do have a [utility that will plot the loss](https://github.com/CosmiQ/simrdwn/blob/master/core/yolt_plot_loss.py).
