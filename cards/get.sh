#!/bin/bash

kitti() {
      mkdir -p train/images \
               train/labels \
               val/images   \
               val/labels   \
               all

    git clone https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10 /tmp/tfpc

    mv /tmp/tfpc/images/test/*.jpg all
    mv /tmp/tfpc/images/test/*.JPG all
    mv /tmp/tfpc/images/train/*.jpg all
    mv /tmp/tfpc/images/train/*.JPG all

    cat /tmp/tfpc/images/test_labels.csv  | head -n -1 >  all_labels.csv
    cat /tmp/tfpc/images/train_labels.csv | head -n -1 >> all_labels.csv

    rm -rf /tmp/tfpc
}

"$@"
