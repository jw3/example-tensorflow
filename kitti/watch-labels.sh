#!/bin/bash

echo -n "train: "
cat train/labels/* | wc -l

echo -n "val:   "
cat val/labels/* | wc -l

echo "======"
df -h | grep /dev/mapper
echo "======"

du -sh train/images
du -sh val/images
