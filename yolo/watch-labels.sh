#!/bin/bash

cat labels/* | wc -l
ls images | wc -l
echo "======"
df -h | grep /dev/mapper
echo "======"
du -sh labels
du -sh images
