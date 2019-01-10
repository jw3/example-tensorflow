import os
import csv
import argparse
import shutil

parser = argparse.ArgumentParser()
args = parser.parse_args()

kitti_text = dict()
with open('all_labels.csv') as f:
    for splits in csv.reader(f):
        fname = splits[0]

        clazz = splits[3]
        x0, y0, x1, y1 = splits[4:]

        if fname not in kitti_text:
            kitti_text[fname] = set()
        kitti_text[fname].add("{} 0 0 0 {} {} {} {} 0 0 0 0 0 0 0".format(clazz, x0, y0, x1, y1))


train = 0
val = 0

for i, k in enumerate(kitti_text):
    name = k.split('.')[0]

    if train * .1 > val:
        mode = 'val'
        val += 1
    else:
        mode = 'train'
        train += 1

    idxname = str(i + 1).rjust(6, '0')
    shutil.copy2('all/%s' % k, '%s/images/%s.jpg' % (mode, idxname))
    with open('%s/labels/%s.txt' % (mode, idxname), 'w') as f:
        f.write('\n'.join(kitti_text[k]))
