import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("type", help="Type: train or val")
parser.add_argument("name", help="Index name, eg 1 or 212")
parser.add_argument("-f", "--format", type=str, default='png',
                    help="Image format by extension")

args = parser.parse_args()

name = args.name.rjust(6, '0')
fmt = args.format

img = Image.open('%s/images/%s.%s' % (args.type, name, fmt))
im = np.array(img, dtype=np.uint8)
size = im.shape

fig,ax = plt.subplots(1)
ax.imshow(im)

total=0
invalid=0

with open('%s/labels/%s.txt' % (args.type, name)) as f:
    for row in csv.reader(f):
        split = row[0].split()
        box = [float(split[4]), float(split[5]), float(split[6]), float(split[7])]
        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]

        if w < 50 or h < 50 or w > 400 or h > 400:
            print('%s w=%s h=%s' % (split[0], w, h))
            invalid += 1

        rect = patches.Rectangle((x, y), w, h ,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        total += 1

print('%s / %s were wrong size for detectnet' % (invalid, total))

plt.show()
