import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("name", help="Index name, eg 000001")
args = parser.parse_args()

name = args.name.rjust(6, '0')

img = Image.open('images/%s.png' % name)
im = np.array(img, dtype=np.uint8)
size = im.shape

fig,ax = plt.subplots(1)
ax.imshow(im)

with open('labels/%s.txt' % name) as f:
    for row in csv.reader(f):
        split = row[0].split()
        box = [float(split[1]), float(split[2]), float(split[3]), float(split[4])]
        x, y, w, h = box
        dw = 1./size[0]
        dh = 1./size[1]

        w0 = w/dw
        h0 = h/dh
        xmid = x/dw
        ymid = y/dh

        x0, x1 = xmid - w0/2., xmid + w0/2.
        y0, y1 = ymid - h0/2., ymid + h0/2.

        rect = patches.Rectangle((x0, y0),x1 - x0, y1 - y0 ,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)


plt.show()
