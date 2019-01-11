import os
import csv
import argparse
import shutil
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--augment", type=bool, default=False,
                    help="A boolean value whether or not to use augmentation")
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

tot = 0
train = 0
val = 0

for i, k in enumerate(kitti_text):
    name = k.split('.')[0]
    tot += 1

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

if args.augment:
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2)
    ], random_order=True)

    for i, k in enumerate(kitti_text):
        name = k.split('.')[0]
        val += 1
        tot += 1

        idxname = str(tot).rjust(6, '0')
        img = np.array(Image.open('all/%s' % k))
        with open('train/labels/%s.txt' % idxname, 'w') as f:
            f.write('\n'.join(kitti_text[k]))

        images_aug = seq.augment_image(img)
        Image.fromarray(images_aug).save('train/images/%s.jpg' % idxname)
