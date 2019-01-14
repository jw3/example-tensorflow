import os
import csv
import argparse
import shutil
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np


def bbc_to_kitti_text(bbc):
    txt = []
    for b in bbc:
        txt.append("{} 0 0 0 {} {} {} {} 0 0 0 0 0 0 0".format(b[0], b[1], b[2], b[3], b[4]))
    return '\n'.join(txt)


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
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-3, 3),
        shear=(-8, 8)
    )
], random_order=True)


def augment(phase, img, boxes, idx, debug):
    seq_det = seq.to_deterministic()

    iaboxes = []
    for b in boxes:
        iaboxes.append(ia.BoundingBox(x1=b[1], y1=b[2], x2=b[3], y2=b[4]))
    bbs = ia.BoundingBoxesOnImage(iaboxes, shape=img.shape)

    img_aug = seq_det.augment_images([img])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    if debug:
        img_aug = bbs.draw_on_image(img_aug, thickness=1, color=[255, 0, 0])
        img_aug = bbs_aug.draw_on_image(img_aug, thickness=3, color=[0, 255, 0])

    idxstr = str(idx).rjust(6, '0')
    Image.fromarray(img_aug).save('%s/images/%s.jpg' % (phase, idxstr))

    with open('%s/labels/%s.txt' % (phase, idxstr), 'w') as f:
        kitti_text_aug = []
        for i in range(len(bbs_aug.bounding_boxes)):
            bb = bbs_aug.bounding_boxes[i]
            kitti_text_aug.append(
                bbc_to_kitti_text([[boxes[i][0], bb.x1, bb.y1, bb.x2, bb.y2]])
            )
        f.write('\n'.join(kitti_text_aug))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--augment", type=int, default=0,
                        help="Number of augmentation batches to generate")
    parser.add_argument("-D", "--debug", action='store_true',
                        help="Enable debug mode (draw bboxes)")
    args = parser.parse_args()

    kitti_text = dict()
    with open('all_labels.csv') as f:
        for splits in csv.reader(f):
            fname = splits[0]

            clazz = splits[3]
            x0, y0, x1, y1 = splits[4:]

            if fname not in kitti_text:
                kitti_text[fname] = []
            kitti_text[fname].append([clazz, int(x0), int(y0), int(x1), int(y1)])

    tot = 0
    train = 0
    val = 0

    for i, k in enumerate(kitti_text):
        name = k.split('.')[0]
        tot += 1
        mode = ''

        isval = train * .1 > val
        if isval:
            mode = 'val'
            val += 1
        else:
            mode = 'train'
            train += 1

        idxname = str(i + 1).rjust(6, '0')
        img_path = '%s/images/%s.jpg' % (mode, idxname)
        shutil.copy2('all/%s' % k, img_path)
        with open('%s/labels/%s.txt' % (mode, idxname), 'w') as f:
            f.write(bbc_to_kitti_text(kitti_text[k]))

        if args.augment:
            for _ in range(args.augment):
                tot += 1
                img = np.array(Image.open(img_path))
                augment(mode, img, kitti_text[k], tot, args.debug)
