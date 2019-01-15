"""
Copyright 2018 Defense Innovation Unit Experimental
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from PIL import Image
import tensorflow as tf
import io
import glob
from tqdm import tqdm
import numpy as np
import logging
import argparse
import os
import json
import wv_util as wv
import tfr_util as tfr
import aug_util as aug
import csv

"""
  A script that processes xView imagery. 
  Args:
      image_folder: A folder path to the directory storing xView .tif files
        ie ("xView_data/")

      json_filepath: A file path to the GEOJSON ground truth file
        ie ("xView_gt.geojson")

      test_percent (-t): The percentage of input images to use for test set

      suffix (-s): The suffix for output TFRecord files.  Default suffix 't1' will output
        xview_train_t1.record and xview_test_t1.record

      augment (-a): A boolean value of whether or not to use augmentation

  Outputs:
    Writes two files to the current directory containing training and test data in
        TFRecord format ('xview_train_SUFFIX.record' and 'xview_test_SUFFIX.record')
"""


def get_images_from_filename_array(coords,chips,classes,folder_names,res=(250,250)):
    """
    Gathers and chips all images within a given folder at a given resolution.

    Args:
        coords: an array of bounding box coordinates
        chips: an array of filenames that each coord/class belongs to.
        classes: an array of classes for each bounding box
        folder_names: a list of folder names containing images
        res: an (X,Y) tuple where (X,Y) are (width,height) of each chip respectively

    Output:
        images, boxes, classes arrays containing chipped images, bounding boxes, and classes, respectively.
    """

    images =[]
    boxes = []
    clses = []

    k = 0
    bi = 0

    for folder in folder_names:
        fnames = glob.glob(folder + "*.tif")
        fnames.sort()
        for fname in tqdm(fnames):
            #Needs to be "X.tif" ie ("5.tif")
            name = fname.split("\\")[-1]
            arr = wv.get_image(fname)

            img,box,cls = wv.chip_image(arr,coords[chips==name],classes[chips==name],res)

            for im in img:
                images.append(im)
            for b in box:
                boxes.append(b)
            for c in cls:
                clses.append(cls)
            k = k + 1

    return images, boxes, clses

def shuffle_images_and_boxes_classes(im,box,cls):
    """
    Shuffles images, boxes, and classes, while keeping relative matching indices

    Args:
        im: an array of images
        box: an array of bounding box coordinates ([xmin,ymin,xmax,ymax])
        cls: an array of classes

    Output:
        Shuffle image, boxes, and classes arrays, respectively
    """
    assert len(im) == len(box)
    assert len(box) == len(cls)

    perm = np.random.permutation(len(im))
    out_b = {}
    out_c = {}

    k = 0
    for ind in perm:
        out_b[k] = box[ind]
        out_c[k] = cls[ind]
        k = k + 1
    return im[perm], out_b, out_c


def write_yolo_labels(img, boxes, class_num, labels):
    """
    Converts a single image with respective boxes into a TFExample.  Multiple TFExamples make up a TFRecord.

    Args:
        img: an image array
        boxes: an array of bounding boxes for the given image
        class_num: an array of class numbers for each bouding box

    Output:
        A TFExample containing encoded image data, scaled bounding boxes with classes, and other metadata.
    """
    encoded = tfr.convertToJpeg(img)

    sw = img.shape[0]
    sh = img.shape[1]

    yolo_text = []
    for ind,box in enumerate(boxes):
        if not class_num[ind] in labels:
            continue

        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])

        if xmin + ymin + xmax + ymax > 0:
            dw = 1./sw
            dh = 1./sh
            xmid = (xmin + xmax)/2.0
            ymid = (ymin + ymax)/2.0
            w0 = xmax - xmin
            h0 = ymax - ymin
            x = xmid*dw
            y = ymid*dh
            w = w0*dw
            h = h0*dh

            clazz = labels[int(class_num[ind])]
            yolo_text.append("{} {} {} {} {}".format(clazz, x, y, w, h))

    # example = tf.train.Example(features=tf.train.Features(feature={
    #     'image/height': int64_feature(height),
    #     'image/width': int64_feature(width),
    #     'image/encoded': bytes_feature(encoded),
    #     'image/format': bytes_feature('jpeg'.encode('utf8')),
    #     'image/object/bbox/xmin': float_list_feature(xmin),
    #     'image/object/bbox/xmax': float_list_feature(xmax),
    #     'image/object/bbox/ymin': float_list_feature(ymin),
    #     'image/object/bbox/ymax': float_list_feature(ymax),
    #     'image/object/class/label': int64_list_feature(classes),
    # }))

    return '\n'.join(yolo_text)

'''
Datasets
_multires: multiple resolutions. Currently [(500,500),(400,400),(300,300),(200,200)]
_aug: Augmented dataset
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder", help="Path to folder containing image chips (ie 'Image_Chips/' ")
    parser.add_argument("json_filepath", help="Filepath to GEOJSON coordinate file")
    parser.add_argument("-t", "--test_percent", type=float, default=0.333,
                        help="Percent to split into test (ie .25 = test set is 25% total)")
    parser.add_argument("-a","--augment", type=bool, default=False,
                        help="A boolean value whether or not to use augmentation")
    parser.add_argument("-c","--classes", type=str, default='',
                        help="A list of class ids to include; empty for all; eg. 1,2,3 or 1")
    parser.add_argument("--class_size", type=str, default='',
                        help="Class size to select: small | medium | large")
    parser.add_argument("-x","--xview_labels", type=str, default='xview-yolo-labels.txt',
                        help="Path to xview class labels file")
    parser.add_argument("-s","--scale", type=float, default=0,
                        help="Resize chips and boxes to the given scale")
    parser.add_argument("-r","--resolution", type=int, default=544,
                        help="Chip resolution, will be used for each dim")
    parser.add_argument("-i","--images", type=str, default='',
                        help="A list of image ids to include, empty for all; eg. 1,2,3 or 1")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    lookup = {}
    with open('xview-labels.txt') as f:
        for row in csv.reader(f):
            splits = row[0].split(":")
            lookup[splits[1]] = splits[0]

    filterz = []
    if args.class_size:
        with open('%s.txt' % args.class_size) as f:
            for row in csv.reader(f):
                splits = row[0].split(":")
                filterz.append(lookup[splits[1]])

    filterc = args.classes.split(',') if args.classes else []
    labels = {}
    with open(args.xview_labels) as f:
        for row in csv.reader(f):
            splits = row[0].split(":")
            if not filterz or splits[0] in filterz:
                if not filterc or splits[0] in filterc:
                    labels[int(splits[0])] = splits[1]

    #resolutions should be largest -> smallest.  We take the number of chips in the largest resolution and make
    #sure all future resolutions have less than 1.5times that number of images to prevent chip size imbalance.
    #res = [(500,500),(400,400),(300,300),(200,200)]
    res = [(args.resolution, args.resolution)]

    AUGMENT = args.augment
    SAVE_IMAGES = False
    images = {}
    boxes = {}
    train_chips = 0
    test_chips = 0
    skip_chips = 0
    images_list = []

    #Parameters
    max_chips_per_res = 100000

    coords,chips,classes = wv.get_labels(args.json_filepath)

    for res_ind, it in enumerate(res):
        tot_box = 0
        logging.info("Res: %s" % str(it))
        ind_chips = 0

        fnames = glob.glob(args.image_folder + "*.tif")

        if args.images:
            fimages = args.images.split(',')
            image_filter = ['%s%s.tif' % (args.image_folder, f) for f in fimages]
            fnames = [f for f in fnames if f in image_filter]

        fnames.sort()

        for fname in tqdm(fnames):
            #Needs to be "X.tif", ie ("5.tif")
            #Be careful!! Depending on OS you may need to change from '/' to '\\'.  Use '/' for UNIX and '\\' for windows
            name = fname.split("/")[-1]
            arr = wv.get_image(fname)

            im,box,classes_final = wv.chip_image(arr,coords[chips==name],classes[chips==name],it)

            if args.scale:
                im, box = aug.resize(im, box, classes_final, args.scale,labels)

            #Shuffle images & boxes all at once. Comment out the line below if you don't want to shuffle images
            im,box,classes_final = shuffle_images_and_boxes_classes(im,box,classes_final)
            split_ind = int(im.shape[0] * args.test_percent)

            for idx, image in enumerate(im):
                istest = idx < split_ind

                tf_example = write_yolo_labels(image,box[idx],classes_final[idx],labels)

                #Check to make sure that the TF_Example has valid bounding boxes.
                #If there are no valid bounding boxes, then don't save the image to the TFRecord.
                #float_list_value = tf_example.features.feature['image/object/bbox/xmin'].float_list.value

                if (ind_chips < max_chips_per_res and tf_example):
                    tot_box+=tf_example.count('\n')

                    if idx < split_ind:
                        test_chips+=1
                    else:
                        train_chips += 1

                    writer = open("labels/%s.txt" % (str(ind_chips).rjust(6, '0')), "w")
                    img_file = 'images/%s.png'%(str(ind_chips).rjust(6, '0'))
                    Image.fromarray(image).save(img_file)
                    images_list.append(os.path.join(os.getcwd(), img_file))

                    writer.write(tf_example)

                    ind_chips +=1


                    #Make augmentation probability proportional to chip size.  Lower chip size = less chance.
                    #This makes the chip-size imbalance less severe.
                    prob = np.random.randint(0,np.max(res))
                    #for 200x200: p(augment) = 200/500 ; for 300x300: p(augment) = 300/500 ...

                    if AUGMENT and prob < it[0]:

                        for extra in range(3):
                            center = np.array([int(image.shape[0]/2),int(image.shape[1]/2)])
                            deg = np.random.randint(-10,10)
                            #deg = np.random.normal()*30
                            newimg = aug.salt_and_pepper(aug.gaussian_blur(image))

                            #.3 probability for each of shifting vs rotating vs shift(rotate(image))
                            p = np.random.randint(0,3)
                            if p == 0:
                                newimg,nb = aug.shift_image(newimg,box[idx])
                            elif p == 1:
                                newimg,nb = aug.rotate_image_and_boxes(newimg,deg,center,box[idx])
                            elif p == 2:
                                newimg,nb = aug.rotate_image_and_boxes(newimg,deg,center,box[idx])
                                newimg,nb = aug.shift_image(newimg,nb)


                            newimg = (newimg).astype(np.uint8)

                            if idx%1000 == 0 and SAVE_IMAGES:
                                Image.fromarray(newimg).save('process/img_%s_%s_%s.png'%(name,extra,it[0]))

                            if len(nb) > 0:
                                tf_example = write_yolo_labels(newimg,nb,classes_final[idx],labels)
                                writer.write(tf_example.SerializeToString())

                                #Don't count augmented chips for chip indices
                                if idx < split_ind:
                                    test_chips += 1
                                else:
                                    train_chips+=1
                            else:
                                if SAVE_IMAGES:
                                    aug.draw_bboxes(newimg,nb).save('process/img_nobox_%s_%s_%s.png'%(name,extra,it[0]))

                    writer.close()
                else:
                    skip_chips += 1

        if res_ind == 0:
            max_chips_per_res = int(ind_chips * 1.5)
            logging.info("Max chips per resolution: %s " % max_chips_per_res)

        logging.info("Tot Box: %d" % tot_box)
        logging.info("Chips: %d" % ind_chips)
        logging.info("Skipped Chips: %d" % skip_chips)

    with open('training_list.txt', 'w') as f:
        f.write('\n'.join(images_list))

    logging.info("saved: %d train chips" % train_chips)
    logging.info("saved: %d test chips" % test_chips)
