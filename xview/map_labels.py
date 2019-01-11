import argparse
import csv

# python xview/map_labels.py kitti/xview-labels.txt > xview-labels.pbtxt

parser = argparse.ArgumentParser()
parser.add_argument("dict", help="Path to dict file' ")

args = parser.parse_args()

labels = {}
with open(args.dict) as f:
    for row in csv.reader(f):
        labels[int(row[0].split(":")[0])] = row[0].split(":")[1]

for idx, l in enumerate(labels):
    print("item {{\n  id: {}\n  name: '{}'\n}}".format(idx+1, labels[l]))
