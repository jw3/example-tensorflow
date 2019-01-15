import argparse
import csv

# python xview/map_labels.py kitti/xview-labels.txt > xview-labels.pbtxt

parser = argparse.ArgumentParser()
parser.add_argument("dict", help="Path to dict file' ")
parser.add_argument("cat", help="Category: small|medium|large")
args = parser.parse_args()

labels = {}
classes = []
with open(args.dict) as f:
    for row in csv.reader(f):
        splits = row[0].split(":")
        labels[int(splits[0])] = splits[1]
        classes.append(splits[1])

with open('%s.txt' % args.cat) as f:
    cat = f.read().split('\n')
    err = [c for c in cat if c and c not in classes]

if err:
    print("invalid mappings:")
    print(err)
else:
    idx = 0
    for _, k in enumerate(labels):
        if labels[k] in cat:
            idx += 1
            print("item {{\n  id: {}\n  name: '{}'\n}}".format(idx, labels[k]))
