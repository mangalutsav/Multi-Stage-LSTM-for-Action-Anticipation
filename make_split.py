import argparse
import os.path
import json

parser = argparse.ArgumentParser(description='splitting dataset into train/val by creating symlinks to original frames')

parser.add_argument(
    "--split-dir",
    metavar="<path>",
    required=True,
    type=str,
    help="input lmdb path",
    default='data/splits')

parser.add_argument(
    "--data-dir",
    metavar="<path>",
    required=True,
    type=str,
    help="input lmdb path",
    default='data/jhmdb_dataset')

parser.add_argument(
    "--index",
    metavar="<n>",
    default=1,
    type=int,
    help="which split 1-3 to use")

parser.add_argument(
    "--output-dir",
    metavar="<path>",
    required=True,
    type=str,
    help="input lmdb path",
    default='data/frames/')

args = parser.parse_args()


classes = [ d for d in os.listdir(args.data_dir) ]
classes = filter(lambda d: os.path.isdir(os.path.join(args.data_dir, d)), classes)

counts = {}
extensions = ['jpg', 'jpeg', 'png', 'webp']

for cls in sorted(classes):
    print(cls)
    with open(os.path.join(args.split_dir, cls + '_test_split%d.txt' % args.index)) as f:
        lines = f.readlines()

    for l in lines:
        video_fn, split = l.split()

        if split == '1':
            cat = 'train'
        elif split == '2':
            cat = 'val'
        else:
            cat = 'dummy'

        src_dir = os.path.join(args.data_dir, cls, video_fn)
        img_files = list(filter(lambda x: os.path.splitext(x)[1][1:] in extensions,
                                os.listdir(src_dir)))

        if cat not in counts:
            counts[cat] = 0
        counts[cat] += len(img_files)

        if cat == 'dummy':
            continue

        target_dir = os.path.join(args.output_dir, cat, cls)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        for i,img in enumerate(sorted(img_files)):
            ext = os.path.splitext(img)[1]
            os.symlink(os.path.join(src_dir, img),
                       os.path.join(target_dir, video_fn + "_%05d%s" % (i, ext)))

nt = counts['train']
nv = counts['val']
nd = counts.get('dummy', 0)

print("Split: %d/%d/%d = %d" % (nt, nv, nd, nt + nv + nd))





