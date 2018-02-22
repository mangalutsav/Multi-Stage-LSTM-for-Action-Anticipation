import os
import fnmatch
import argparse

import make_frames

parser = argparse.ArgumentParser(description='converting videos to frames')

parser.add_argument(
    "--input-dir",
    metavar="<path>",
    required=True,
    type=str,
    help="base directory of classes")

parser.add_argument(
    "--output-dir",
    metavar="<path>",
    required=True,
    type=str,
    help="output base dir")

parser.add_argument(
    "--format",
    metavar="<path>",
    default='jpg',
    choices=['jpg', 'png', 'webp'],
    type=str,
    help="output image format")

args = parser.parse_args()

def mkfp(*dirs):
    return os.path.join(args.input_dir, *dirs)

total_frames = 0
for cat in sorted(os.listdir(args.input_dir)):
    print(cat)
    cat_odir = os.path.join(args.output_dir, cat)
    if not os.path.isdir(cat_odir):
        os.makedirs(cat_odir)

    for vid in fnmatch.filter(os.listdir(mkfp(cat)), "*.avi"):
        vpath = mkfp(cat, vid)

        odir = os.path.join(cat_odir, vid)
        if not os.path.isdir(odir):
            os.mkdir(odir)

        print("Decoding '%s'" % vpath)
        total_frames += make_frames.cv2_dump_frames(vpath, odir, args.format, 94)

print("Total frames decoded: %d" % total_frames)



