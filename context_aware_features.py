import argparse
import glob
import json
import cv2
import numpy as np
import os
from models import vgg_context
from keras import backend as K

parser = argparse.ArgumentParser(description='extracting context-aware features')

parser.add_argument(
    "--data-dir",
    metavar="<path>",
    required=True,
    type=str,
    default='data/jhmdb_dataset/',
    help="path to video files")

parser.add_argument(
    "--classes",
    type=int,
    default=21,
    help="number of classes in target dataset")

parser.add_argument(
    "--model",
    required=True,
    type=str,
    default='model_weights/context_aware_vgg16_final.h5',
    help="path to the trained model of context_aware")

parser.add_argument(
    "--split-dir",
    type=str,
    default='model_weights/context_aware_vgg16_final.h5',
    help="path to the dataset splits directory")

parser.add_argument(
    "--temporal-length",
    default=50,
    type=int,
    elp="number of frames representing each video")

parser.add_argument(
    "--split",
    default='1',
    type=str,
    elp="the split")

parser.add_argument(
    "--output",
    default='data/context_features/',
    type=str,
    help="path to the directory of features")

parser.add_argument(
    "--fixed-width",
    default=224,
    type=int,
    help="crop or pad input images to ensure given width")


args = parser.parse_args()

model = vgg_context(args.classes, input_shape=(args.fixed_width,args.fixed_width,3))
model.load_weights(args.model)

context_aware = K.function([model.layers[0].input, K.learning_phase()], [model.layers[22].output])

data_mean = json.load(open('config/mean.json', 'rb'))

classes = [ d for d in os.listdir(args.data_dir) ]
classes = filter(lambda d: os.path.isdir(os.path.join(args.data_dir, d)), classes)

for cls in sorted(classes):
    with open(os.path.join(args.split_dir, cls + '_test_split%d.txt' % args.split)) as f:
        lines = f.readlines()

    for l in lines:
        video_fn, split = l.split()

        if split == '1':
            cat = 'train'
        elif split == '2':
            cat = 'val'
        else:
            cat = 'dummy'

        print (video_fn, split, cls)

        feature = np.zeros((args.temporal_length,4096))
        label = np.zeros((args.temporal_length,len(classes)))

        vid_path = os.path.join(args.data_dir, cls)
        cap = cv2.VideoCapture(os.path.join(vid_path, video_fn))

        for fr in range(args.temporal_length):

            try:
                frame, ret = cap.read()
                if ret:
                    f2 = cv2.resize(frame, (args.fixed_width,args.fixed_width), interpolation=cv2.INTER_CUBIC)
                    f2_arr = np.array(f2, dtype=np.double)

                    f2_arr[:, :, 0] -= data_mean[0]
                    f2_arr[:, :, 1] -= data_mean[1]
                    f2_arr[:, :, 2] -= data_mean[2]

                    in_ = np.expand_dims(f2_arr, axis=0)

                    feature[fr] = np.array(context_aware([in_, 0]))[0][0]
                    label[fr][classes.index(cls)] = 1

            except:
                pass

        np.save(os.path.join(args.output, cat+'/feature_'+video_fn.split('.')[0]+'.npy'), feature)
        np.save(os.path.join(args.output, cat+'/label_' + video_fn.split('.')[0] + '.npy'), label)

        print ("[Done] " + video_fn + " " + cls)