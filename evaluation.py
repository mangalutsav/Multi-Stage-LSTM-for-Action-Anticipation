from models import MS_LSTM
import numpy as np
import argparse
from from feature_generator import CustomDataGenerator


parser = argparse.ArgumentParser(description='Train multi-stage LSTM (MS-LSTM)')

parser.add_argument(
    "--action-aware",
    metavar="<path>",
    required=True,
    type=str,
    help="path to action-aware features")

parser.add_argument(
    "--context-aware",
    metavar="<path>",
    required=True,
    type=str,
    help="path to context-aware features")

parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="model you want to test")

parser.add_argument(
    "--classes",
    type=int,
    default=21,
    help="number of classes in target dataset")

parser.add_argument(
    "--temporal-length",
    default=50,
    type=int,
    help="number of frames representing each video")

parser.add_argument(
    "--cell",
    default=2048,
    type=int,
    help="number of hidden units in LSTM cells")

args = parser.parse_args()

model = MS_LSTM(INPUT_LEN=args.temporal_length,
                INPUT_DIM=4096,
                OUTPUT_LEN=args.classes,
                cells=args.cell)


model.load_weights(args.model)

validation_generator_obj = CustomDataGenerator(
    data_path_context=args.context_aware + '/val/',
    data_path_action=args.action_aware + '/val/',
    batch_size=1,
    temporal_length=args.temporal_length,
    N_C=args.classes)

validation_generator = validation_generator_obj.generator()

performance_w_avg = np.zeros((args.temporal_length,1))
performance_wo_avg = np.zeros((args.temporal_length,1))
anticipation = np.zeros((validation_generator_obj.data_size, args.temporal_length, 1))

y_test = np.zeros((validation_generator_obj.data_size,args.temporal_length, args.classes))
x_test_context = np.zeros((validation_generator_obj.data_size,args.temporal_length, 4096))
x_test_action = np.zeros((validation_generator_obj.data_size,args.temporal_length, 4096))

for index, item in enumerate(validation_generator):

    x_test_context = item[0]
    x_test_action = item[1]
    y_test[index] = item[-1]

gt = np.argmax(y_test, axis=2)

for t in range(1,args.temporal_length):

    x_context = np.zeros((validation_generator_obj.data_size, args.temporal_length, 4096))
    x_action = np.zeros((validation_generator_obj.data_size, args.temporal_length, 4096))

    x_context[:, :t, :] = x_test_context[:, :t, :]
    x_action[:, :t, :] = x_test_action[:, :t, :]

    out = model.evaluate([x_context, x_action], [y_test, y_test])

    pred = model.predict([x_context, x_action])
    prediction = pred[0]

    avg = np.mean(prediction[:, :t, :], axis=1)
    anticipation[:, t, 0] = np.argmax(avg, axis=1)

    correct = 0
    incorrect = 0

    for sample in range(validation_generator_obj.data_size):
        if anticipation[sample, t] == gt[sample, t]:
            correct += 1
        else:
            incorrect += 1

    performance_w_avg[t] = np.double(correct) / (correct + incorrect)

    performance_wo_avg[t] = out[-2]


for i in range(args.temporal_length):

    print('w/ Temporal Average Pooling: ' + str(performance_w_avg[i][0]) +
          ' -- wo/ Temporal Average Pooling: ' + str(performance_wo_avg[i][0]))

