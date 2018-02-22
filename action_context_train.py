from models import vgg_action, vgg_context
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import argparse
from keras.callbacks import ModelCheckpoint
import os

parser = argparse.ArgumentParser(description='tune vgg16 network on new dataset')

parser.add_argument(
    "--data-dir",
    metavar="<path>",
    required=True,
    type=str,
    help="train/val data base directory")

parser.add_argument(
    "--classes",
    type=int,
    default=21,
    help="number of classes in target dataset")

parser.add_argument(
    "--model-type",
    choices=['action_aware', 'context_aware'],
    default='action_aware',
    help="action-aware model or context-aware model")

parser.add_argument(
    "--epochs",
    default=128,
    type=int,
    help="number of epochs")

parser.add_argument(
    "--samples-per-epoch",
    default=None,
    type=int,
    help="samples per epoch, default=all")

parser.add_argument(
    "--save-model",
    metavar="<prefix>",
    default=None,
    type=str,
    help="save model at the end of each epoch")

parser.add_argument(
    "--save-best-only",
    default=False,
    action='store_true',
    help="only save model if it is the best so far")

parser.add_argument(
    "--num-val-samples",
    default=None,
    type=int,
    help="number of validation samples to use (default=all)")

parser.add_argument(
    "--fixed-width",
    default=224,
    type=int,
    help="crop or pad input images to ensure given width")

parser.add_argument(
    "--seed",
    default=10,
    type=int,
    help="random seed")

parser.add_argument(
    "--workers",
    default=1,
    type=int,
    help="number of data preprocessing worker threads to launch")

parser.add_argument(
    "--learning-rate",
    default=0.001,
    type=float,
    help="initial/fixed learning rate")

parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="batch size")


args = parser.parse_args()

correct_model = False

if args.model_type == 'action_aware':
    model = vgg_action(args.classes, input_shape=(args.fixed_width,args.fixed_width,3))
    correct_model = True
elif args.model_type == 'context_aware':
    model = vgg_context(args.classes, input_shape=(args.fixed_width, args.fixed_width, 3))
    correct_model = True
else:
    print("Wrong model type name!")

if correct_model:

    test_datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=True,
        featurewise_std_normalization=True)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
            os.path.join(args.data_dir , 'train/'),
            target_size=(args.fixed_width, args.fixed_width),
            batch_size=args.batch_size,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(args.data_dir, 'val/'),
            target_size=(args.fixed_width, args.fixed_width),
            batch_size=args.batch_size,
            class_mode='categorical')


    sgd = SGD(lr=args.learning_rate, decay=0.005, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    callbacks = []

    if args.save_model:
        callbacks.append(ModelCheckpoint(args.save_model,
                                         verbose=0,
                                         monitor='val_acc',
                                         save_best_only=args.save_best_only))

    samples_per_epoch = args.samples_per_epoch or train_generator.samples // args.batch_size
    samples_per_epoch -= (samples_per_epoch % args.batch_size)
    num_val_samples = args.num_val_samples or validation_generator.samples // args.batch_size


    print("Starting to train...")
    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples // args.batch_size,
                        verbose=1,
                        callbacks=callbacks,
                        epochs=args.epochs,
                        workers=args.workers,
                        shuffle=True,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.samples // args.batch_size)

    if args.model_type == 'action_aware':
        model.save_weights('data/model_weights/action_aware_vgg16_final.h5')
    else:
        model.save_weights('data/model_weights/context_aware_vgg16_final.h5')



