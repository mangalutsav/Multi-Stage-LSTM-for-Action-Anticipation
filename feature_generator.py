import numpy as np
from random import shuffle
import glob


class CustomDataGenerator(object):

    def __init__(self,
                 data_path_action,
                 data_path_context,
                 batch_size=32,
                 temporal_length=50,
                 N_C=21):

        self.batch_size = batch_size
        self.data_path_action = data_path_action
        self.data_path_context = data_path_context
        self.temporal_length = temporal_length
        self.classes = N_C

        self.features_action = glob.glob(self.data_path_action + '/feature_*.npy')
        self.features_action.sort()
        print(self.data_path_context)
        self.features_context = glob.glob(self.data_path_context + '/feature_*.npy')
        self.features_context.sort()

        self.labels = glob.glob(self.data_path_context + '/label_*.npy')
        self.labels.sort()

        self.pairs = zip(self.features_context, self.features_action, self.labels)
        shuffle(self.pairs)

        self.data_size = len(self.pairs)
        self.current = 0

    def generate(self):

        while True:

            if self.current < self.data_size - self.batch_size:

                X_c = np.zeros((self.batch_size,self.temporal_length, 4096))
                X_a = np.zeros((self.batch_size, self.temporal_length, 4096))
                y = np.zeros((self.batch_size, self.temporal_length, self.classes))

                cnt = 0
                for pair in range(self.current,self.current+self.batch_size):

                    X_c[cnt] = np.load(self.pairs[pair][0])
                    X_a[cnt] = np.load(self.pairs[pair][1])
                    y[cnt] = np.load(self.pairs[pair][2])

                    cnt += 1

                yield X_c, X_a, y, y

                self.current += self.batch_size

            else:

                self.current = 0
                shuffle(self.pairs)

                X_c = np.zeros((self.batch_size, self.temporal_length, 4096))
                X_a = np.zeros((self.batch_size, self.temporal_length, 4096))
                y = np.zeros((self.batch_size, self.temporal_length, self.classes))

                cnt = 0
                for pair in range(self.current, self.current + self.batch_size):

                    X_c[cnt] = np.load(self.pairs[pair][0])
                    X_a[cnt] = np.load(self.pairs[pair][1])
                    y[cnt] = np.load(self.pairs[pair][2])

                    cnt += 1

                yield X_c, X_a, y, y

                self.current += self.batch_size


'''
how to use:
train_generator = CustomDataGenerator(*params1).generator()
validation_generator = CustomDataGenerator(*params2).generator()

model.fit_generator(generator = training_generator,
                    validation_data = validation_generator,
                    nb_epoch = 50,
                    verbose = 1)

'''