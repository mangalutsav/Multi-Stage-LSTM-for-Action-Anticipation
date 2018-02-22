
from keras.layers import ZeroPadding2D, Conv2D, AveragePooling2D
from keras.layers.core import Flatten
from keras.applications.vgg16 import VGG16
from keras.layers import LSTM, Dense, TimeDistributed, merge, Activation, Dropout, Input
from keras.models import Model


def vgg_action(N_C, input_shape=(224, 224, 3)):

    input_tensor = Input(shape=input_shape)

    base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    base_model_conv5 = Model(base_model.layers[0].input, base_model.layers[17].output)

    x = base_model_conv5.output
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(1024, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = AveragePooling2D((14, 14), strides=(14, 14))(x)
    x = Flatten()(x)

    predictions = Dense(N_C, activation='softmax')(x)

    vgg = Model(inputs=base_model_conv5.input, outputs=predictions)

    return vgg


def vgg_context(N_C, input_shape=(224, 224, 3)):

    input_tensor = Input(shape=input_shape)
    base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(N_C, activation='softmax')(x)

    vgg = Model(inputs=base_model.input, outputs=predictions)

    return vgg


def MS_LSTM(INPUT_DIM, INPUT_LEN, OUTPUT_LEN, cells=2048):

    input_vs_sp = Input(shape=(INPUT_DIM, INPUT_LEN))
    input_vs_tp = Input(shape=(INPUT_DIM, INPUT_LEN))

    lstm_l1 = LSTM(cells, return_sequences=True)(input_vs_sp)
    do_lstm_l1 = Dropout(0.5)(lstm_l1)
    tdd_lstm_l1 = TimeDistributed(Dense(OUTPUT_LEN))(do_lstm_l1)
    act_lstm_l1 = Activation('softmax', name='stage1')(tdd_lstm_l1)
    pool_inp_l2 = merge([input_vs_tp, do_lstm_l1], mode='concat')

    lstm_l2 = LSTM(cells, return_sequences = True)(pool_inp_l2)
    do_lstm_l2 = Dropout(0.5)(lstm_l2)
    fc_pool_l2 = TimeDistributed(Dense(OUTPUT_LEN))(do_lstm_l2)
    act_lstm_l2 = Activation('softmax', name='stage2')(fc_pool_l2)

    MODEL = Model(input=[input_vs_sp, input_vs_tp],output=[act_lstm_l1, act_lstm_l2])

    return MODEL

