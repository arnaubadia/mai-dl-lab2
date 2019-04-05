import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
.. module:: WindPrediction

WindPrediction
*************

:Description: WindPrediction

:Authors: bejar


:Version:

:Created on: 06/09/2017 9:47

"""


import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
import os
import tensorflow as tf
import json
import argparse
from time import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

__author__ = 'bejar'

def load_config_file(nfile, abspath=False):
    """
    Read the configuration from a json file

    :param abspath:
    :param nfile:
    :return:
    """
    ext = '.json' if 'json' not in nfile else ''
    pre = '' if abspath else './'
    fp = open(pre + nfile + ext, 'r')

    s = ''

    for l in fp:
        s += l

    return json.loads(s)


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def my_generate_dataset(config, data_path='madrid_dataset/madrid.h5', val_ratio=0.1, test_ratio=0.1):
    f = h5py.File(data_path, 'r')
    # these are stations
    # stations = list(f.keys())
    # select station '28079004' (Plaza España)
    station = '28079004'
    # select particle 'NO_2' aka Nitrogen Dioxide
    particle = 2
    # values is 1D array with hourly measurement of the particle (2001-2018)
    values = np.array(f[station]['block0_values'][:, particle])
    # Impute NaNs with the previous value
    nan_indices = np.argwhere(np.isnan(values))
    for i in nan_indices:
        values[i] = values[i-1]

    # standardize values
    scaler = StandardScaler()
    values = scaler.fit_transform(values.reshape(-1, 1)).reshape(-1)
    # windowed_values is an array which is obtained from applying a rolling window of size window_size
    # we will use window_size - 1, and the last one will be the target value
    window_size = config['lag']
    windowed_values = rolling_window(values, window_size)
    # if lag window is too large we can subsample (once every four)
    windowed_values = windowed_values[::4]

    # train-test split
    test_split_point = int(len(windowed_values)*(1-test_ratio))
    val_split_point = int(len(windowed_values)*(1-test_ratio-val_ratio))
    train = windowed_values[:val_split_point]
    val = windowed_values[val_split_point:test_split_point]
    test = windowed_values[test_split_point:]
    x_train = train[:, :-1].reshape(-1, window_size-1, 1)
    y_train = train[:, -1].reshape(-1, 1)
    x_val = val[:, :-1].reshape(-1, window_size-1, 1)
    y_val = val[:, -1].reshape(-1, 1)
    x_test = test[:, :-1].reshape(-1, window_size-1, 1)
    y_test = test[:, -1].reshape(-1, 1)

    return x_train, y_train, x_val, y_val, x_test, y_test




def architecture(neurons, drop, nlayers, activation, activation_r, rnntype, impl=1):
    """
    RNN architecture

    :return:
    """
    RNN = LSTM if rnntype == 'LSTM' else GRU
    model = Sequential()
    if nlayers == 1:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r))
    else:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                      return_sequences=True))
        for i in range(1, nlayers - 1):
            model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                          activation=activation, recurrent_activation=activation_r, return_sequences=True))
        model.add(RNN(neurons, recurrent_dropout=drop, activation=activation,
                      recurrent_activation=activation_r, implementation=impl))

    model.add(Dense(1))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config', help='Experiment configuration')
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true',
                        default=False)
    parser.add_argument('--best', help="Save weights best in test", action='store_true', default=False)
    parser.add_argument('--tboard', help="Save log for tensorboard", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2

    config = load_config_file(args.config)
    ############################################
    # Data

    ahead = config['data']['ahead']

    if args.verbose:
        print('-----------------------------------------------------------------------------')
        print('Steps Ahead = %d ' % ahead)

    # Modify conveniently with the path for your data
    aq_data_path = './'

    #train_x, train_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, data_path=aq_data_path)
    train_x, train_y, val_x, val_y, test_x, test_y = my_generate_dataset(config['data'],
                            data_path='madrid_dataset/madrid.h5', test_ratio=0.1, val_ratio=0.1)

    print(train_x.shape)
    print(train_y.shape)
    print(val_x.shape)
    print(val_y.shape)
    print(test_x.shape)
    print(test_y.shape)


    ############################################
    # Model

    model = architecture(neurons=config['arch']['neurons'],
                         drop=config['arch']['drop'],
                         nlayers=config['arch']['nlayers'],
                         activation=config['arch']['activation'],
                         activation_r=config['arch']['activation_r'], rnntype=config['arch']['rnn'], impl=impl)
    if args.verbose:
        model.summary()
        print('lag: ', config['data']['lag'],
              '/Neurons: ', config['arch']['neurons'],
              '/Layers: ', config['arch']['nlayers'],
              '/Activations:', config['arch']['activation'], config['arch']['activation_r'])
        print('Tr:', train_x.shape, train_y.shape, 'Ts:', test_x.shape, test_y.shape)
        print()

    ############################################
    # Training

    optimizer = config['training']['optimizer']

    if optimizer == 'rmsprop':
        if 'lrate' in config['training']:
            optimizer = RMSprop(lr=config['training']['lrate'])
        else:
            optimizer = RMSprop(lr=0.001)

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    cbacks = []

    if args.tboard:
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        cbacks.append(tensorboard)

    modfile = './model.h5'
    modelCheckpoint = ModelCheckpoint(filepath=modfile, monitor='val_loss', verbose=0, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')
    callbacks = [modelCheckpoint, earlyStopping]

    start_time = time()

    history = model.fit(train_x, train_y, batch_size=config['training']['batch'],
              epochs=config['training']['epochs'],
              validation_data=(val_x, val_y),
              verbose=True, callbacks=callbacks)


    print('Total training time = {0:.2f} seconds'.format(time() - start_time))

    ############################################
    # Results

    model = load_model(modfile)

    score = model.evaluate(test_x, test_y, batch_size=config['training']['batch'], verbose=0)

    print()
    print('MSE test= ', score)
    print('MSE test persistence =', mean_squared_error(test_y[ahead:], test_y[0:-ahead]))
    test_yp = model.predict(test_x, batch_size=config['training']['batch'], verbose=0)
    r2test = r2_score(test_y, test_yp)
    r2pers = r2_score(test_y[ahead:, 0], test_y[0:-ahead, 0])
    print('R2 test= ', r2test)
    print('R2 test persistence =', r2pers)

    #Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig('loss_plot.pdf')


"""
    plt.title('Test results')
    plt.plot(test_y)
    plt.plot(test_yp)
    plt.legend(['true','predicted'], loc='upper left')
    plt.show()
"""

"""
    resfile = open('result-%s.txt' % config['data']['datanames'][0], 'a')
    resfile.write('DATAS= %d, LAG= %d, AHEAD= %d, RNN= %s, NLAY= %d, NNEUR= %d, DROP= %3.2f, ACT= %s, RACT= %s, '
                  'OPT= %s, R2Test = %3.5f, R2pers = %3.5f\n' %
                  (config['data']['dataset'],
                   config['data']['lag'],
                   config['data']['ahead'],
                   config['arch']['rnn'],
                   config['arch']['nlayers'],
                   config['arch']['neurons'],
                   config['arch']['drop'],
                   config['arch']['activation'],
                   config['arch']['activation_r'],
                   config['training']['optimizer'],
                   r2test, r2pers
                   ))
    resfile.close()
"""
    # Deletes the model file
    #try:
    #    os.remove(modfile)
    #except OSError:
    #    pass
