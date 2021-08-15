#!/usr/bin/env python

import os
import random
import logging
import itertools
import numpy as np
from _logger import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 屏蔽TF日志输出

logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_ELEMENT = ['Ce', 'O', 'C']

current_dir = os.getcwd()
input_dir = os.path.join(current_dir, "input")
output_dir = os.path.join(current_dir, "output")


def generator(filename):
    with open(filename, "r") as f:
        data = f.readlines()
    coor = [[float(_) for _ in line.split()[:3]] for line in data[9:47]]
    return np.array(coor)


def read_dir(dirname):
    if dirname.endswith("input"):
        prefix = "POSCAR"
    else:
        prefix = "CONTCAR"
    for file in range(1, 101):
        for _ in range(1, 3):
            filename = os.path.join(dirname, f'{prefix}_{_}-{file}')
            yield generator(filename)


def data_ztrans(coor, boundary: float, num: int):
    """
    Put the Slab+Mol translate along the Z-axis. (Data Improver)
    :param coor: Slab+Mol coor ndarray
    :param boundary: max-value of the translate
    :param num: the num of part split between the [0,boundary]
    :return: the translate coor (len = 2*num*ori_len)
    """
    ori_coor = coor.copy()
    for _ in range(num):
        delta_z = (_ * boundary / num)
        coor_p = ori_coor.copy()
        coor_n = ori_coor.copy()
        coor_p = coor_p + np.array([0.0, 0.0, delta_z])
        coor_n = coor_n + np.array([0.0, 0.0, -delta_z])
        coor = np.append(coor, coor_p, axis=0)
        coor = np.append(coor, coor_n, axis=0)
    return coor


def PBC_apply(input_coor, output_coor):

    data_input = input_coor.copy()
    data_output = output_coor.copy()
    data_output = np.where((data_input - data_output) > 0.5, data_output + 1, data_output)
    data_output = np.where((data_input - data_output) < -0.5, data_output - 1, data_output)

    return data_input, data_output


def remove_repeat(findit_ij):

    findit = [(i, j) for i, j in zip(findit_ij[0], findit_ij[1])]
    findit_set = set(findit)
    findit_arr = np.array(list(findit_set))
    try:
        findit = (findit_arr[:, 0], findit_arr[:, 1])
    except IndexError:
        findit = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    return findit


def run_tailor_xy(data_input, data_output, repeat, flag = "n"):

    trans_ = [_ for _ in range(repeat)]
    trans = list(itertools.product(trans_, trans_, [0]))

    for index, (i, j) in enumerate(zip(data_input, data_output)):
        finditi = np.where((i<0)) if flag=="n" else np.where((i>1))
        finditj = np.where((j<0)) if flag=="n" else np.where((j>1))
        findit_ij = (np.concatenate([finditi[0], finditj[0]]), np.concatenate([finditi[1], finditj[1]]))
        findit = remove_repeat(findit_ij)
        i_news = []
        j_news = []
        for i_new, j_new in zip(i[findit], j[findit]):
            count = 0
            ori_i_new, ori_j_new = i_new, j_new
            while True:
                i_new = i_new + 1 if flag=="n" else i_new - 1
                j_new = j_new + 1 if flag=="n" else j_new - 1
                count += 1
                if ((i_new >= 0 and i_new <= 1 and j_new >= 0 and j_new <= 1)):
                    i_news.append(i_new)
                    j_news.append(j_new)
                    break
                if count>=10:
                    i_news.append(ori_i_new)
                    j_news.append(ori_j_new)
                    logger.warning(f"i_new = {ori_i_new:8.6f} j_new = {ori_j_new:8.6f} index = {trans[(index+1) % 4]}")
                    break
        i[findit], j[findit] = i_news, j_news

    return data_input, data_output


def tailor_xy(input_coor, output_coor, repeat):

    data_input = input_coor.copy()
    data_output = output_coor.copy()

    data_input, data_output = run_tailor_xy(data_input, data_output, repeat, "n")
    data_input, data_output = run_tailor_xy(data_input, data_output, repeat, "p")

    return data_input, data_output


def expand_xy(input_coor, output_coor, repeat):

    trans_coor_i, trans_coor_o = [], []
    for coor_i, coor_o in  zip(input_coor, output_coor):
        trans = [_ for _ in range(repeat)]
        coor_trans_i, coor_trans_o = [], []

        for item in itertools.product(trans,trans,[0]):
            coor_t_i = coor_i + np.array(item)/repeat
            coor_t_o = coor_o + np.array(item)/repeat

            coor_t_i = np.where(coor_t_i>1, coor_t_i-1, coor_t_i)
            coor_t_o = np.where(coor_t_o>1, coor_t_o-1, coor_t_o)

            coor_trans_i.append(coor_t_i)
            coor_trans_o.append(coor_t_o)

        trans_coor_i += coor_trans_i
        trans_coor_o += coor_trans_o

    return np.array(trans_coor_i), np.array(trans_coor_o)

if __name__ == "__main__":

    logger.info("Load the structure information.")
    input_coor = np.array([coor for coor in read_dir(input_dir)])
    output_coor = np.array([coor for coor in read_dir(output_dir)])

    logger.info("Apply the PBC and tailor the x-y coordinates.")
    repeat = 2 # supercell (2x2)
    trans_coor_i, trans_coor_o = expand_xy(input_coor, output_coor, repeat)
    input_coor, output_coor = PBC_apply(trans_coor_i, trans_coor_o)
    input_coor, output_coor = tailor_xy(input_coor, output_coor, repeat)

    logger.info("Expand the data in z-direction.")
    input_coor = data_ztrans(input_coor, 0.2, 2)
    output_coor = data_ztrans(output_coor, 0.2, 2)

    logger.info("Shuffle the data.")
    data_input, data_output = input_coor, output_coor
    index = list(range(len(data_input)))
    random.shuffle(index)
    data_input = data_input[index]
    data_output = data_output[index]

    from keras import models
    from keras import layers
    from sklearn.model_selection import KFold

    model = models.Sequential()
    model.add(layers.Dense(1024, activation='relu', input_shape=(38 * 3,)))
    model.add(layers.Dense(114))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

    n_split = 5
    avg_mae = 0
    avg_loss = 0

    logger.info("Train and test the model applying the K-fold validation method.")
    for train_index, test_index in KFold(n_split).split(data_input):

        train_input, test_input = data_input[train_index], data_input[test_index]
        train_output, test_output = data_output[train_index], data_output[test_index]

        shuffle_train_input = []
        shuffle_train_output = []
        for i, j in zip(train_input, train_output):
            k1 = list(range(12))
            k2 = list(range(12, 36))
            random.shuffle(k1)
            random.shuffle(k2)
            k = k1 + k2 + [36, 37]
            i = i[k]
            j = j[k]
            i = i.reshape(38 * 3)
            j = j.reshape(38 * 3)
            shuffle_train_input.append(i)
            shuffle_train_output.append(j)

        train_input = np.array(shuffle_train_input)
        train_output = np.array(shuffle_train_output)

        test_input = test_input.reshape((test_input.shape[0], 38 *3))
        test_output = test_output.reshape((test_output.shape[0], 38 * 3))

        history = model.fit(train_input, train_output, epochs=30, batch_size=2, validation_split=0.1)
        scores = model.evaluate(test_input, test_output)

        avg_loss += scores[0]
        avg_mae += scores[1]


    logger.info("K fold average mae: {}".format(avg_mae / n_split))
    logger.info("K fold average loss: {}".format(avg_loss / n_split))

# test=test_input[5]
# test=test.reshape((1,38*3))
# print(test)
# predict=model.predict(test)[0]
# true=test_output[5].reshape((38*3))
# print(predict)
# print(true)
# print(predict-true)

##### Figure #####
# from matplotlib import pyplot as plt
# acc = history.history['acc']
# loss = history.history['loss']
# val_acc = history.history['val_acc']
# val_loss = history.history['val_loss']
# epochs = range(1,len(acc)+1)

# plt.plot(epochs,acc,"ro")
# plt.plot(epochs,val_acc,'r')
# plt.plot(epochs,loss,"bo")
# plt.plot(epochs,val_loss,'b')
# plt.show()
