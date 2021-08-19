#!/usr/bin/env python

import itertools
import random

import numpy as np

from _logger import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 屏蔽TF日志输出

logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_ELEMENT = ['Ce', 'O', 'C']

input_dir = os.path.join(current_dir, "input")
output_dir = os.path.join(current_dir, "output")


class FileManager:
    """
    TODO
    """
    pass


class CoorTailor:
    """
    TODO
    """
    pass


class Model:
    """
    TODO
    """
    pass


def __generator(filename: str):
    """
    The coordinates __generator for reading the input/output. <helper func>

    :param filename:        the filename of input/output
    :return:                coordinate array
    """
    with open(filename, "r") as f:
        data = f.readlines()
    coor = [[float(_) for _ in line.split()[:3]] for line in data[9:47]]
    return np.array(coor)


def read_dir(dirname: str):
    """
    read the input/output coordinates

    :param dirname:         the name of the directory storing the POSCAR/CONTCAR file
    :return:                coordinates array
    """
    if dirname.endswith("input"):
        prefix = "POSCAR"
    else:
        prefix = "CONTCAR"
    for file in range(1, 101):
        for _ in range(1, 3):
            filename = os.path.join(dirname, f'{prefix}_{_}-{file}')
            yield __generator(filename)


def data_ztrans(coor, boundary: float, num: int):
    """
    Put the Slab+Mol translate along the Z-axis. (Data Improver)

    :param coor:              Slab+Mol coor ndarray
    :param boundary:          max-value of the translate
    :param num:               the num of part split between the [0,boundary]
    :return:                  the translate coor (len = 2*num*ori_len)
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


def pbc_apply(input_arr, output_arr):
    """
    Handling the periodic-boundary-condition

    :param input_arr:           input-data array
    :param output_arr:          output-data array
    :return:                    data_input, data_output after pbc handle
    """
    data_input_arr = input_arr.copy()
    data_output_arr = output_arr.copy()
    data_output_arr = np.where((data_input_arr - data_output_arr) > 0.5, data_output_arr + 1, data_output_arr)
    data_output_arr = np.where((data_input_arr - data_output_arr) < -0.5, data_output_arr - 1, data_output_arr)

    return data_input_arr, data_output_arr


def __remove_repeat(findit_ij):
    """
    Removing the repeat mesh for which (coordinates < 0 or coordinates > 1) <helper func>

    :param findit_ij:           input and output merge mesh (may including the repeat item)
    :return:                    mesh after removing the repeat
    """
    findit = [(mesh_x, mesh_y) for mesh_x, mesh_y in zip(findit_ij[0], findit_ij[1])]
    findit_set = set(findit)
    findit_arr = np.array(list(findit_set))
    try:
        findit = (findit_arr[:, 0], findit_arr[:, 1])
    except IndexError:
        findit = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    return findit


def run_tailor_xy(data_input_arr, data_output_arr, repeat_unit: int, flag: str = "n"):
    """
    tailor_xy coordinates in (0, 1) area. (if failed print warning)

    :param data_input_arr:              data_input array
    :param data_output_arr:             data_output array
    :param repeat_unit:                 slab-supercell (e.g. 2 represents the 2x2 slab)
    :param flag::                       falg determing handling the (<0 or >1) case
    :return:                            data_input_arr, data_output_arr after tailoring the xy coordinates
    """
    trans_ = [_ for _ in range(repeat_unit)]
    trans = list(itertools.product(trans_, trans_, [0]))

    for fileno, (i_, j_) in enumerate(zip(data_input_arr, data_output_arr)):
        finditi = np.where((i_ < 0)) if flag == "n" else np.where((i_ > 1))
        finditj = np.where((j_ < 0)) if flag == "n" else np.where((j_ > 1))
        findit_ij = (np.concatenate([finditi[0], finditj[0]]), np.concatenate([finditi[1], finditj[1]]))
        findit = __remove_repeat(findit_ij)
        i_news = []
        j_news = []
        for i_new, j_new in zip(i_[findit], j_[findit]):
            count = 0
            ori_i_new, ori_j_new = i_new, j_new
            while True:
                i_new = i_new + 1 if flag == "n" else i_new - 1
                j_new = j_new + 1 if flag == "n" else j_new - 1
                count += 1
                if 0 <= i_new <= 1 and 0 <= j_new <= 1:
                    i_news.append(i_new)
                    j_news.append(j_new)
                    break
                if count >= 10:
                    i_news.append(ori_i_new)
                    j_news.append(ori_j_new)
                    logger.warning(
                        f"i_new = {ori_i_new:8.6f} j_new = {ori_j_new:8.6f} index = {trans[(fileno + 1) % 4]}")
                    break
        i_[findit], j_[findit] = i_news, j_news

    return data_input_arr, data_output_arr


def tailor_xy(input_arr, output_arr, repeat_unit: int):
    """
    tailor xy coordinates for (<0 or >1) case

    :param input_arr:                 data_input array
    :param output_arr:                data_output array
    :param repeat_unit:               slab-supercell (e.g. 2 represents the 2x2 slab)
    :return:                          data_input_arr, data_output_arr after tailoring the xy coordinates
    """
    data_input_arr = input_arr.copy()
    data_output_arr = output_arr.copy()

    data_input_arr, data_output_arr = run_tailor_xy(data_input_arr, data_output_arr, repeat_unit, "n")  # xy < 0 case
    data_input_arr, data_output_arr = run_tailor_xy(data_input_arr, data_output_arr, repeat_unit, "p")  # xy > 1 case

    return data_input_arr, data_output_arr


def expand_xy(input_arr, output_arr, repeat_unit: int):
    """
    For supercell, translate the left-bottom region into other area (Data Improver)

    :param input_arr:           data_input array
    :param output_arr:          data_output array
    :param repeat_unit:         slab-supercell (e.g. 2 represents the 2x2 slab)
    :return:                    data_input_arr, data_output_arr after expanding the xy coordinates
    """
    trans_coor_i_iner, trans_coor_o_iner = [], []
    for coor_i, coor_o in zip(input_arr, output_arr):
        trans = [_ for _ in range(repeat_unit)]
        coor_trans_i, coor_trans_o = [], []

        for item in itertools.product(trans, trans, [0]):
            coor_t_i = coor_i + np.array(item) / repeat_unit
            coor_t_o = coor_o + np.array(item) / repeat_unit

            coor_t_i = np.where(coor_t_i > 1, coor_t_i - 1, coor_t_i)
            coor_t_o = np.where(coor_t_o > 1, coor_t_o - 1, coor_t_o)

            coor_trans_i.append(coor_t_i)
            coor_trans_o.append(coor_t_o)

        trans_coor_i_iner += coor_trans_i
        trans_coor_o_iner += coor_trans_o

    return np.array(trans_coor_i_iner), np.array(trans_coor_o_iner)


def k_fold_validation(model_iner, n_split_iner: int, data_input_arr, data_output_arr):
    """
    K fold validation method for the Model test.

    :param model_iner:                      Keras.model
    :param n_split_iner:                    how many fold
    :param data_input_arr:                  data_input array 
    :param data_output_arr:                 data_output array
    :return:                                avg_mae, ave loss
    """
    from sklearn.model_selection import KFold

    avg_mae_iner = 0
    avg_loss_iner = 0

    def tailor_atom_order(train_input_iner, train_output_iner):

        shuffle_train_input, shuffle_train_output = [], []
        for i, j in zip(train_input_iner, train_output_iner):
            k1 = list(range(12))  # _Ce atom
            k2 = list(range(12, 36))  # _O atom
            random.shuffle(k1)
            random.shuffle(k2)
            k = k1 + k2 + [36, 37]  # _CO molecule
            i = i[k]
            j = j[k]
            i = i.reshape(38 * 3)
            j = j.reshape(38 * 3)
            shuffle_train_input.append(i)
            shuffle_train_output.append(j)

        train_input_iner = np.array(shuffle_train_input)
        train_output_iner = np.array(shuffle_train_output)

        return train_input_iner, train_output_iner

    for train_index, test_index in KFold(n_split_iner).split(data_input_arr):
        train_input, test_input = data_input_arr[train_index], data_input_arr[test_index]
        train_output, test_output = data_output_arr[train_index], data_output_arr[test_index]

        train_input, train_output = tailor_atom_order(train_input, train_output)

        test_input = test_input.reshape((test_input.shape[0], 38 * 3))
        test_output = test_output.reshape((test_output.shape[0], 38 * 3))

        model_iner.fit(train_input, train_output, epochs=30, batch_size=2, validation_split=0.1)
        scores = model_iner.evaluate(test_input, test_output)

        avg_loss_iner += scores[0]
        avg_mae_iner += scores[1]

    return avg_mae_iner, avg_loss_iner


if __name__ == "__main__":

    logger.info("Load the structure information.")
    input_coor = np.array([coor for coor in read_dir(input_dir)])
    output_coor = np.array([coor for coor in read_dir(output_dir)])

    logger.info("Apply the PBC and tailor the x-y coordinates.")
    repeat = 2  # supercell (2x2)
    trans_coor_i, trans_coor_o = expand_xy(input_coor, output_coor, repeat)
    input_coor, output_coor = pbc_apply(trans_coor_i, trans_coor_o)
    input_coor, output_coor = tailor_xy(input_coor, output_coor, repeat)

    logger.info("Expand the data in z-direction.")
    input_coor = data_ztrans(input_coor, 0.2, 2)
    output_coor = data_ztrans(output_coor, 0.2, 2)
    logger.info(f"Data Shape: {input_coor.shape}")

    logger.info("Shuffle the data.")
    data_input, data_output = input_coor, output_coor
    index = list(range(len(data_input)))
    random.shuffle(index)
    data_input = data_input[index]
    data_output = data_output[index]

    from keras import models
    from keras import layers

    model = models.Sequential()
    model.add(layers.Dense(1024, activation='relu', input_shape=(38 * 3,)))
    model.add(layers.Dense(114))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

    K_fold_flag = True

    if K_fold_flag:
        logger.info("Train and test the model_iner applying the K-fold validation method.")
        n_split = 5
        avg_mae, avg_loss = k_fold_validation(model, n_split, data_input, data_output)
        logger.info("K fold average mae: {}".format(avg_mae / n_split))
        logger.info("K fold average loss: {}".format(avg_loss / n_split))

# test=test_input[5]
# test=test.reshape((1,38*3))
# print(test)
# predict=model_iner.predict(test)[0]
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
