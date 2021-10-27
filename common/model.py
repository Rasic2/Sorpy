import os
import copy
import json
import random
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from common.logger import logger, root_dir
from common.base import Lattice
from common.utils import plot_clsss_wrap as plot_wrap
from common.operate import Operator as op
from common.io_file import POSCAR
from common.manager import DirManager


class Model:
    """
    Machine Learning using the Keras framework.
    1.  Prepare the data <data normalization and expand>
    2.  Train the Model using the prepared data
    """

    def __init__(self, model, train_input, train_output, normalization=None, expand=None):

        self.model = model
        self.train_input = train_input
        self.train_output = train_output

        # Model data normalization
        NormalizeFunc = {"mcoord": self.normalize_mcoord,
                         "vcoord": self.normalize_vcoord}
        self.normalization = normalization
        if self.normalization in NormalizeFunc.keys():
            NormalizeFunc[self.normalization]()

        # Model data expand
        ExpandFunc = {"expand_z": self.expand_z}
        self.expand = expand
        if isinstance(self.expand, dict):
            expand_func = list(self.expand.keys())[0]
            if expand_func in ExpandFunc.keys():
                ExpandFunc[expand_func](**self.expand[expand_func])

    def expand_z(self, boundary: float = 0.2, expand_num: int = 2, ignore_index=None):
        """data expand <mcoord format and only expand z>"""
        shape = self.train_input.shape
        index = list(range(shape[1]))
        frac_index = list(set(index).difference(set(ignore_index)))
        data_input, data_output = np.copy(self.train_input), np.copy(self.train_output)

        def expand_inner(coor):
            ori_coor = coor.copy()
            for _ in range(expand_num):
                delta_z = ((_ + 1) * boundary / expand_num)
                coor_p, coor_n = ori_coor.copy(), ori_coor.copy()
                increment_p, increment_n = np.zeros((shape[1], shape[2])), np.zeros((shape[1], shape[2]))
                increment_p[frac_index], increment_n[frac_index] = [0.0, 0.0, delta_z], [0.0, 0.0, -delta_z]

                coor_p = coor_p + increment_p
                coor_n = coor_n + increment_n

                coor = np.append(coor, coor_p, axis=0)
                coor = np.append(coor, coor_n, axis=0)

            return coor

        setattr(self, "train_input", expand_inner(data_input))
        setattr(self, "train_output", expand_inner(data_output))

    def normalize_mcoord(self):
        """data normalization <mcoord format>"""

        data_input, data_output = np.copy(self.train_input), np.copy(self.train_output)

        def normalize_inner(data):
            data[:, 37, 2] = np.where(data[:, 37, 2] >= 0, data[:, 37, 2], 360 + data[:, 37, 2])
            data[:, 37, :] = data[:, 37, :] / [1, 180, 360] - [1.142, 0, 0]
            return data

        setattr(self, "train_input", normalize_inner(data_input))
        setattr(self, "train_output", normalize_inner(data_output))

    def normalize_vcoord(self):
        """data normalization <vcoord format>"""

        data_input, data_output = np.copy(self.train_input), np.copy(self.train_output)

        setattr(self, "train_input", op.normalize_vcoord(data_input))
        setattr(self, "train_output", op.normalize_vcoord(data_output))

    @staticmethod
    def decode_mcoord(coords, lattice: Lattice = None):
        ori_coords = copy.deepcopy(coords)
        anchor_cart = np.dot(ori_coords[:, 36, :], lattice.matrix)
        inter_coords = ori_coords[:, [37], :] * [1, 180, 360] + [1.142, 0, 0]

        r, theta, phi = inter_coords[:, :, 0], np.deg2rad(inter_coords[:, :, 1]), np.deg2rad(inter_coords[:, :, 2])
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        xyz_cart = np.concatenate((x, y, z), axis=1)
        inter_cart = anchor_cart + xyz_cart
        inter_frac = np.dot(inter_cart, lattice.inverse)

        trans_coords = copy.deepcopy(coords)
        trans_coords[:, 37, :] = inter_frac

        return trans_coords

    @staticmethod
    def decode_vcoord(dm, test_output, orders, lattice, rotate=None):

        test_output[:, -1, :] = test_output[:, -1, :] * [1, 180, 360] + [1.142, 0, 0]

        # handle Ce-O
        Ce_anchor_cart = np.dot(test_output[:, 0, :], lattice.matrix)
        Ce_xyz_cart = (test_output[:, 1:-2, :] * 2 - 1) * 2.356
        if rotate is not None:
            Ce_xyz_cart = np.dot(Ce_xyz_cart, np.linalg.inv(rotate))
        test_output[:, 1:-2, :] = np.dot((Ce_anchor_cart.reshape((50, 1, 3)) + Ce_xyz_cart), lattice.inverse)

        # handle C-O
        C_anchor_cart = np.dot(test_output[:, -2, :], lattice.matrix)
        r, theta, phi = test_output[:, -1, 0], np.deg2rad(test_output[:, -1, 1]), np.deg2rad(test_output[:, -1, 2])
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        C_xyz_cart = np.concatenate((x.reshape(50, 1, 1), y.reshape(50, 1, 1), z.reshape(50, 1, 1)), axis=2)
        test_output[:, -1, :] = np.dot((C_anchor_cart.reshape((50, 3)) + C_xyz_cart.reshape((50, 3))),
                                      lattice.inverse)

        outputs = []

        for file, order, coord in zip(dm.all_files, orders, test_output):
            order += dm.mol_index  # orders: mol_slab, need to plus the mol_CO order
            s1 = file.structure
            output_frac = np.copy(s1.frac_coords)
            output_frac[order] = coord
            outputs.append(output_frac)
        return np.array(outputs)

    def __call__(self, method, mname=None, **kargs):

        TrainFunc = {"hold out": self.train_hold_out,
                     "Kfold": self.train_kfold_validation}

        logger.info(f"Data Shape: {self.train_input.shape}")
        logger.info("Shuffle the data.")

        index = list(range(len(self.train_input)))
        random.shuffle(index)

        setattr(self, "train_input", self.train_input[index])
        setattr(self, "train_output", self.train_output[index])

        if method in TrainFunc.keys():
            return TrainFunc[method](mname=mname, **kargs)
        else:
            raise KeyError(
                f"'{method}' method can not to train the ML model. <optional arguments: ('hold out', 'Kfold')>")

    @staticmethod
    def split_data(model_data, data):
        shape = [item.shape[1] for item in model_data]  # Dense layer
        start = 0
        data_new = []
        for item in shape:
            data_new.append(data[:, start:start + item])
            start = item
        return data_new

    def train_hold_out(self, mname=None, percent=0.8, epochs=50):
        """
        Hold-out method for the model train.
        :param mname:                   save model name
        :param percent:                 train:test_set percent
        :return:                        mae and loss
        """
        import math
        logger.info(f"Train and test_set the model applying the hold-out method. <train:test_set = "
                    f"{math.ceil(percent * 100)}:{math.ceil((1 - percent) * 100)}>")

        shape = self.train_input.shape
        samples = shape[0]
        train_count = math.ceil(samples * percent)
        test_count = samples - train_count

        train_input_arr, train_output_arr = self.train_input[:train_count], self.train_output[:train_count]
        train_input_arr = train_input_arr.reshape((train_count, shape[1] * shape[2]))
        train_output_arr = train_output_arr.reshape((train_count, shape[1] * shape[2]))

        test_input_arr, test_output_arr = self.train_input[train_count:], self.train_output[train_count:]
        test_input_arr = test_input_arr.reshape((test_count, shape[1] * shape[2]))
        test_output_arr = test_output_arr.reshape((test_count, shape[1] * shape[2]))

        train_inputs = Model.split_data(self.model.inputs, train_input_arr)
        train_outputs = Model.split_data(self.model.outputs, train_output_arr)
        test_inputs = Model.split_data(self.model.inputs, test_input_arr)
        test_outputs = Model.split_data(self.model.outputs, test_output_arr)

        history = self.model.fit(train_inputs, train_outputs, epochs=epochs, batch_size=2, validation_split=0.1)
        scores = self.model.evaluate(test_inputs, test_outputs)

        logger.info(f"mae = {scores[1]}")
        logger.info(f"loss = {scores[0]}")

        if mname is not None:
            self.model.save(mname)

        return history

    def train_kfold_validation(self, mname=None, num_of_split: int = 5):
        """
        K fold validation method for the Model train.
        :param mname:                           save model name
        :param num_of_split:                    how many fold
        :return:                                avg_mae, ave loss
        """
        logger.info("Train and test_set the model applying the K-fold validation method.")
        from sklearn.model_selection import KFold

        avg_mae = 0
        avg_loss = 0
        history = None

        for train_index, test_index in KFold(num_of_split).split(self.train_input):
            train_input, test_input = self.train_input[train_index], self.train_input[test_index]
            train_output, test_output = self.train_output[train_index], self.train_output[test_index]

            train_input = train_input.reshape((train_input.shape[0], 38 * 3))
            train_output = train_output.reshape((train_output.shape[0], 38 * 3))

            test_input = test_input.reshape((test_input.shape[0], 38 * 3))
            test_output = test_output.reshape((test_output.shape[0], 38 * 3))

            history = self.model.fit(train_input, train_output, epochs=30, batch_size=2, validation_split=0.1)
            scores = self.model.evaluate(test_input, test_output)

            avg_loss += scores[0]
            avg_mae += scores[1]

        logger.info("K fold average mae: {}".format(avg_mae / num_of_split))
        logger.info("K fold average loss: {}".format(avg_loss / num_of_split))

        if mname is not None:
            self.model.save(mname)

        return history


class Ploter:

    def __init__(self, history=None):

        if history is not None:
            self.history = history
            self.acc = self.history.history['mae']
            self.loss = self.history.history['loss']
            self.val_acc = self.history.history['val_mae']
            self.val_loss = self.history.history['val_loss']
            self.epochs = list(range(1, len(self.acc) + 1))
        else:
            self.load(Path(root_dir) / "results/history.json")

        self.write()

    def load(self, fname):
        with open(fname) as f:
            results = json.load(f)
        self.acc = results['acc']
        self.loss = results['loss']
        self.val_acc = results['val_acc']
        self.val_loss = results['val_loss']
        self.epochs = results['epochs']

    def write(self):
        results = {'acc': self.acc, 'loss': self.loss, 'val_acc': self.val_acc, 'val_loss': self.val_loss,
                   'epochs': self.epochs}
        with open(Path(root_dir) / "results/history.json", "w") as f:
            json.dump(results, f)

    @plot_wrap
    def plot(self, fname=None):
        logger.info("Plotting the acc and loss curve.")

        plt.plot(self.epochs, self.acc, "ro", label='acc')
        plt.plot(self.epochs, self.loss, "bo", label='loss')
        plt.plot(self.epochs, self.val_acc, 'r', label='val_acc')
        plt.plot(self.epochs, self.val_loss, 'b', label='val_loss')
        plt.legend(loc='best', fontsize=14)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("epochs", fontsize=22)

        if fname is not None:
            plt.savefig(fname)
        else:
            plt.show()
