#!/usr/bin/env python

import math
import random
import itertools

import numpy as np
from pymatgen.io.vasp import Poscar

from _logger import *
from common.structure import Molecule, Latt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 屏蔽TF日志输出

_ELEMENT = ['Ce', 'O', 'C']

input_dir = os.path.join(current_dir, "input")
output_dir = os.path.join(current_dir, "output")


class FileManager:
    """
        single POSCAR-like file Manager
    """

    def __init__(self, fname: str, mol_index=None):
        """
        :param fname:   file name
        """
        self.fname = fname
        self.type = fname.split("_")[0].split("/")[-1]
        self.index = fname.split("_")[-1]

        if type(mol_index) == list:
            self.mol_index = mol_index
        elif type(mol_index) == str:
            if "-" in mol_index:
                self.mol_index = list(range(int(mol_index.split("-")[0]) - 1, int(mol_index.split("-")[1])))
            else:
                self.mol_index = [int(mol_index)]
        else:
            self.mol_index = None
            logger.warning("Can't align the Molecule.")

        from collections import defaultdict
        self.atom_dict = defaultdict(list)

    def __eq__(self, other):
        return self.type == other.type and self.index == other.index

    def __le__(self, other):
        return self.type == other.type and self.index <= other.index

    def __gt__(self, other):
        return self.type == other.type and self.index > other.index

    def __repr__(self):
        return f"{self.type}: {self.index}"

    @property
    def poscar(self):
        """
        Call the Pymatgen to read the POSCAR-like file

        :return:    pymatgen.io.vasp.Poscar <class>
        """
        return Poscar.from_file(self.fname)

    @property
    def structure(self):
        return self.poscar.structure

    @property
    def latt(self):
        return Latt(self.structure.lattice.matrix)

    @property
    def sites(self):
        return self.structure.sites

    @property
    def species(self):
        return self.structure.species

    @property
    def coords(self):
        return self.structure.frac_coords

    @property
    def atom_num(self):
        return len(self.coords)

    @property
    def molecule(self):
        if type(self.mol_index) == list and len(self.mol_index):
            return [(ii + 1, site) for ii, site in zip(self.mol_index, np.array(self.structure)[self.mol_index])]
        else:
            return None

    def _setter_slab_index(self):
        if self.molecule:
            self.slab_index = list(set(list(range(self.atom_num))).difference(set(self.mol_index)))
        else:
            self.slab_index = list(range(self.atom_num))

    @property
    def slab(self):
        self._setter_slab_index()
        return [(ii, site) for ii, site in zip(self.slab_index, np.array(self.structure)[self.slab_index])]

    def align_the_element(self):

        self._setter_slab_index()
        for ii, item in enumerate(self.species):
            if ii in self.slab_index:
                self.atom_dict[item].append(ii)
            elif ii in self.mol_index:
                self.atom_dict["mol"].append(ii)

    @property
    def mcoords(self):
        """
        Cal Slab frac_coors + Mol <anchor_frac + bond length + theta + phi>
        """
        self._setter_slab_index()

        if self.molecule:
            m = Molecule(self.coords[self.mol_index], self.latt.matrix)
            slab_coor = self.coords[self.slab_index]
            m_anchor = m[0].frac_coord.reshape((1, 3))
            m.phi_m = m.phi if m.phi >= 0 else 360 + m.phi
            m_intercoor = np.array([m.bond_length, m.theta, m.phi_m]).reshape((1, 3))
            m_intercoor = (m_intercoor/ [[1, 180, 360]] - [[1.142, 0, 0]])
            return np.concatenate((slab_coor, m_anchor, m_intercoor), axis=0)


class DirManager:
    """
        Input/Output directory manager
    """

    def __init__(self, dname: str, ftype: str, mol_index=None):
        """
        :param dname:       directory name
        :param ftype:        determine which type of file including (e.g. POSCAR or CONTCAR)
        """
        self.dname = dname
        self.type = ftype
        self.mol_index = mol_index
        if self.mol_index:
            logger.info(f"Molecule was align to {self.mol_index} location.")

    def one_file(self, fname):
        """
        The single file manager

        :param fname:   file name
        :return:        FileManager(fname)
        """
        return FileManager(f"{self.dname}/{fname}", mol_index=self.mol_index)


    def __all_files(self):
        for fname in os.listdir(self.dname):
            if fname.startswith(self.type):
                yield FileManager(f"{self.dname}/{fname}", mol_index=self.mol_index)

    @property
    def all_files(self):
        return sorted(list(self.__all_files()))

    @property
    def count(self):
        return len(self.all_files)

    @property
    def coords(self):
        return np.array([file.coords for file in self.all_files])

    def split_slab_mol(self):
        for file in self.all_files:
            file.align_the_element()
            return file.atom_dict

    @property
    def mcoords(self):
        return np.array([file.mcoords for file in self.all_files])


class CoorTailor:

    def __init__(self, input_arr, output_arr, repeat_unit: int, intercoor_index = None):

        self.input_arr, self.output_arr = input_arr, output_arr
        self.repeat_unit = repeat_unit

        self.intercoor_index = intercoor_index if intercoor_index is not None else []
        self.total_index = list(range(input_arr.shape[1]))

        if self.intercoor_index is not None:
            self.frac_index = list(set(self.total_index).difference(set(self.intercoor_index)))
        else:
            self.frac_index = self.total_index

    def run(self, boundary: float = 0.2, num: int = 2):
        #self._expand_xy() # TODO delete make the model work like trash
        self._pbc_apply()
        #self._tailor_xy() # TODO delete make the model work like trash
        self.input_arr = CoorTailor._tailor_z(self.input_arr, self.frac_index, self.intercoor_index, boundary, num)
        self.output_arr = CoorTailor._tailor_z(self.output_arr, self.frac_index, self.intercoor_index, boundary, num)

    def _expand_xy(self):
        """
        For supercell, translate the left-bottom region into other area (Data Improver)

        :return:                    data_input_arr, data_output_arr after expanding the xy coordinates
        """
        trans_coor_i_iner, trans_coor_o_iner = [], []
        input_arr, output_arr = self.input_arr.copy(), self.output_arr.copy()

        for coor_i, coor_o in zip(input_arr, output_arr):
            trans = [_ for _ in range(self.repeat_unit)]
            coor_trans_i, coor_trans_o = [], []

            for item in itertools.product(trans, trans, [0]):
                coor_t_i = coor_i[self.frac_index] + np.array(item) / self.repeat_unit
                coor_t_o = coor_o[self.frac_index] + np.array(item) / self.repeat_unit

                coor_t_i = np.where(coor_t_i > 1, coor_t_i - 1, coor_t_i)
                coor_t_o = np.where(coor_t_o > 1, coor_t_o - 1, coor_t_o)

                coor_t_i = np.concatenate((coor_t_i, coor_i[self.intercoor_index]), axis=0) # inter_coor append
                coor_t_o = np.concatenate((coor_t_o, coor_o[self.intercoor_index]), axis=0) # inter_coor append

                coor_trans_i.append(coor_t_i)
                coor_trans_o.append(coor_t_o)

            trans_coor_i_iner += coor_trans_i
            trans_coor_o_iner += coor_trans_o

        self.input_arr, self.output_arr = np.array(trans_coor_i_iner), np.array(trans_coor_o_iner)

    def _pbc_apply(self):
        """
        Handling the periodic-boundary-condition

        :return:                    data_input, data_output after pbc handle
        """
        data_input_arr, data_output_arr = self.input_arr.copy()[:, self.frac_index, :], self.output_arr.copy()[:, self.frac_index, :]

        data_output_arr = np.where((data_input_arr - data_output_arr) > 0.5, data_output_arr + 1, data_output_arr)
        data_output_arr = np.where((data_input_arr - data_output_arr) < -0.5, data_output_arr - 1, data_output_arr)

        data_input_arr = np.concatenate((data_input_arr, self.input_arr[:, self.intercoor_index, :]), axis=1)
        data_output_arr = np.concatenate((data_output_arr, self.output_arr[:, self.intercoor_index, :]), axis=1)

        self.input_arr, self.output_arr = data_input_arr, data_output_arr

    @staticmethod
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

    def __run_tailor_xy(self, data_input_arr, data_output_arr, flag: str = "n"):
        """
        tailor_xy coordinates in (0, 1) area. (if failed print warning)

        :param data_input_arr:              data_input array
        :param data_output_arr:             data_output array
        :param flag::                       falg determing handling the (<0 or >1) case
        :return:                            data_input_arr, data_output_arr after tailoring the xy coordinates
        """
        trans_ = [_ for _ in range(self.repeat_unit)]
        trans = list(itertools.product(trans_, trans_, [0]))

        for fileno, (i_, j_) in enumerate(zip(data_input_arr, data_output_arr)):
            finditi = np.where((i_ < 0)) if flag == "n" else np.where((i_ > 1))
            finditj = np.where((j_ < 0)) if flag == "n" else np.where((j_ > 1))
            findit_ij = (np.concatenate([finditi[0], finditj[0]]), np.concatenate([finditi[1], finditj[1]]))
            findit = self.__remove_repeat(findit_ij)
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

    def _tailor_xy(self):
        """
        tailor xy coordinates for (<0 or >1) case

        :return:                          data_input_arr, data_output_arr after tailoring the xy coordinates
        """
        data_input_arr, data_output_arr = self.input_arr.copy()[:, self.frac_index, :], self.output_arr.copy()[:, self.frac_index, :]

        data_input_arr, data_output_arr = self.__run_tailor_xy(data_input_arr, data_output_arr, "n")  # xy < 0 case
        data_input_arr, data_output_arr = self.__run_tailor_xy(data_input_arr, data_output_arr, "p")  # xy > 1 case

        data_input_arr = np.concatenate((data_input_arr, self.input_arr[:, self.intercoor_index, :]), axis=1)
        data_output_arr = np.concatenate((data_output_arr, self.output_arr[:, self.intercoor_index, :]), axis=1)

        self.input_arr, self.output_arr = data_input_arr, data_output_arr

    @staticmethod
    def _tailor_z(coor, frac_index, intercoor_index, boundary: float, num: int):
        """
        Put the Slab+Mol translate along the Z-axis. (Data Improver)

        :param coor:              Slab+Mol coor ndarray
        :param boundary:          max-value of the translate
        :param num:               the num of part split between the [0,boundary]
        :return:                  the translate coor (len = 2*num*ori_len)
        """
        ori_coor = coor.copy()

        for _ in range(num):
            delta_z = ((_+1) * boundary / num)
            coor_p = ori_coor.copy()[:, frac_index, :]
            coor_n = ori_coor.copy()[:, frac_index, :]
            coor_p = coor_p + np.array([0.0, 0.0, delta_z])
            coor_n = coor_n + np.array([0.0, 0.0, -delta_z])
            coor_p = np.concatenate((coor_p, ori_coor[:, intercoor_index, :]), axis=1)
            coor_n = np.concatenate((coor_n, ori_coor[:, intercoor_index, :]), axis=1)

            coor = np.append(coor, coor_p, axis=0)
            coor = np.append(coor, coor_n, axis=0)

        return coor


class Model:

    def __init__(self, data_input_arr, data_output_arr, atom_list_iner, k_fold_flag):

        from keras import models
        from keras import layers

        self.model = models.Sequential()
        self.model.add(layers.Dense(1024, activation='relu', input_shape=(38 * 3,)))
        self.model.add(layers.Dense(114))
        self.model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

        self.data_input = data_input_arr
        self.data_output = data_output_arr

        self.atom_list = atom_list_iner

        self.K_fold_flag = k_fold_flag

    @staticmethod
    def __tailor_atom_order(train_input_iner, train_output_iner, atom_list_iner):
        """
        Shuffle the atom order in the data_input, data_output (train set) <helper func>
        TODO: Something wrong make the model work like trash !!!

        :param train_input_iner:            train_input array
        :param train_output_iner:           train_output array
        :return:                            train_input array, train_output array after suffering the atom order
        """

        shuffle_train_input, shuffle_train_output = [], []

        for i, j in zip(train_input_iner, train_output_iner):
            random_list = []
            for key in atom_list_iner.keys():
                k = atom_list_iner[key]
                random.shuffle(k) if key != "mol" else None
                random_list += k
            i = i[random_list]
            j = j[random_list]
            i = i.reshape(38 * 3)
            j = j.reshape(38 * 3)
            shuffle_train_input.append(i)
            shuffle_train_output.append(j)

        train_input_iner = np.array(shuffle_train_input)
        train_output_iner = np.array(shuffle_train_output)

        return train_input_iner, train_output_iner

    def hold_out(self, percent_iner=0.8):
        """
        Hold-out method for the model test.

        :param percent_iner:                train:test percent
        :return:                            mae and loss
        """

        shape = self.data_input.shape
        count = shape[0]
        count_train = math.ceil(count * percent_iner)
        count_test = count - count_train

        # train_input_arr, train_output_arr = Model.__tailor_atom_order(self.data_input[:count_train],
        #                                                              self.data_output[:count_train],
        #                                                              self.atom_list)
        train_input_arr, train_output_arr = self.data_input[:count_train], self.data_output[:count_train]
        train_input_arr, train_output_arr = train_input_arr.reshape(
            (count_train, shape[1] * shape[2])), train_output_arr.reshape((count_train, shape[1] * shape[2]))

        test_input_arr, test_output_arr = self.data_input[count_train:], self.data_output[count_train:]
        test_input_arr, test_output_arr = test_input_arr.reshape(
            (count_test, shape[1] * shape[2])), test_output_arr.reshape((count_test, shape[1] * shape[2]))

        history = self.model.fit(train_input_arr, train_output_arr, epochs=50, batch_size=2, validation_split=0.1)
        predict = self.model.predict(test_input_arr)
        self.model.save("CeO2_111_CO_test.h5")
        scores = self.model.evaluate(test_input_arr, test_output_arr)

        return history, scores

    def k_fold_validation(self, n_split_iner: int):
        """
        K fold validation method for the Model test.

        :param n_split_iner:                    how many fold
        :return:                                avg_mae, ave loss
        """
        from sklearn.model_selection import KFold

        avg_mae_iner = 0
        avg_loss_iner = 0

        for train_index, test_index in KFold(n_split_iner).split(self.data_input):
            train_input, test_input = self.data_input[train_index], self.data_input[test_index]
            train_output, test_output = self.data_output[train_index], self.data_output[test_index]

            # train_input, train_output = Model.__tailor_atom_order(train_input, train_output, self.atom_list)

            train_input = train_input.reshape((train_input.shape[0], 38 * 3))
            train_output = train_output.reshape((train_output.shape[0], 38 * 3))

            test_input = test_input.reshape((test_input.shape[0], 38 * 3))
            test_output = test_output.reshape((test_output.shape[0], 38 * 3))

            history = self.model.fit(train_input, train_output, epochs=30, batch_size=2, validation_split=0.1)
            scores = self.model.evaluate(test_input, test_output)

            avg_loss_iner += scores[0]
            avg_mae_iner += scores[1]

        self.model.save("CeO2_111_CO_kfold.h5")
        return history, avg_mae_iner, avg_loss_iner


class Ploter:

    def __init__(self, history_i):
        self.history = history_i

        self.acc = self.history.history['mae']
        self.loss = self.history.history['loss']
        self.val_acc = self.history.history['val_mae']
        self.val_loss = self.history.history['val_loss']
        self.epochs = range(1, len(self.acc) + 1)

    def plot(self, fname):
        logger.info("Plotting the acc and loss curve.")

        from matplotlib import pyplot as plt
        plt.plot(self.epochs, self.acc, "ro")
        plt.plot(self.epochs, self.val_acc, 'r')
        plt.plot(self.epochs, self.loss, "bo")
        plt.plot(self.epochs, self.val_loss, 'b')
        plt.savefig(fname)


if __name__ == "__main__":

    logger.info("Load the structure information.")
    input_DM = DirManager("input", "POSCAR", "37-38")
    output_DM = DirManager("output", "CONTCAR", "37-38")

    input_coor = input_DM.mcoords
    output_coor = output_DM.mcoords

    atom_list = input_DM.split_slab_mol()
    logger.info(f"The atom_list is {atom_list}")

    logger.info("Apply the PBC and tailor the x-y, z coordinates.")
    repeat = 2  # supercell (2x2)

    CT = CoorTailor(input_coor, output_coor, repeat, intercoor_index=[37])
    CT.run(boundary=0.2, num=2)

    input_coor, output_coor = CT.input_arr, CT.output_arr
    logger.info(f"Data Shape: {input_coor.shape}")

    logger.info("Shuffle the data.")
    data_input, data_output = input_coor, output_coor
    index = list(range(len(data_input)))
    random.shuffle(index)
    data_input = data_input[index]
    data_output = data_output[index]

    model = Model(data_input, data_output, atom_list, k_fold_flag=False)

    if model.K_fold_flag:
        logger.info("Train and test the model applying the K-fold validation method.")
        n_split = 5
        history, avg_mae, avg_loss = model.k_fold_validation(n_split)
        logger.info("K fold average mae: {}".format(avg_mae / n_split))
        logger.info("K fold average loss: {}".format(avg_loss / n_split))
    else:
        persent = 0.8
        logger.info(
            f"Train and test the model applying the hold-out method. \
            <train:test = {math.ceil(persent * 100)}:{math.ceil((1 - persent) * 100)}>")
        history, (loss, mae) = model.hold_out()
        logger.info(f"mae = {mae}")
        logger.info(f"loss = {loss}")

    p = Ploter(history)
    p.plot("CeO2_111_history.svg")
