import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from common.io_file import POSCAR, CONTCAR
from common.logger import logger
from common.manager import DirManager

PERIOD = 7
GROUP = 18


class StructureDataset(Dataset):
    def __init__(self, input_dir: (DirManager, None), output_dir: (DirManager, None), data=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.data = self.load_data() if data is None else data

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.data[0][index], self.data[1][index], self.data[2][index], self.data[3][index], self.data[4][index]
        elif isinstance(index, slice):
            data = self.data[0][index], self.data[1][index], self.data[2][index], self.data[3][index], self.data[4][index]
            return StructureDataset(input_dir = None, output_dir = None, data=data)

    def __len__(self):
        return self.data[0].shape[0]

    def load_data(self):
        atom_feature_input = []
        bond_dist3d_input = []
        bond_dist3d_output = []
        adj_matrix_input = []
        adj_matrix_tuple_input = []
        count = 0
        for fname_input, fname_out in zip(self.input_dir.all_files_path, self.output_dir.all_files_path):

            # load structures
            structure, atom_feature, bond_dist3d, adj_matrix, adj_matrix_tuple = self._load_input(
                fname=fname_input)
            bond_dist3d_target = self._load_output(fname=fname_out, adj_matrix=adj_matrix)

            # align the input and output
            # diff = (bond_dist3d_target - bond_dist3d) / (0.5 * structure.lattice.length)
            # diff_around = np.around(diff)
            # diff_around = diff_around * 0.5 * structure.lattice.length
            # bond_dist3d_target -= diff_around

            count += 1
            if count % 100 == 0:
                logger.info(f"-----{count} files have been loaded!-----")

            # put the structure into the dataset
            atom_feature_input.append(atom_feature)
            bond_dist3d_input.append(bond_dist3d) # POSCAR data
            adj_matrix_input.append(adj_matrix)
            adj_matrix_tuple_input.append(adj_matrix_tuple)
            bond_dist3d_output.append(bond_dist3d_target) # CONTCAR data

        # # shuffle the dataset
        # index = list(range(len(atom_feature_input)))
        # random.shuffle(index)
        # atom_feature_input = np.array(atom_feature_input)[index]
        # bond_dist3d_input = np.array(bond_dist3d_input)[index]
        # adj_matrix_input = np.array(adj_matrix_input)[index]
        # adj_matrix_tuple_input = np.array(adj_matrix_tuple_input)[index]
        # bond_dist3d_output = np.array(bond_dist3d_output)[index]

        return torch.Tensor(np.array(atom_feature_input)), torch.Tensor(np.array(bond_dist3d_input)), \
               torch.LongTensor(np.array(adj_matrix_input)), torch.LongTensor(np.array(adj_matrix_tuple_input)), \
               torch.Tensor(np.array(bond_dist3d_output))

    def _load_input(self, fname):
        structure = POSCAR(fname=fname).to_structure(style="Slab")
        structure.find_neighbour_table(neighbour_num=12)

        atom_feature_period = F.one_hot(torch.LongTensor(structure.atoms.period), num_classes=PERIOD)
        atom_feature_group = F.one_hot(torch.LongTensor(structure.atoms.group), num_classes=GROUP)
        atom_feature = torch.cat((atom_feature_period, atom_feature_group), dim=1).numpy()
        adj_matrix = structure.neighbour_table.index
        adj_matrix_tuple = structure.neighbour_table.index_tuple
        bond_dist3d = structure.neighbour_table.dist3d

        return structure, atom_feature, bond_dist3d, adj_matrix, adj_matrix_tuple

    def _load_output(self, fname, adj_matrix):
        structure = CONTCAR(fname=fname).to_structure(style="Slab")
        structure.find_neighbour_table_from_index(adj_matrix=adj_matrix)

        return structure.neighbour_table.dist3d
