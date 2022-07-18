import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from common.io_file import POSCAR
from common.logger import logger
from common.manager import DirManager

PERIOD = 7
GROUP = 18
MAX_CN = 12
FILTER = np.arange(0.5, 4.5, 0.15)


class StructureDataset(Dataset):
    def __init__(self, xdat_dir: (DirManager, None), energy_file: (Path, None), data=None):
        self.xdat_dir = xdat_dir
        self.energy_file = energy_file
        self.data = self.load_data() if data is None else data

    def __getitem__(self, index):
        if isinstance(index, int):
            return tuple(self.data[i][index] for i in range(len(self.data)))
        elif isinstance(index, slice):
            data = tuple(self.data[i][index] for i in range(len(self.data)))
            return StructureDataset(xdat_dir=None, energy_file=self.energy_file, data=data)

    def __len__(self):
        return self.data[0].shape[0]

    def load_data(self):
        count = 0

        atom_feature_input = []
        adj_matrix_input = []
        adj_matrix_tuple_input = []
        bond_dist3d_input = []
        energy = []

        energy_dict = self.load_energy(self.energy_file)
        # energy_numpy = np.array(sum([value for value in energy_dict.values()], []))
        # energy_min = np.min(energy_numpy)
        # energy_max = np.max(energy_numpy)

        for structure_dir in self.xdat_dir.sub_dir:
            logger.info(f"Loading the {structure_dir.dname.name}, which have {len(structure_dir)} total files")
            key = structure_dir.dname.name.split("_")[-1]
            energy.append(energy_dict[key])

            assert len(energy[-1]) == len(structure_dir.all_files_path), \
                f"The len(energy) = {len(energy[-1])} && len(structure) = {len(structure_dir.all_files_path)} are different!"

            for fname in structure_dir.all_files_path:
                structure = POSCAR(fname=fname).to_structure()
                structure.find_neighbour_table(neighbour_num=12)
                atom_feature, adj_matrix, adj_matrix_tuple, bond_dist3d = StructureDataset.transformer(structure)

                # put the structure into the dataset
                atom_feature_input.append(atom_feature)
                bond_dist3d_input.append(bond_dist3d)
                adj_matrix_input.append(adj_matrix)
                adj_matrix_tuple_input.append(adj_matrix_tuple)

            count += 1
            if count % 100 == 0:
                logger.info(f"-----{count} directories have been loaded!-----")

        # flatten
        energy = sum(energy, [])
        # energy = np.array(energy)
        # energy = (energy - energy_min) / (energy_max - energy_min)

        # shuffle the dataset
        index = list(range(len(atom_feature_input)))
        random.shuffle(index)
        atom_feature_input = np.array(atom_feature_input)[index]
        bond_dist3d_input = np.array(bond_dist3d_input)[index]
        adj_matrix_input = np.array(adj_matrix_input)[index]
        adj_matrix_tuple_input = np.array(adj_matrix_tuple_input)[index]
        energy = np.array(energy)[index]

        return torch.Tensor(np.array(atom_feature_input)), torch.Tensor(np.array(bond_dist3d_input)), \
               torch.LongTensor(np.array(adj_matrix_input)), torch.LongTensor(np.array(adj_matrix_tuple_input)), \
               torch.Tensor(np.array(energy))

    @staticmethod
    def transformer(structure_input):
        atom_feature_period = F.one_hot(torch.LongTensor(structure_input.atoms.period), num_classes=PERIOD)
        atom_feature_group = F.one_hot(torch.LongTensor(structure_input.atoms.group), num_classes=GROUP)
        atom_feature_coordination = F.one_hot(torch.LongTensor(structure_input.atoms.coordination_number),
                                              num_classes=MAX_CN)
        bond_dist_neighbor = structure_input.neighbour_table.dist[:, 0]  # neighbour bond-length
        atom_bond = torch.Tensor(np.exp(-(bond_dist_neighbor[:, np.newaxis] - FILTER) ** 2 / 0.15 ** 2))
        atom_feature = torch.cat((atom_feature_period, atom_feature_group, atom_feature_coordination, atom_bond),
                                 dim=1).numpy()
        adj_matrix = structure_input.neighbour_table.index
        adj_matrix_tuple = structure_input.neighbour_table.index_tuple
        bond_dist3d = structure_input.neighbour_table.dist3d

        return atom_feature, adj_matrix, adj_matrix_tuple, bond_dist3d

    @staticmethod
    def load_energy(energy_file):
        with open(energy_file) as f:
            cfg = f.readlines()

        energy_dict = defaultdict(list)
        for line in cfg:
            if len(line.split()) == 1:
                key = line.split()[0]
            else:
                energy_dict[key].append(float(line.split()[-1]))

        return energy_dict
