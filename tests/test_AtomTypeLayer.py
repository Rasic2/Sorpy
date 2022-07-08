import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.io_file import POSCAR
from common.logger import root_dir
from network.layers import AtomTypeLayer

if __name__ == '__main__':
    structure = POSCAR(fname=Path(f"{root_dir}/train_set/input/POSCAR_1-1")).to_structure()
    structure.find_neighbour_table(neighbour_num=12)

    atom = structure.atoms[0]
    atom_feature_period = F.one_hot(torch.LongTensor([atom.period]), num_classes=7)
    atom_feature_group = F.one_hot(torch.LongTensor([atom.group]), num_classes=18)
    atom_feature_coordination = F.one_hot(torch.LongTensor([atom.coordination_number]), num_classes=12)

    filter = np.arange(0.5, 4.5, 0.15)
    bond_dist = np.array([2.395])
    atom_bond = torch.Tensor(np.exp(-(bond_dist[:, np.newaxis] - filter)**2 / 0.15**2))

    atom_feature = torch.cat((atom_feature_period, atom_feature_group, atom_feature_coordination, atom_bond), dim=1)
    model = AtomTypeLayer(64, 25)
    atom_type = model(atom_feature)

    print()
