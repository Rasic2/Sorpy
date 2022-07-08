import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from common.io_file import POSCAR
from common.logger import root_dir
from network.layers import AtomTypeLayer
from network.model import Model

if __name__ == '__main__':
    structure = POSCAR(fname=Path(f"{root_dir}/train_set/input/POSCAR_1-1")).to_structure()
    structure.find_neighbour_table(neighbour_num=12)

    # atom_feature
    atoms = structure.atoms
    atom_feature_period = F.one_hot(torch.LongTensor(atoms.period), num_classes=7)
    atom_feature_group = F.one_hot(torch.LongTensor(atoms.group), num_classes=18)
    atom_feature_coordination = F.one_hot(torch.LongTensor(atoms.coordination_number), num_classes=12)
    bond_dist = structure.neighbour_table.dist
    bond_dist_neighbor = bond_dist[:, 0]
    filter = np.arange(0.5, 4.5, 0.15)
    atom_bond = torch.Tensor(np.exp(-(bond_dist_neighbor[:, np.newaxis] - filter) ** 2 / 0.15 ** 2))
    atom_feature = torch.cat((atom_feature_period, atom_feature_group, atom_feature_coordination, atom_bond), dim=1)

    # atom_type
    atom_type = structure.atoms.atom_type
    atom_type_index = [(index, item) for index, item in enumerate(atom_type)]
    atom_type_sort_index = sorted(atom_type_index, key=lambda x: x[1])
    atom_type_group = [list(item) for key, item in itertools.groupby(atom_type_sort_index, key=lambda x: x[1])]
    atom_type_group_name = [group[0][1] for group in atom_type_group]
    atom_type_group_index = [[item[0] for item in group] for group in atom_type_group]

    # model
    model = Model(atom_type=(atom_type_group_name, atom_type_group_index), atom_in_fea_num=25, atom_out_fea_num=25,
                  bond_in_fea_num=3,
                  bond_out_fea_num=3)
    loss_fn = nn.L1Loss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    atom_feature = atom_feature.view(-1, *atom_feature.shape)
    adj_matrix = structure.neighbour_table.index

    structure_output = POSCAR(fname=Path(f"{root_dir}/train_set/output/CONTCAR_1-1")).to_structure()
    structure_output.find_neighbour_table(adj_matrix=adj_matrix)
    bond_dist3d_output = torch.unsqueeze(torch.Tensor(structure_output.neighbour_table.dist3d), 0)

    adj_matrix = torch.unsqueeze(torch.LongTensor(adj_matrix), 0)
    adj_matrix_tuple = torch.unsqueeze(torch.LongTensor(structure.neighbour_table.index_tuple), 0)
    bond_dist3d_input = torch.unsqueeze(torch.Tensor(structure.neighbour_table.dist3d), 0)

    parameters = [(name, param) for name, param in model.named_parameters()]

    for i in range(100):
        atom_update, bond_update = model(atom_feature, bond_dist3d_input, adj_matrix, adj_matrix_tuple)
        loss = loss_fn(bond_update, bond_dist3d_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
        print(torch.max(parameters[16][1].grad))
        print()

    print()
