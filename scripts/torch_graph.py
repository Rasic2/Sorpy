from pathlib import Path

import torch
from torchviz import make_dot

from common.io_file import POSCAR
from common.logger import root_dir
from network.dataset import StructureDataset

if __name__ == '__main__':
    model = torch.load(f"{root_dir}/model.pth")

    structure = POSCAR(fname=Path(f"{root_dir}/train_set/input/POSCAR_1-1")).to_structure()
    structure.find_neighbour_table()
    atom_feature, adj_matrix, adj_matrix_tuple, bond_dist3d = StructureDataset.transformer(structure)
    atom_feature = torch.unsqueeze(torch.Tensor(atom_feature), 0)
    adj_matrix = torch.unsqueeze(torch.LongTensor(adj_matrix), 0)
    adj_matrix_tuple = torch.unsqueeze(torch.LongTensor(adj_matrix_tuple), 0)
    bond_dist3d = torch.unsqueeze(torch.Tensor(bond_dist3d), 0)
    if torch.cuda.is_available():
        atom_feature = atom_feature.cuda()
        adj_matrix = adj_matrix.cuda()
        adj_matrix_tuple = adj_matrix_tuple.cuda()
        bond_dist3d = bond_dist3d.cuda()

    atom_update, bond_update = model(atom_feature, bond_dist3d, adj_matrix, adj_matrix_tuple)

    graph = make_dot(bond_update)
    graph.format = "png"
    graph.directory = f"{root_dir}/network"
    graph.view()