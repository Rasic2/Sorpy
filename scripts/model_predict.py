from pathlib import Path

import torch

from common.logger import root_dir
from common.manager import DirManager

import torch.nn.functional as F

from common.structure import Structure

model = torch.load(f"{root_dir}/model.pth")
predict_dir = DirManager(dname=Path(f'{root_dir}/test_set/ori-input'))

model.eval()
for item in predict_dir.all_files:
    structure = item.file.to_structure()
    structure.find_neighbour_table()

    atom_feature_period = F.one_hot(torch.LongTensor(structure.atoms.period), num_classes=7)
    atom_feature_group = F.one_hot(torch.LongTensor(structure.atoms.group), num_classes=18)
    atom_feature = torch.cat((atom_feature_period, atom_feature_group), dim=1).numpy()
    adj_matrix = structure.neighbour_table.index
    adj_matrix_tuple = structure.neighbour_table.index_tuple
    bond_dist3d = structure.neighbour_table.dist3d

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

    structure_predict = Structure.from_adj_matrix(structure, adj_matrix[0].cpu().detach().numpy(),
                                                 adj_matrix_tuple[0].cpu().detach().numpy(),
                                                 bond_update[0].cpu().detach().numpy(), 0)
    structure_predict.to_POSCAR(f"../test_set/ori-CGCNN/POSCAR_{item.index}")

print(model)
