from pathlib import Path

import numpy as np
import torch

from common.logger import root_dir
from common.manager import DirManager
from common.structure import Structure
from network.dataset import StructureDataset

model = torch.load(f"{root_dir}/model.pth")
predict_dir = DirManager(dname=Path(f'{root_dir}/test_set/ori-input'))
target_dir = DirManager(dname=Path(f'{root_dir}/test_set/ori-output'))

model.eval()
for predict, target in zip(predict_dir.all_files, target_dir.all_files):
    structure_input = predict.file.to_structure()
    structure_input.find_neighbour_table()
    structure_target = target.file.to_structure()

    atom_feature, adj_matrix, adj_matrix_tuple, bond_dist3d = StructureDataset.transformer(structure_input)

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

    structure_predict = Structure.from_adj_matrix(structure_input, adj_matrix[0].cpu().detach().numpy(),
                                                  adj_matrix_tuple[0].cpu().detach().numpy(),
                                                  bond_update[0].cpu().detach().numpy(), 0)
    diff = structure_predict - structure_target
    print(f"max: {np.max(diff):.3f}, min: {np.min(diff):.3f}")
    # structure_predict.to_POSCAR(f"../test_set/ori-CGCNN/POSCAR_{item.index}")

# print(model)
print()