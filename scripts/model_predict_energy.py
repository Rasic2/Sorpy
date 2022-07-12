import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

from common.logger import root_dir
from common.manager import DirManager
from network.dataset import StructureDataset
from network.layers import AtomTypeLayer, AtomConvLayer, EmbeddingLayer, BondConvLayer


class Model(nn.Module):
    def __init__(self, atom_type, atom_in_fea_num, atom_out_fea_num, bond_in_fea_num, bond_out_fea_num, bias=True):
        super(Model, self).__init__()

        self.atom_type = atom_type
        self._AtomType = {}
        for name in atom_type[0]:
            atom_type_layer = AtomTypeLayer(in_features=atom_in_fea_num, out_features=atom_out_fea_num)
            setattr(self, f"AtomType_{name}", atom_type_layer)
            self._AtomType[f"AtomType_{name}"] = atom_type_layer

        self.AtomConv = AtomConvLayer(atom_out_fea_num, atom_out_fea_num, bias=bias)
        self.embedding = EmbeddingLayer(atom_out_fea_num, bond_in_fea_num, bias=bias)
        self.BondConv = BondConvLayer(bond_in_fea_num, bond_out_fea_num, bias=bias)
        self.linear = Linear(in_features=25, out_features=1)

    def forward(self, atom, bond, adj_matrix):
        atom_type_update = torch.Tensor(atom.shape[0], atom.shape[1], 25)

        if torch.cuda.is_available():
            atom_type_update = atom_type_update.cuda()

        for name, group in zip(*self.atom_type):
            atom_type_update[:, group] = self._AtomType[f"AtomType_{name}"](atom[:, group])

        atom_update = self.AtomConv(atom_type_update, bond, adj_matrix)
        energy_predict = torch.mean(atom_update, dim=1)
        energy_predict = self.linear(energy_predict)
        energy_predict = torch.relu(energy_predict)
        energy_predict = torch.squeeze(energy_predict, dim=-1)

        return energy_predict


model = torch.load(f"{root_dir}/model-energy.pth")
loss_fn = nn.MSELoss(reduction='mean')
predict_dir = DirManager(dname=Path(f"{root_dir}/train_set/xdat"))
sample = random.sample(range(len(predict_dir.sub_dir)), 1)
predict_dir._sub_dir = np.array(predict_dir.sub_dir, dtype=object)[sample]

energy_file = Path(f"{root_dir}/train_set/energy_summary")

dataset = StructureDataset(xdat_dir=predict_dir, energy_file=energy_file)

batch_size = 1
predict_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

energy_predict_result = []
energy_target_result = []
model.eval()
for step, data in enumerate(predict_dataloader):
    atom_feature, bond_dist3d, adj_matrix, _, energy = data

    if torch.cuda.is_available():
        atom_feature = atom_feature.cuda()
        adj_matrix = adj_matrix.cuda()
        bond_dist3d = bond_dist3d.cuda()
        energy = energy.cuda()

    energy_predict = model(atom_feature, bond_dist3d, adj_matrix)
    energy_predict_result.append(energy_predict.cpu().detach().numpy())
    energy_target_result.append(energy.cpu().detach().numpy())

plt.plot(energy_predict_result, energy_target_result, "-o")
plt.savefig("loss.png")

print()
