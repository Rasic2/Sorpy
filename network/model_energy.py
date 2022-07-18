import torch
from torch import nn
from torch.nn import Linear, BatchNorm1d, Dropout

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
        # self.embedding = EmbeddingLayer(atom_out_fea_num, bond_in_fea_num, bias=bias)
        # self.BondConv = BondConvLayer(bond_in_fea_num, bond_out_fea_num, bias=bias)
        self.linear = Linear(in_features=25, out_features=1)
        self.BatchNorm = BatchNorm1d(num_features=atom_out_fea_num)
        self.Dropout = Dropout(p=0.25)

    def forward(self, atom, bond, adj_matrix):
        atom_type_update = torch.Tensor(atom.shape[0], atom.shape[1], 25)

        if torch.cuda.is_available():
            atom_type_update = atom_type_update.cuda()

        for name, group in zip(*self.atom_type):
            atom_type_update[:, group] = self._AtomType[f"AtomType_{name}"](atom[:, group])

        atom_type_update = torch.permute(input=atom_type_update, dims=(0, 2, 1))
        atom_type_update = self.BatchNorm(atom_type_update)
        atom_type_update = torch.permute(input=atom_type_update, dims=(0, 2, 1))

        atom_type_update = self.Dropout(atom_type_update)

        atom_update = self.AtomConv(atom_type_update, bond, adj_matrix)  # atom_update.grad.max ~ 0.001

        energy_predict = torch.mean(atom_update, dim=1)  # energy_predict.grad.max ~ 0.1
        energy_predict = self.linear(energy_predict)  # energy.predict.grad = 0
        energy_predict = torch.relu(energy_predict)  # energy_predict.grad = -1
        energy_predict = torch.squeeze(energy_predict, dim=-1)  # energy_predict.grad = -1

        return energy_predict
