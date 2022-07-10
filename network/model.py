import torch
from torch import nn

from network.layers import AtomConvLayer, EmbeddingLayer, BondConvLayer, AtomTypeLayer


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

    def forward(self, atom, bond, adj_matrix, adj_matrix_tuple, ):
        atom_type_update = torch.Tensor(atom.shape[0], atom.shape[1], 25)

        if torch.cuda.is_available():
            atom_type_update = atom_type_update.cuda()

        for name, group in zip(*self.atom_type):
            atom_type_update[:, group] = self._AtomType[f"AtomType_{name}"](atom[:, group])

        atom_update = self.AtomConv(atom_type_update, bond, adj_matrix)
        bond_diatom, temp = self.embedding(atom_update, adj_matrix_tuple)
        bond_update = self.BondConv(bond_diatom, bond, adj_matrix)

        return atom_update, bond_update, temp
