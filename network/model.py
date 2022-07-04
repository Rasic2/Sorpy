from torch import nn

from network.layers import AtomConvLayer, EmbeddingLayer, BondConvLayer


class Model(nn.Module):
    def __init__(self, atom_in_fea_num, atom_out_fea_num, bond_in_fea_num, bond_out_fea_num, bias=True):
        super(Model, self).__init__()
        self.AtomConv = AtomConvLayer(atom_in_fea_num, atom_out_fea_num, bias=bias)
        self.embedding = EmbeddingLayer(atom_out_fea_num, bond_in_fea_num, bias=bias)
        self.BondConv = BondConvLayer(bond_in_fea_num, bond_out_fea_num, bias=bias)

    def forward(self, atom, bond, adj_matrix, adj_matrix_tuple, ):
        atom_update = self.AtomConv(atom, bond, adj_matrix)
        bond_diatom = self.embedding(atom_update, adj_matrix_tuple)
        bond_update = self.BondConv(bond_diatom, bond, adj_matrix)

        return atom_update, bond_update
