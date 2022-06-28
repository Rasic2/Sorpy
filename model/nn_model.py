import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    def __init__(self, atom_in_fea_num, atom_out_fea_num, bond_in_fea_num, bond_out_fea_num): # Nx2, Nx12x3
        super(GraphConvLayer, self).__init__()
        self.atom_in_fea_num = atom_in_fea_num
        self.atom_out_fea_num = atom_out_fea_num
        self.bond_in_fea_num = bond_in_fea_num
        self.bond_out_fea_num = bond_out_fea_num
        self.weight = Parameter(torch.FloatTensor(bond_in_fea_num, bond_out_fea_num))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, atom, bond):
        bond = torch.reshape(bond,shape=(-1, self.bond_in_fea_num))
        bond = torch.mm(bond, self.weight)
        bond = torch.reshape(bond, shape=(atom.shape[0], -1, self.bond_out_fea_num))
        return atom, bond


class Model(nn.Module):
    def __init__(self, atom_in_fea_num, atom_out_fea_num, bond_in_fea_num, bond_out_fea_num):
        super(Model, self).__init__()
        self.conv1 = GraphConvLayer(atom_in_fea_num, atom_out_fea_num, bond_in_fea_num, bond_out_fea_num)
        self.conv2 = GraphConvLayer(atom_in_fea_num, atom_out_fea_num, bond_in_fea_num, bond_out_fea_num)

    def forward(self, atom, bond):
        atom, bond = self.conv1(atom, bond)
        atom, bond = self.conv2(atom, bond)
        return atom, bond
