import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter, Flatten, ReLU, Tanh, Dropout, Linear


class GraphConvLayer(nn.Module):
    def __init__(self, atom_in_fea_num, atom_out_fea_num, bond_in_fea_num, bond_out_fea_num, bias=True):  # 2, 2, 3, 3
        super(GraphConvLayer, self).__init__()
        self.atom_in_fea_num = atom_in_fea_num
        self.atom_out_fea_num = atom_out_fea_num
        self.bond_in_fea_num = bond_in_fea_num
        self.bond_out_fea_num = bond_out_fea_num

        self.flatten = Flatten()
        self.relu = ReLU()
        self.tanh = Tanh()

        self.bias = bias
        self.weight_node = Parameter(torch.FloatTensor(atom_in_fea_num, atom_out_fea_num))
        self.weight_edge = Parameter(torch.FloatTensor(bond_in_fea_num, bond_out_fea_num))
        self.weight_node_to_edge = Parameter(torch.FloatTensor(2 * atom_out_fea_num, bond_in_fea_num))
        if self.bias:
            self.bias_node = Parameter(torch.FloatTensor(atom_out_fea_num))
            self.bias_edge = Parameter(torch.FloatTensor(bond_out_fea_num))
            self.bias_node_to_edge = Parameter(torch.FloatTensor(bond_in_fea_num))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_edge.size(1))
        self.weight_node.data.uniform_(-stdv, stdv)
        self.weight_edge.data.uniform_(-stdv, stdv)
        self.weight_node_to_edge.data.uniform_(-stdv, stdv)
        if self.bias:
            self.bias_node.data.uniform_(-stdv, stdv)
            self.bias_edge.data.uniform_(-stdv, stdv)
            self.bias_node_to_edge.data.uniform_(-stdv, stdv)

    def forward(self, atom, bond, adj_matrix, adj_matrix_tuple):
        """
        @parameter
            atom:               shape=(B, N, F_atom)
            bond:               shape=(B, N, M, F_bond)
            adj_matrix:         shape=(B, N, M)
            adj_matrix_tuple:   shape=(B, NxM, 2)

            B: represent the batch_size
            N: represent the number of atoms
            M: represent the number of neighbours of one atom
            F_atom: represent the number of features of atom
            F_bond: represent the number of features of bond
        """

        # update atom-feature
        atom_root = torch.pow(torch.pow(atom, 2), 0.25)  # positive value
        atom_neighbor = torch.FloatTensor(*adj_matrix.shape, self.atom_in_fea_num)  # shape: (B, N, M, F_atom)
        for batch, _ in enumerate(atom):
            atom_neighbor[batch] = torch.pow(torch.pow(atom[batch][adj_matrix[batch]], 2),
                                             0.25)  # shape: (B, N, M, F_atom)
        if torch.cuda.is_available():
            atom_neighbor = atom_neighbor.cuda()

        bond_norm = torch.pow(torch.sum(torch.pow(bond, 2), dim=-1), 0.5)  # shape: (B, N, M), positive value
        bond_norm = torch.pow(bond_norm, -2)  # 1/(bond-length), shape: (B, N, M)
        bond_norm = F.normalize(bond_norm, p=1, dim=-1)  # row normalization, shape: (B, N, M)
        bond_norm = torch.unsqueeze(bond_norm, -1)  # shape: (B, N, M, 1)
        atom_neighbour_weight = torch.sum(bond_norm * atom_neighbor, -2)  # shape: (B, N, F_atom)
        atom_update = atom_root * atom_neighbour_weight  # shape: (B, N, F_atom)
        atom_update = torch.matmul(atom_update, self.weight_node)  # shape: (B, N, F_atom)
        if self.bias:
            atom_update = atom_update + self.bias_node
        atom_update = self.relu(atom_update)  # positive value

        # update bond-feature, accumulate the two node information
        adj_matrix_tuple_flatten = self.flatten(adj_matrix_tuple)  # shape: (B, NxMx2)
        bond_diatom = torch.Tensor(*adj_matrix_tuple_flatten.shape, self.atom_out_fea_num)  # shape: (B, NxMx2, F_atom)
        for batch, _ in enumerate(atom_update):
            bond_diatom[batch] = atom_update[batch][adj_matrix_tuple_flatten[batch]]
        if torch.cuda.is_available():
            bond_diatom = bond_diatom.cuda()

        bond_diatom = torch.reshape(bond_diatom, shape=(
        adj_matrix_tuple_flatten.shape[0], -1, 2 * self.atom_out_fea_num))  # shape: (B, NxM, 2xF_atom)
        bond_diatom = F.normalize(bond_diatom, p=1, dim=-2)  # column normalization, shape: (B, NxM, 2xF_atom)
        bond_diatom = torch.matmul(bond_diatom, self.weight_node_to_edge)  # shape: (B, NxM, F_bond)
        if self.bias:
            bond_diatom += self.bias_node_to_edge
        bond_diatom = self.tanh(bond_diatom)

        # update bond-feature, transfer the node-information in the edge <sum function>
        bond_update = torch.reshape(bond, shape=bond_diatom.shape)  # shape: (B, NxM, F_bond)
        bond_update = bond_update + bond_diatom
        bond_update = torch.matmul(bond_update, self.weight_edge)
        bond_update = torch.reshape(bond_update,
                                    shape=(*adj_matrix.shape, self.bond_out_fea_num))  # shape: (B, N, M, F_bond)
        if self.bias:
            bond_update = bond_update + self.bias_edge

        return atom_update, bond_update


class Model(nn.Module):
    def __init__(self, atom_in_fea_num, atom_out_fea_num, bond_in_fea_num, bond_out_fea_num, bias=True):
        super(Model, self).__init__()
        self.conv1 = GraphConvLayer(atom_in_fea_num, atom_out_fea_num, bond_in_fea_num, bond_out_fea_num, bias=bias)
        self.conv2 = GraphConvLayer(atom_in_fea_num, atom_out_fea_num, bond_in_fea_num, bond_out_fea_num, bias=bias)
        self.linear1 = Linear(in_features=bond_in_fea_num, out_features=64)
        self.tanh = Tanh()
        self.linear2 = Linear(in_features=64, out_features=bond_out_fea_num)

    def forward(self, atom, bond, adj_matrix, adj_matrix_tuple, ):
        atom, bond = self.conv1(atom, bond, adj_matrix, adj_matrix_tuple)
        # bond = self.linear1(bond)
        # # bond = self.tanh(bond)
        # bond = self.linear2(bond)
        # atom, bond = torch.tanh(atom), torch.tanh(bond)
        # atom, bond = self.conv2(atom, bond, adj_matrix, adj_matrix_tuple)
        return atom, bond
