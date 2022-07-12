import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter, Linear, BatchNorm1d, Sequential, Tanh, ReLU, Sigmoid


class AtomTypeLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(AtomTypeLayer, self).__init__()

        self.sequential = Sequential(
            Linear(in_features=in_features, out_features=out_features),
            Sigmoid(),
        )

    def forward(self, atom):
        atom_type = self.sequential(atom)

        return atom_type


class AtomConvLayer(nn.Module):
    def __init__(self, atom_in_fea_num, atom_out_fea_num, bias=True):
        super(AtomConvLayer, self).__init__()
        # self.BatchNorm = BatchNorm1d(num_features=atom_out_fea_num)
        self.atom_in_fea_num = atom_in_fea_num
        self.atom_out_fea_num = atom_out_fea_num

        self.weight_atom_1 = Parameter(torch.FloatTensor(atom_in_fea_num, atom_out_fea_num))
        self.weight_atom_2 = Parameter(torch.FloatTensor(atom_out_fea_num, atom_out_fea_num))

        self.bias = bias
        if self.bias:
            self.bias_atom_1 = Parameter(torch.FloatTensor(atom_out_fea_num))
            self.bias_atom_2 = Parameter(torch.FloatTensor(atom_out_fea_num))

        self.reset_parameters()

    def reset_parameters(self):
        stdv_node = 1. / math.sqrt(self.atom_out_fea_num)

        self.weight_atom_1.data.uniform_(-stdv_node, stdv_node)
        self.weight_atom_2.data.uniform_(-stdv_node, stdv_node)

        if self.bias:
            self.bias_atom_1.data.uniform_(-stdv_node, stdv_node)
            self.bias_atom_2.data.uniform_(-stdv_node, stdv_node)

    def forward(self, atom, bond, adj_matrix):
        """
        @parameter
            atom:               shape=(B, N, F_atom)
            bond:               shape=(B, N, M, F_bond)
            adj_matrix:         shape=(B, N, M)

            B: represent the batch_size
            N: represent the number of atoms
            M: represent the number of neighbours of one atom
            F_atom: represent the number of features of atom
            F_bond: represent the number of features of bond
        """
        assert atom.dim() == 3, f"The dimension of input Tensor is {atom.dim()}, expect 3"

        # update atom-feature
        atom_neighbor = torch.FloatTensor(*adj_matrix.shape, self.atom_in_fea_num)  # shape: (B, N, M, F_atom)
        for batch, _ in enumerate(atom):
            atom_neighbor[batch] = atom[batch, adj_matrix[batch]]

        if torch.cuda.is_available():
            atom_neighbor = atom_neighbor.cuda()

        bond_norm = torch.pow(torch.sum(torch.pow(bond, 2), dim=-1), 0.5)  # shape: (B, N, M), positive value
        bond_norm = torch.pow(bond_norm, -2)  # 1/(bond-length)^2, shape: (B, N, M)
        bond_norm = F.normalize(bond_norm, p=1, dim=-1)  # row normalization, shape: (B, N, M)
        bond_norm = torch.unsqueeze(bond_norm, -1)  # shape: (B, N, M, 1)
        atom_neighbour_weight = torch.sum(bond_norm * atom_neighbor, -2)  # shape: (B, N, F_atom), grad ~ 1E-03

        # atom.grad.abs.mean ~  0.001
        atom_update = atom * atom_neighbour_weight  # shape: (B, N, F_atom), atom_update.grad ~ 0.01
        atom_update = torch.matmul(atom_update, self.weight_atom_1)  # atom_update.grad.max ~ 0.05

        if self.bias:
            atom_update += self.bias_atom_1  # atom_update.grad.max ~ 0.1

        atom_update = F.relu(atom_update)  # positive value, atom_update.grad.max ~ 1E+11

        return atom_update


class EmbeddingLayer(nn.Module):
    def __init__(self, atom_fea_num, bond_fea_num, bias=True):
        super(EmbeddingLayer, self).__init__()
        self.BatchNorm = BatchNorm1d(num_features=bond_fea_num)
        self.atom_fea_num = atom_fea_num

        self.weight_node_to_edge = Parameter(torch.FloatTensor(2 * atom_fea_num, bond_fea_num))

        self.bias = bias
        if self.bias:
            self.bias_node_to_edge = Parameter(torch.FloatTensor(bond_fea_num))

        self.reset_parameters()

    def reset_parameters(self):
        stdv_edge = 1. / math.sqrt(self.weight_node_to_edge.size(1))

        self.weight_node_to_edge.data.uniform_(-stdv_edge, stdv_edge)

        if self.bias:
            self.bias_node_to_edge.data.uniform_(-stdv_edge, stdv_edge)

    def forward(self, atom, adj_matrix_tuple):
        """
        @parameter
            atom:               shape=(B, N, F_atom)
            adj_matrix_tuple:   shape=(B, NxM, 2)

            B: represent the batch_size
            N: represent the number of atoms
            M: represent the number of neighbours of one atom
            F_atom: represent the number of features of atom
        """

        # update bond-feature, accumulate the two node information
        adj_matrix_tuple_flatten = torch.reshape(adj_matrix_tuple, shape=(adj_matrix_tuple.shape[0], -1))
        # shape = (B, NxMx2)
        bond_diatom = torch.Tensor(*adj_matrix_tuple_flatten.shape, self.atom_fea_num)  # shape: (B, NxMx2, F_atom)
        for batch, _ in enumerate(atom):
            bond_diatom[batch] = atom[batch][adj_matrix_tuple_flatten[batch]]

        if torch.cuda.is_available():
            bond_diatom = bond_diatom.cuda()

        bond_diatom = torch.reshape(bond_diatom, shape=(adj_matrix_tuple_flatten.shape[0], -1, 2 * self.atom_fea_num))
        # shape: (B, NxM, 2xF_atom), bond_diatom.grad ~ 1 / x^2, bond_diatom.grad.max ~ 1E+11 (too big)
        bond_diatom = F.normalize(bond_diatom, p=1, dim=-2)
        # column normalization, shape: (B, NxM, 2xF_atom), bond_diatom.max ~ 0.01 (too small), bond_diatom.grad ~ 0.01

        bond_diatom = torch.matmul(bond_diatom, self.weight_node_to_edge)  # weight.grad ~ 1E-04

        if self.bias:
            bond_diatom += self.bias_node_to_edge  # shape: (B, NxM, F_bond)
        # bias.grad ~ 1E-06 (positive and negative offset)

        bond_diatom = torch.permute(bond_diatom, dims=(0, 2, 1))
        bond_diatom = self.BatchNorm(bond_diatom)
        bond_diatom = torch.permute(bond_diatom, dims=(0, 2, 1))  # grad ~ 1. / (batch*N*M*3)

        bond_diatom = torch.tanh(bond_diatom)

        return bond_diatom


class BondConvLayer(nn.Module):
    def __init__(self, bond_in_fea_num, bond_out_fea_num, bias=True):
        super(BondConvLayer, self).__init__()
        self.bond_in_fea_num = bond_in_fea_num
        self.bond_out_fea_num = bond_out_fea_num

        self.weight_edge = Parameter(torch.FloatTensor(bond_in_fea_num, bond_out_fea_num))

        self.bias = bias
        if self.bias:
            self.bias_edge = Parameter(torch.FloatTensor(bond_out_fea_num))  # grad = n / (batch*N*M*3)

        self.reset_parameters()

    def reset_parameters(self):
        stdv_edge = 1. / math.sqrt(self.weight_edge.size(1))

        self.weight_edge.data.uniform_(-stdv_edge, stdv_edge)

        if self.bias:
            self.bias_edge.data.uniform_(-stdv_edge, stdv_edge)

    def forward(self, bond_diatom, bond, adj_matrix):
        """
        @parameter
            bond_diatom:        shape=(B, NxM, F_bond)
            bond:               shape=(B, N, M, F_bond)
            adj_matrix:         shape=(B, N, M)

            B: represent the batch_size
            N: represent the number of atoms
            M: represent the number of neighbours of one atom
            F_bond: represent the number of features of bond
        """
        assert bond.dim() == 4, f"The dimension of input Tensor is {bond.dim()}, expect 4"

        # update bond-feature, transfer the node-information in the edge <sum function>
        bond_update = torch.reshape(bond, shape=bond_diatom.shape)  # shape: (B, NxM, F_bond)
        # (+): bond_diatom.grad.abs.max ~ 1E-04, too small; (*): bond_diatom.grad.abs.max ~ 1E-03
        bond_update = bond_update + bond_diatom  # bond_update.grad ~ 1. / (batch*N*M*3)
        bond_update = torch.matmul(bond_update, self.weight_edge)
        bond_update = torch.reshape(bond_update,  # bond_update.grad ~ 1. / (batch*N*M*3)
                                    shape=(*adj_matrix.shape, self.bond_out_fea_num))  # shape: (B, N, M, F_bond)
        if self.bias:
            bond_update = bond_update + self.bias_edge

        return bond_update
