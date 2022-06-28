from torch import nn


class Model(nn.Module):
    def __init__(self, atom_in_fea_num, atom_out_fea_num, bond_in_fea_num, bond_out_fea_num): # Nx2, Nx12x3
        super(Model, self).__init__()
        self.atom_in_fea_num = atom_in_fea_num
        self.atom_out_fea_num = atom_out_fea_num
        self.bond_in_fea_num = bond_in_fea_num
        self.bond_out_fea_num = bond_out_fea_num

    def forward(self):
        pass