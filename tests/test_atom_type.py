import itertools
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from common.io_file import POSCAR
from common.logger import root_dir

if __name__ == '__main__':
    structure = POSCAR(fname=Path(f"{root_dir}/train_set/input/POSCAR_1-1")).to_structure()
    structure.find_neighbour_table(neighbour_num=12)
    atom_type = structure.atoms.atom_type
    atom_type_index = [(index, item) for index, item in enumerate(atom_type)]
    atom_type_sort_index = sorted(atom_type_index, key=lambda x: x[1])
    atom_type_group = [list(item) for key, item in itertools.groupby(atom_type_sort_index, key=lambda x: x[1])]
    atom_type_group_index = [[item[0] for item in group] for group in atom_type_group]
    # atom_type_group_index_tensor = pad_sequence(atom_type_group_index)
    print()
