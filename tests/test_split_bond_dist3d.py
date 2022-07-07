from collections import defaultdict
from pathlib import Path

import numpy as np

from common.io_file import POSCAR
from common.logger import root_dir

if __name__ == '__main__':
    structure = POSCAR(fname=Path(f"{root_dir}/train_set/input/POSCAR_1-1")).to_structure()
    structure.find_neighbour_table(neighbour_num=12)
    atom_type = np.array(structure.atoms.atom_type)
    adj_matrix_tuple = np.reshape(structure.neighbour_table.index_tuple, newshape=(-1, 2))
    bond_dist3d = structure.neighbour_table.dist3d
    # dtype=[('first', '<U5'), ('second', '<U5')]
    result = atom_type[adj_matrix_tuple]
    result_new = np.char.add(result[:, 0], "-")
    result_new = np.char.add(result_new, result[:, 1])
    result_new_sort = np.sort(result_new)
    result_new_sort_arg = np.argsort(result_new)

    result_default = defaultdict(list)
    for item, arg in zip(result_new_sort, result_new_sort_arg):
        result_default[f"{item}"].append(arg)
    print()
