from pathlib import Path

from common.io_file import POSCAR
from common.logger import root_dir

if __name__ == '__main__':
    structure = POSCAR(fname=Path(f"{root_dir}/train_set/input/POSCAR_1-1")).to_structure()
    structure.find_neighbour_table(neighbour_num=12)

    bond_dist3d = structure.neighbour_table.dist3d
