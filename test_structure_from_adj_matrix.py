from pathlib import Path

from common.io_file import POSCAR
from common.logger import root_dir
from common.structure import Structure

structure = POSCAR(fname=Path(f"{root_dir}/train_set/input/POSCAR_1-1")).to_structure(style="Slab")
structure.find_neighbour_table(neighbour_num=12)

bond_dist3d = structure.neighbour_table.dist3d

structure_new = Structure.from_adj_matrix(structure, bond_dist3d=bond_dist3d, known_first_order=0)
structure_new.to_POSCAR(fname="POSCAR_test")