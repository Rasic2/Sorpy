import os

from common.operate import Operator as op
from common.io_file import POSCAR, CONTCAR
from common.manager import FileManager, DirManager
from common.model import Model
from common.structure import Molecule
from common.base import Coordinates

from common.logger import root_dir
from pathlib import Path
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def create_mol(s, cut_radius=5.0):
    max_length = cut_radius
    center = None
    for index in s.mol_index:
        for atom in s.NNT.index(index):
            if atom[0].element.formula == "Ce" and atom[1] <= max_length:
                center = atom
                max_length = atom[1]

    if center is None:
        raise TypeError("Can't find the Ce element in the cut_radius.")

    elements = [atom[0].element for atom in s.bonds.index(center[0].order)]
    orders = [atom[0].order for atom in s.bonds.index(center[0].order)]
    coords = [atom[0].coord.frac_coords for atom in s.bonds.index(center[0].order)]

    elements.insert(0, center[0].element)
    orders.insert(0, center[0].order)
    coords.insert(0, center[0].coord.frac_coords)
    coords = Coordinates(frac_coords=np.array(coords), lattice=center[0].coord.lattice)

    return Molecule(elements=elements, orders=orders, coords=coords, anchor=center[0].order)

def test_NNT():
    cut_radius = 5.0
    for file in os.listdir(f"{root_dir}/train_set/input/"):
        print(file)
        s1 = POSCAR(fname=f"{root_dir}/train_set/input/{file}").to_structure(style="Slab+Mol", mol_index=[36,37])
        s1.find_nearest_neighbour_table(cut_radius=cut_radius)
        m1 = create_mol(s1)
        for item in m1.inter_coords:
            print(item)
        print()

    #s2 = CONTCAR(fname=f"{root_dir}/train_set/output/CONTCAR_1-1").to_structure(style="Slab+Mol", mol_index=[36,37])
    #s2.find_nearest_neighbour_table(cut_radius=cut_radius)
    #m2 = create_mol(s2)

    #for i, j in zip(m1.inter_coords, m2.inter_coords):
    #    print(i)
    #    print(j)
    #    print()


if __name__ == "__main__":
    test_NNT()