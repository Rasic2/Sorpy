from common.operate import Operator as op
from common.io_file import POSCAR, CONTCAR
from common.manager import FileManager, DirManager
from logger import current_dir
from pathlib import Path
import numpy as np

def test_poscar():
    p1 = POSCAR(fname=f"{current_dir}/input/POSCAR_1-1")
    p2 = CONTCAR(fname=f"{current_dir}/output/CONTCAR_1-1")
    print(p1.ftype)
    print(p2.ftype)
    print(p1 - p2)

def test_structure():
    s1 = POSCAR(fname=f"{current_dir}/input/POSCAR_1-1").to_structure(style="Slab+Mol", mol_index=[36,37])
    s2 = POSCAR(fname=f"{current_dir}/output/CONTCAR_1-1").to_structure(style="Slab+Mol", mol_index=[36,37])
    print(op.dist(s1, s2))

def test_filemanager():
    fm = FileManager(fname=Path(current_dir)/"input/POSCAR_3-1", style="Slab+Mol", mol_index=[36, 37])
    print(fm.file)
    print(fm.index)
    print(fm.structure.molecule.inter_coords)

def test_dirmanager():
    dm = DirManager(dname=Path(current_dir)/"input", style="Slab+Mol", mol_index=[36, 37])
    #print(dm.single_file("POSCAR_1-1"))
    #print(len(dm.all_files))
    #print(np.array(dm.coords)[0][0])
    #print(np.array(dm.coords))
    #print(dm.frac_coords)

    print(dm.inter_coords.shape)
    #for file in dm.all_files:
    #    print(file.structure.molecule.inter_coords[0][2])
    #    exit()
    #print(fm.structure.molecule.inter_coords)

if __name__ == "__main__":
    #test_filemanager()
    test_dirmanager()
    pass
