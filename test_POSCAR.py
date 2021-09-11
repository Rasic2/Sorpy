from common.operate import Operator as op
from common.io_file import POSCAR, CONTCAR
from common.manager import FileManager
from logger import current_dir

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
    #print(f"{current_dir}/input/POSCAR_1-1")
    fm = FileManager(fname=f"{current_dir}/input/POSCAR_1-1", style="Slab+Mol")
    print(fm.file)
    print(fm.structure)

if __name__ == "__main__":
    test_filemanager()
