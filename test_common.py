import os

from common.operate import Operator as op
from common.io_file import POSCAR, CONTCAR
from common.manager import FileManager, DirManager
from common.model import Model

from common.logger import current_dir
from pathlib import Path
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

def test_operator():
    kargs={"style":"Slab+Mol", "mol_index": [36,37], "anchor": 36, "ignore_mol": True}
    s1 = POSCAR(fname=Path(current_dir)/"examples/CeO2_111/POSCAR_template").to_structure(**kargs)
    s2 = POSCAR(fname=Path(current_dir)/"test_set/ori/POSCAR_ori_1").to_structure(**kargs)
    template = s1.coords
    coords = s2.coords
    s1.find_nearest_neighbour_table()
    s2.find_nearest_neighbour_table()
    #print(s1.NNT)
    #print(op.dist(s1, s2))
    #print(s2.coords.cart_coords-s1.coords.cart_coords)
    print(op.align(s1, s2).coords.cart_coords-s1.coords.cart_coords)
    #print(s1.bonds)
    #print(op.align(s1, s2))
    #print(op.__dict__)
    #op._Operator__tailor_atom_order(s1, s2)

def test_main():

    kargs = {"style":"Slab+Mol",
             "mol_index": [36,37],
             "anchor": 36,
             "ignore_mol": True,
             'expand':{'expand_z':{'boundary': 0.2, 'expand_num': 2, 'ignore_index': [37]}}}

    template = POSCAR(fname=Path(current_dir)/"examples/CeO2_111/POSCAR_template").to_structure(**kargs)

    input_dm = DirManager(dname=Path(current_dir)/"input", template=template, **kargs)
    output_dm = DirManager(dname=Path(current_dir)/"output", template=template, **kargs)

    data_input, data_output = np.copy(input_dm.mcoords), np.copy(output_dm.mcoords)

    from keras import models, layers
    model = models.Sequential()
    model.add(layers.Dense(1024, activation='relu', input_shape=(38 * 3,)))
    model.add(layers.Dense(114))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
    m = Model(model, data_input, data_output, normalization="mcoord", expand=kargs['expand'])
    print(m("hold out"))

def test_mass_center():
    kargs = {"style": "Slab+Mol", "mol_index": [36, 37], "anchor": 36, "ignore_mol": True}

    template = POSCAR(fname=Path(current_dir) / "examples/CeO2_111/POSCAR_template").to_structure(**kargs)
    p2 = CONTCAR(fname=Path(current_dir) / "output/CONTCAR_1-1").to_structure(**kargs)
    print(template.mass_center)
    #print(template.slab.mass_center)
    #print(p2.slab.mass_center)

    #print(np.where(np.abs(p2.frac_coords-template.frac_coords)>0.5))
    #print((p2.frac_coords-template.frac_coords)[5])

if __name__ == "__main__":
    #test_filemanager()
    #test_dirmanager()
    #test_operator()
    test_main()
    #test_mass_center()
    pass
