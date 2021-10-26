import logging
import os

from common.operate import Operator as op
from common.io_file import POSCAR, CONTCAR
from common.manager import FileManager, DirManager
from common.model import Model

from common.logger import root_dir, logger
from pathlib import Path
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def test_dist():
    logger.setLevel(logging.ERROR)

    with open(Path(root_dir) / "test_set/summary/ori/cpu_spent") as f, open(Path(root_dir) / "test_set/summary/ML_v2/cpu_spent") as g:
        ori_cpu = f.readlines()
        vcoord_cpu = g.readlines()

    ori_cpu = np.array([float(line.split()[6]) for line in ori_cpu])
    vcoord_cpu = np.array([float(line.split()[6]) for line in vcoord_cpu])

    findlist = np.where(ori_cpu<vcoord_cpu)[0]

    for i in range(50):
        # print(i+1)
        poscar = POSCAR(fname=f"{root_dir}/tests/result/POSCAR_{i+1}").to_structure(style="Slab+Mol", mol_index=[36, 37], anchor=36)
        # ori = CONTCAR(fname=f"{root_dir}/test_set/guess/ori/POSCAR_ori_{i+1}").to_structure(style="Slab+Mol", mol_index=[36, 37], anchor=36)
        ori = CONTCAR(fname=f"{root_dir}/test_set/guess/ML-2_2/POSCAR_ML_{i + 1}").to_structure(style="Slab+Mol",
                                                                                              mol_index=[36, 37],
                                                                                              anchor=36)
        # frac_diff = poscar - contcar
        # frac_diff = np.where(frac_diff > 0.5, frac_diff - 1, frac_diff)
        # frac_diff = np.where(frac_diff < -0.5, frac_diff + 1, frac_diff)
        # cart_diff = np.dot(frac_diff, poscar.structure.lattice.matrix)

        CO_frac = np.copy(poscar.molecule.frac_coords)
        CO_frac[:, :2] = np.where(CO_frac[:, :2]>0.5, CO_frac[:, :2]-0.5, CO_frac[:, :2])

        ori_CO_frac = np.copy(ori.molecule.frac_coords)
        ori_CO_frac[:, :2] = np.where(ori_CO_frac[:, :2]>0.5, ori_CO_frac[:, :2]-0.5, ori_CO_frac[:, :2])

        # frac_diff = poscar
        # cart_diff = np.dot(frac_diff.frac_coords, poscar.lattice.matrix)
        # if i in findlist:
        if i+1 == 16:
            input_dm = DirManager(dname=Path(root_dir)/"train_set/input", style="Slab+Mol", mol_index=[36, 37], anchor=36)
            output_dm = DirManager(dname=Path(root_dir) / "train_set/output", style="Slab+Mol", mol_index=[36, 37],
                                  anchor=36)
            shortest_pos = []
            ori_shortest_pos = []
            for si, so in zip(input_dm.structures, output_dm.structures):
                # ori_shortest_pos.append((si.name.name, np.linalg.norm(np.dot(ori_CO_frac - si.molecule.frac_coords, si.lattice.matrix))))
                shortest_pos.append((so.name.name, np.linalg.norm(np.dot(CO_frac - so.molecule.frac_coords, so.lattice.matrix))))
            # print(i + 1, sorted(ori_shortest_pos, key=lambda x: abs(x[1]))[0])
            print(i+1, sorted(shortest_pos, key=lambda x : abs(x[1]))[0])
            print()
        # print(cart_diff[-2:])
        # diff.append((i+1, np.max(cart_diff)))
        # print()
    # for i in sorted(diff, key=lambda x : x[1])[::-1]:
    #     print(i)
    pass

def test_topdist():
    logger.setLevel(logging.ERROR)
    output_dm = DirManager(dname=Path(root_dir)/"train_set/output", style="Slab+Mol", mol_index=[36, 37], anchor=36)
    for si in output_dm.structures:
        print(si.name.name, np.min(si.molecule.cart_coords[:, 2]) - np.max(si.slab.cart_coords[:, 2]))
    pass

def test_poscar():
    p1 = POSCAR(fname=f"{root_dir}/input/POSCAR_1-1")
    p2 = CONTCAR(fname=f"{root_dir}/output/CONTCAR_1-1")
    print(p1.ftype)
    print(p2.ftype)
    print(p1 - p2)

def test_structure():
    template = POSCAR(fname=Path(root_dir) / "examples/CeO2_111/POSCAR_template").to_structure(style="Slab+Mol", mol_index=[36,37], anchor=36)
    m_template = template.create_mol()
    s1 = POSCAR(fname=Path(root_dir) / "train_set/input/POSCAR_1-1").to_structure(style="Slab+Mol", mol_index=[36,37], anchor=36)
    s2 = POSCAR(fname=Path(root_dir) / "train_set/output/CONTCAR_1-1").to_structure(style="Slab+Mol", mol_index=[36,37])
    # print(op.dist(self, s2))
    # print(self.kargs)
    print(s1.vcoord(m_template))

def test_filemanager():
    fm = FileManager(fname=Path(root_dir)/"input/POSCAR_3-1", style="Slab+Mol", mol_index=[36, 37])
    print(fm.file)
    print(fm.index)
    print(fm.structure.molecule.inter_coords)

def test_dirmanager():
    dm = DirManager(dname=Path(root_dir)/"train_set/input-2", style="Slab+Mol", mol_index=[36, 37], anchor=36)
    template = POSCAR(fname=Path(root_dir) / "examples/CeO2_111/POSCAR_template").to_structure(style="Slab+Mol",
                                                                                               mol_index=[36, 37],
                                                                                               anchor=36)
    #print(dm.single_file("POSCAR_1-1"))
    #print(len(dm.all_files))
    #print(np.array(dm.coords)[0][0])
    #print(np.array(dm.coords))
    #print(dm.frac_coords)
    # print(dm.inter_coords.shape)
    print(dm.vcoords(template=template))
    #for file in dm.all_files:
    #    print(file.structure.molecule.inter_coords[0][2])
    #    exit()
    #print(fm.structure.molecule.inter_coords)

def test_operator():
    kargs={"style":"Slab+Mol", "mol_index": [36,37], "anchor": 36, "ignore_mol": True}
    s1 = POSCAR(fname=Path(root_dir)/"examples/CeO2_111/POSCAR_template").to_structure(**kargs)
    s2 = POSCAR(fname=Path(root_dir)/"train_set/xinput-m_align/POSCAR_4-31-1").to_structure(**kargs)
    #template = self.coords
    #coords = s2.coords
    #self.find_nearest_neighbour_table()
    #s2.find_nearest_neighbour_table()
    #print(self.NNT)
    #print(op.dist(self, s2))
    #print(s2.coords.cart_coords-self.coords.cart_coords)
    print(op.align_structure(s1, s2))
    #print(self.bonds)
    #print(op.align_structure(self, s2))
    #print(op.__dict__)
    #op._Operator__tailor_atom_order(self, s2)

def test_main():

    kargs = {"style":"Slab+Mol",
             "mol_index": [36,37],
             "anchor": 36,
             "ignore_mol": True,
             'expand':{'expand_z':{'boundary': 0.2, 'expand_num': 2, 'ignore_index': [37]}}}

    template = POSCAR(fname=Path(root_dir)/"examples/CeO2_111/POSCAR_template").to_structure(**kargs)

    input_dm = DirManager(dname=Path(root_dir)/"input", template=template, **kargs)
    output_dm = DirManager(dname=Path(root_dir)/"output", template=template, **kargs)

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

    template = POSCAR(fname=Path(root_dir) / "examples/CeO2_111/POSCAR_template").to_structure(**kargs)
    p2 = CONTCAR(fname=Path(root_dir) / "output/CONTCAR_1-1").to_structure(**kargs)
    print(template.mass_center)
    #print(template.slab.mass_center)
    #print(p2.slab.mass_center)
    #print(np.where(np.abs(p2.frac_coords-template.frac_coords)>0.5))
    #print((p2.frac_coords-template.frac_coords)[5])

if __name__ == "__main__":
    # test_dist()
    test_topdist()
    #test_filemanager()
    # test_dirmanager()
    # test_operator()
    #test_main()
    # test_structure()
    #test_mass_center()
    pass
