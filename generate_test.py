#!/usr/bin/env python

from logger import *
from common.structure import POSCAR
from main import DirManager

template = POSCAR("input/POSCAR_1-1", action="r")
TEMP_DM = DirManager("test/ori", "POSCAR", mol_index='37-38')
files = [file.fname for file in TEMP_DM.all_files]
vectors, indexs = [], []

TEMP_DIR = os.path.join(current_dir, "temp")
if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)

for file in files:
    index = file.split("_")[-1]
    vector, coor = POSCAR(file, action="r").align(template)
    #POSCAR(f"{TEMP_DIR}/POSCAR_ML_{index}", action="w").write(template, coor)
    vectors.append(vector)
    indexs.append(index)

import random
import numpy as np
from keras.models import load_model
from pymatgen.io.vasp import Poscar


from load_yaml import ParameterManager
from generate_adsorbate import surface_cleave, random_molecule_getter



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 屏蔽TF日志输出

Test_DIR = os.path.join(current_dir, "test")

if not os.path.exists(Test_DIR):
    os.mkdir(Test_DIR)

ori_test_dir = os.path.join(Test_DIR, "ori-2")
ML_test_dir = os.path.join(Test_DIR, "ML-2_2")

PM = ParameterManager("test_111.yaml")
asf_CeO2_surf = surface_cleave(PM.MillerIndex)
latt = asf_CeO2_surf.slab.lattice.matrix[:2, :2]

if not os.path.exists(ori_test_dir):
    os.mkdir(ori_test_dir)

for ii in range(PM.TestNum):
    molecule = random_molecule_getter()
    j = np.random.random((1, 2)) / 2
    Mat2 = np.dot(j, latt)
    shiftz = 2 * random.random() - 1
    CO_ads = asf_CeO2_surf.add_adsorbate(molecule, [Mat2[0, 0], Mat2[0, 1], PM.z_height + shiftz])
    for site in CO_ads.sites:
        site.properties['selective_dynamics'] = [True, True, True]
    p = Poscar(CO_ads)
    #p.write_file(f"{ori_test_dir}/POSCAR_ori_{ii+1}")

#exit()
test_ori_DM = DirManager("temp", "POSCAR", "37-38")
test_ori_coor = test_ori_DM.mcoords
test_ori_coor = test_ori_coor.reshape((test_ori_coor.shape[0], 38*3))

model = load_model("CeO2_111_CO_test.h5")
test_ML_coor = model.predict(test_ori_coor)
test_ML_coor = test_ML_coor.reshape((test_ML_coor.shape[0], 38, 3))

if not os.path.exists(ML_test_dir):
    os.mkdir(ML_test_dir)

template = POSCAR("test/ori/POSCAR_ori_1", action="r")
lattice = template.latt

for ii, item in enumerate(test_ML_coor):
    coor = POSCAR.mcoord_to_coord(item, lattice, anchor=36, intercoor_index=[37])
    coor_new = coor - vectors[ii]
    POSCAR(f"{ML_test_dir}/POSCAR_ML_{indexs[ii]}", action="w").write(template, coor_new)
