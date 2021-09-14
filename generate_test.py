#!/usr/bin/env python

#from logger import *
#from common.io_file import POSCAR
#from common.manager import DirManager
#
#template = POSCAR("input/POSCAR_1-1", action="r")
#TEMP_DM = DirManager("test/ori", "POSCAR", mol_index='37-38')
#files = [file.fname for file in TEMP_DM.all_files]
#vectors, indexs = [], []
#
#TEMP_DIR = os.path.join(current_dir, "temp")
#if not os.path.exists(TEMP_DIR):
#    os.mkdir(TEMP_DIR)
#
#for file in files:
#    index = file.split("_")[-1]
#    vector, coor = POSCAR(file, action="r").align(template)
#    #POSCAR(f"{TEMP_DIR}/POSCAR_ML_{index}", action="w").write(template, coor)
#    vectors.append(vector)
#    indexs.append(index)
#
#import random
#import numpy as np
#from pymatgen.io.vasp import Poscar
#####

import os
import random
from pathlib import Path
import numpy as np
from keras.models import load_model
from pymatgen.io.vasp import Poscar

from common.manager import DirManager
from common.io_file import POSCAR
from utils import normalize_coord
from logger import current_dir, logger
from load_yaml import ParameterManager
from generate_adsorbate import surface_cleave, random_molecule_getter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 屏蔽TF日志输出

Test_DIR = os.path.join(current_dir, "test")
Path(Test_DIR).mkdir(exist_ok=True)

test_ori_dir = os.path.join(Test_DIR, "ori-3")
Path(test_ori_dir).mkdir(exist_ok=True)

test_ML_dir = os.path.join(Test_DIR, "ML-3_1")
Path(test_ML_dir).mkdir(exist_ok=True)

PM = ParameterManager("test_111.yaml")
asf_CeO2_surf = surface_cleave(PM.MillerIndex)
latt = asf_CeO2_surf.slab.lattice.matrix[:2, :2]

logger.info(f"Generate {PM.TestNum} random POSCAR.")
for ii in range(PM.TestNum):
    molecule = random_molecule_getter()
    j = np.random.random((1, 2)) / 2
    Mat2 = np.dot(j, latt)
    shiftz = 2 * random.random() - 1
    CO_ads = asf_CeO2_surf.add_adsorbate(molecule, [Mat2[0, 0], Mat2[0, 1], PM.z_height + shiftz])
    for site in CO_ads.sites:
        site.properties['selective_dynamics'] = [True, True, True]
    p = Poscar(CO_ads)
    p.write_file(f"{test_ori_dir}/POSCAR_ori_{ii + 1}")

logger.info("Generate the ML trained POSCAR.")

kargs = {"style": "Slab+Mol",
         "mol_index": [36, 37],
         "anchor": 36,
         "ignore_mol": True,
         'expand': {'expand_z': {'boundary': 0.2, 'expand_num': 2, 'ignore_index': [37]}}}
template = POSCAR(fname=Path(current_dir) / "examples/CeO2_111/POSCAR_template").to_structure(**kargs)

test_ori_DM = DirManager(dname=Path(test_ori_dir), template=template, **kargs)
test_ori_coor = test_ori_DM.mcoords
test_ori_coor = normalize_coord(test_ori_coor)
print(test_ori_coor[:, 37, :])
test_ori_coor = test_ori_coor.reshape((test_ori_coor.shape[0], 38*3))

exit()
model = load_model("CeO2_111_CO_test.h5")
test_ML_coor = model.predict(test_ori_coor)
test_ML_coor = test_ML_coor.reshape((test_ML_coor.shape[0], 38, 3))
print(test_ML_coor[:, 37, :])
exit()





template = POSCAR("test/ori/POSCAR_ori_1", action="r")
lattice = template.latt

for ii, item in enumerate(test_ML_coor):
    coor = POSCAR.mcoord_to_coord(item, lattice, anchor=36, intercoor_index=[37])
    coor_new = coor - vectors[ii]
    POSCAR(f"{test_ML_dir}/POSCAR_ML_{indexs[ii]}", action="w").write(template, coor_new)
