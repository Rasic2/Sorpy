#!/usr/bin/env python

import math
import numpy as np
from keras.models import load_model
from pymatgen.io.vasp import Poscar

from main import DirManager
from _logger import *
from load_yaml import ParameterManager
from generate_adsorbate import surface_cleave, random_molecule_getter


def array_to_POSCAR(array, POSCAR_t: str, fname):
    with open(POSCAR_t, "r") as f:
        cfg = f.readlines()
    head = cfg[:7]
    element_num = sum([int(ii) for ii in head[6].split()])
    selective_flag = cfg[7][0].lower() == "s"
    if selective_flag:
        head += cfg[7:9]
        TF = [item.split()[3:6] for item in cfg[9:9+element_num]]
    else:
        raise NotImplementedError("Selective Dynamics = False is not implemented")

    with open(fname, "w") as f:
        f.writelines(head)
        for ii, jj in zip(array, TF):
            item = f"{ii[0]:.6f} {ii[1]:.6f} {ii[2]:.6f} {jj[0]} {jj[1]} {jj[2]} \n"
            f.write(item)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 屏蔽TF日志输出

Test_DIR = os.path.join(current_dir, "test")

if not os.path.exists(Test_DIR):
    os.mkdir(Test_DIR)

ori_test_dir = os.path.join(Test_DIR, "ori")
ML_test_dir = os.path.join(Test_DIR, "ML")

PM = ParameterManager("test.yaml")
side_ref = PM.LatticeParameter / math.sqrt(2) * 2  # 111 surface <latt_abc>

asf_CeO2_surf = surface_cleave(PM.MillerIndex)
latt = asf_CeO2_surf.slab.lattice.matrix[:2, :2]
side_a = np.linspace(0, 0.5, num=math.ceil(10 * asf_CeO2_surf.slab.lattice.a / side_ref))
side_b = np.linspace(0, 0.5, num=math.ceil(10 * asf_CeO2_surf.slab.lattice.b / side_ref))
side_arr = np.linspace(0, 0.5, num=10)

if not os.path.exists(ori_test_dir):
    os.mkdir(ori_test_dir)

for ii in range(PM.TestNum):
    molecule = random_molecule_getter()
    CO_ads = asf_CeO2_surf.add_adsorbate(molecule, [0, 0, PM.z_height])
    for site in CO_ads.sites:
        site.properties['selective_dynamics'] = [True, True, True]
    p = Poscar(CO_ads)
    p.write_file(f"{ori_test_dir}/POSCAR_ori_{ii+1}")

test_ori_DM = DirManager("test/ori", "POSCAR", "37-38")
test_ori_coor = test_ori_DM.coords
test_ori_coor = test_ori_coor.reshape((test_ori_coor.shape[0], 38*3))

model = load_model("CeO2_111_CO.h5")
test_ML_coor = model.predict(test_ori_coor)
test_ML_coor = test_ML_coor.reshape((test_ML_coor.shape[0], 38, 3))

if not os.path.exists(ML_test_dir):
    os.mkdir(ML_test_dir)

for ii, item in enumerate(test_ML_coor):
    array_to_POSCAR(item, f"{ori_test_dir}/POSCAR_ori_1", f"{ML_test_dir}/POSCAR_ML_{ii+1}")
