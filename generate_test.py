import os
import random
import numpy as np
from pathlib import Path
from keras.models import load_model
from pymatgen.io.vasp import Poscar

from common.manager import DirManager, ParameterManager
from common.io_file import POSCAR
from common.model import Model
from common.structure import Structure
from common.base import Coordinates
from common.operate import Operator as op
from common.utils import normalize_coord
from common.logger import current_dir, logger

from generate_adsorbate import surface_cleave, random_molecule_getter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

Test_DIR = os.path.join(current_dir, "test_set")
Path(Test_DIR).mkdir(exist_ok=True)

test_ori_dir = os.path.join(Test_DIR, "ori-3")
Path(test_ori_dir).mkdir(exist_ok=True)

test_ML_dir = os.path.join(Test_DIR, "ML-3_1")
Path(test_ML_dir).mkdir(exist_ok=True)

PM = ParameterManager("configuration/test_111.yaml")
asf_CeO2_surf = surface_cleave(PM.MillerIndex)
latt = asf_CeO2_surf.slab.lattice.matrix[:2, :2]

logger.info(f"Generate {PM.TestNum} random POSCAR.")
for ii in range(PM.TestNum):
    molecule = random_molecule_getter()
    j = np.random.random((1, 2))
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

logger.info("Step 0. Get the input data from the directory.")
test_ori_DM = DirManager(dname=Path(test_ori_dir), template=template, **kargs)
test_ori_coor = test_ori_DM.mcoords

logger.info("Step 1. Normalization the data.")
test_ori_coor = normalize_coord(test_ori_coor)

logger.info("Step 2. Translate the molecule and record the vectors.")
test_ori_coor, trans_vectors = op.find_trans_vector(test_ori_coor)
test_ori_coor = test_ori_coor.reshape((test_ori_coor.shape[0], 38 * 3))

logger.info("Step 3. Predict the input data from the Model.")
model = load_model("results/CeO2_111_CO_test.h5")
test_ML_coor = model.predict(test_ori_coor)
test_ML_coor = test_ML_coor.reshape((test_ML_coor.shape[0], 38, 3))
for index in range(test_ML_coor.shape[0]):
    test_ML_coor[index, 36, :] -= trans_vectors[index]

logger.info("Step 4. Decode the mcoord to the POSCAR fractional coordinates.")
test_ML_coor = Model.decode_mcoord(test_ML_coor, template.lattice)

logger.info("Step 5. Write the predict coordinates to the POSCAR.")
style, elements, lattice, TF = template.style, template.elements, template.lattice, template.TF
for index, item in enumerate(test_ML_coor):
    s = Structure(style=style, elements=elements, coords=Coordinates(frac_coords=item, lattice=lattice),
                  lattice=lattice, TF=TF)
    s.write_to_POSCAR(fname=f'{test_ML_dir}/POSCAR_ML_{index + 1}')
