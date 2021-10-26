import os
import random
import numpy as np
from pathlib import Path
from pymatgen.io.vasp import Poscar

from common.manager import ParameterManager
from common.logger import root_dir, logger

from generate_adsorbate import surface_cleave, random_molecule_getter

if __name__ == "__main__":

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    Test_DIR = os.path.join(root_dir, "test_set/guess/110")
    test_ori_dir = os.path.join(Test_DIR, "ori")

    Path(Test_DIR).mkdir(exist_ok=True)
    Path(test_ori_dir).mkdir(exist_ok=True)

    PM = ParameterManager("config/test_110.yaml")

    asf_CeO2_surf = surface_cleave(PM)
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