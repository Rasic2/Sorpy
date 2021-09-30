import os
import numpy as np
from pathlib import Path
from keras.models import load_model

from common.io_file import POSCAR
from common.model import Model
from common.structure import Structure
from common.base import Coordinates
from common.logger import root_dir


# TODO: from the 111 training model produce the 100 result, may need to rotate the frac_coord

if __name__ == "__main__":

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    kargs = {"style": "Slab+Mol",
             "mol_index": [48, 49],
             "anchor": 48,
             "ignore_mol": True,
             'expand': {'expand_z': {'boundary': 0.2, 'expand_num': 2, 'ignore_index': [49]}}}

    template = POSCAR(fname=Path(root_dir) / "examples/CeO2_110/POSCAR_template").to_structure(**kargs)

    test_110_dir = Path(f"{root_dir}/train_set/guess/110")
    test_110 = POSCAR(fname=f"{test_110_dir}/POSCAR_1-1").to_structure(**kargs)
    mcoord = np.array(test_110.molecule.inter_coords[0][2]) - [1.142, 0, 0]

    # construct the Ce12O24C1O1 structure
    input_mcoord = np.concatenate((test_110.frac_coords[4:16].reshape((12, 3)), test_110.frac_coords[24:48].reshape((24, 3)), test_110.frac_coords[48].reshape((1, 3)), mcoord.reshape((1, 3))), axis=0)
    input_mcoord[:37, :] = 0
    input_mcoord = input_mcoord.reshape((1, 38 * 3))

    model = load_model("results/model-3layer-lr-1e-05.h5")
    output_mcoord = model.predict(input_mcoord).reshape((1, 38, 3))
    output_mcoord[:, :12] = output_mcoord[:, :12] + test_110.frac_coords[4:16]
    output_mcoord[:, 12:37] = output_mcoord[:, 12:37] + test_110.frac_coords[24:49]

    output_mcoord = Model.decode_mcoord(output_mcoord, lattice=template.lattice)
    output_frac = np.concatenate((test_110.frac_coords[:4].reshape((4, 3)), output_mcoord[:, :12].reshape((12, 3)), test_110.frac_coords[16:24].reshape((8, 3)), output_mcoord[:, 12:].reshape((26, 3))), axis=0)
    style, elements, lattice, TF = template.style, template.elements, template.lattice, template.TF

    s = Structure(style=style, elements=elements, coords=Coordinates(frac_coords=output_frac, lattice=lattice), lattice=lattice, TF=TF)
    s.write_to_POSCAR(fname=f'POSCAR_ML_110-1')
