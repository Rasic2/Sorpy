import numpy as np
from pathlib import Path

from common.logger import root_dir
from common.io_file import POSCAR
from common.base import Coordinates
from common.structure import Structure
from common.operate import Operator as op
from common.manager import DirManager
from common.model import Model

if __name__ == "__main__":

    kargs = {"style": "Slab+Mol",
             "mol_index": [36, 37],
             "anchor": 36}

    template_file = Path(root_dir) / "examples/CeO2_111/POSCAR_template"
    template = POSCAR(fname=template_file).to_structure(**kargs)
    ori_dir = Path(root_dir) / "test_set/guess/ori"
    test_dir = Path(root_dir) / "test_set/guess/ML-test"

    ori_input_dm = DirManager(dname=ori_dir, template=template, **kargs)
    test_input_frac = np.array([file.structure.molecule.frac_coords for file in ori_input_dm.all_files])

    shiftZ = np.array([0.723895 - np.min(item[:, 2]) for item in test_input_frac])  # 将CO分子最低点拉至 0.723895 水平线

    for i, j in zip(test_input_frac, shiftZ):
        i[:, 2] = j + i[:, 2]

    test_input, orders = ori_input_dm.vcoords()
    test_input[:, 8] = test_input_frac[:, 0]
    test_input = op.normalize_vcoord(test_input)
    tcoord = np.copy(test_input[:, 0])  # record the Ce fractional coordinate
    test_input[:, 0] = 0.0  # use the <fractional difference> to train the model

    test_input, trans_vectors = op.find_trans_vector(test_input, anchor=8)
    test_input = test_input.reshape((test_input.shape[0], test_input.shape[1] * test_input.shape[2]))

    from keras.models import load_model

    model = load_model("results/intercoord_3layer.h5")
    test_output = model.predict([test_input[:, :24], test_input[:, 24:]])
    test_output = test_output.reshape(50, 10, 3)
    test_output[:, 0] = test_output[:, 0] + tcoord

    test_output = Model.decode_vcoord(ori_dir, test_output, orders, template.lattice)
    for index in range(test_output.shape[0]):
        test_output[index] -= trans_vectors[index]

    # print(test_input_frac[0, -2])
    # print(test_output[0, -2])
    # diff_frac = test_output-test_input_frac
    #
    # diff_frac = op.pbc(diff_frac)
    # delta = np.array([np.linalg.norm(item[-2:]) for item in diff_frac])
    # print(delta)
    # if len(np.where(delta>0.1)[0]):
    #     print(len(np.where(delta>0.1)[0]))
    #     test_output[:, :-2] = test_input_frac[:, :-2]
    # else:
    #     break
    # print(np.sum(np.sum(diff_frac[:, -2:], axis=2), axis=1))
    # exit()
    # print(np.sum(np.sum(test_output-test_input_frac, axis = 2), axis = 1))
    # exit()

    style, elements, lattice, TF = template.style, template.elements, template.lattice, template.TF
    for index, item in enumerate(test_output):
        s = Structure(style=style, elements=elements, coords=Coordinates(frac_coords=item, lattice=lattice),
                      lattice=lattice, TF=TF)
        s.write_to_POSCAR(fname=f'{test_dir}/POSCAR_ML_{index + 1}')

    # ori_input_dm = DirManager(dname=test_dir, template=template, **kargs)
    #
    # if len(np.where(delta>0.1)[0]) == 0:
    #     logger.info("Iteration reached the accuracy")
    #     break
