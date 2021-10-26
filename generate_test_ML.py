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
             "mol_index": [48, 49],
             "anchor": 48}

    template_file = Path(root_dir) / "examples/CeO2_110/POSCAR_template"
    template = POSCAR(fname=template_file).to_structure(**kargs)
    ori_dir = Path(root_dir) / "test_set/guess/110/ori"
    test_dir = Path(root_dir) / "test_set/guess/110/ML-test"
    Path(test_dir).mkdir(exist_ok=True)

    ori_input_dm = DirManager(dname=ori_dir, template=template, **kargs)
    # test_input_frac = np.array([file.structure.molecule.frac_coords for file in ori_input_dm.all_files])
    #
    # shiftZ = np.array([0.723895 - np.min(item[:, 2]) for item in test_input_frac])  # 将CO分子最低点拉至 0.723895 水平线
    #
    # for i, j in zip(test_input_frac, shiftZ):
    #     i[:, 2] = j + i[:, 2]

    test_input, orders = ori_input_dm.vcoords()

    # 111-vector
    # [ 1.92686215e+00,   1.11247889e+00,  -7.86646043e-01]           20
    # [-1.92686985e+00,   1.11247889e+00,  -7.86646043e-01]           22
    # [-3.85373200e-06,  -2.22495111e+00,  -7.86646043e-01]           23
    # [ 0.00000000e+00,   0.00000000e+00,  -2.35990981e+00]           24
    # [ 3.85373200e-06,   2.22495111e+00,   7.86646043e-01]           32  remove
    # [ 1.92686985e+00,  -1.11247889e+00,   7.86646043e-01]           35
    # [-1.92686215e+00,  -1.11247889e+00,   7.86646043e-01]           33

    # 110-vector
    # [ 0.      ,  -1.3625,  -1.92685829]                     21
    # [ 1.926866,  -1.3625,   0.        ]                     25
    # [-1.926866,  -1.3625,   0.        ]                     27
    # [ 1.926866,   1.3625,   0.        ]                     32
    # [-1.926866,   1.3625,   0.        ]                     34
    # [ 0.      ,   1.3625,  -1.92685829]]                    44

    # reconstruct the test_input <(9,3) --> (10,3)>
    A = [[ 0.00000000e+00,   0.00000000e+00,  -2.35990981e+00],
         [ 1.92686985e+00,  -1.11247889e+00,   7.86646043e-01],
         [-1.92686215e+00,  -1.11247889e+00,   7.86646043e-01]]  # three points in 111-surface

    B = [[ 0.,         1.3625,  -1.92685829],
         [-1.926866,  -1.3625,   0.        ],
         [ 1.926866,  -1.3625,   0.        ]]  # three points in 110-surface

    rotate = np.dot(np.linalg.inv(np.array(B)), np.array(A))  # rotate matrix
    new_test_input = np.zeros((test_input.shape[0], 10, 3))  # init with the zero-matrix
    new_test_input[:, 0] = test_input[:, 0]  # store the Ce-frac
    new_test_input[:, [1, 2, 3, 4, 6, 7]] = np.dot(test_input[0][[5, 4, 1, 6, 3, 2]], rotate)  # responding the O-vector
    new_test_input[:, 5] = [ 3.85373200e-06,   2.22495111e+00,   7.86646043e-01]
    new_test_input[:, -2:] = test_input[:, -2:]  # store the CO-mcoord
    test_input = np.copy(new_test_input)
    orders = [[item[index] for index in [0, 5, 4, 1, 6, 3, 2]] for item in orders]

    # test_input[:, 8] = test_input_frac[:, 0]
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

    test_output[:, 1:8] = np.dot(test_output[:, 1:8], np.linalg.inv(rotate))  # TODO: bugfix
    test_output = np.delete(test_output, [5], axis=1)  # delete the extra-column

    test_output = Model.decode_vcoord(ori_input_dm, test_output, orders, template.lattice)
    for index in range(test_output.shape[0]):
        test_output[index] -= trans_vectors[index]

    style, elements, lattice, TF = template.style, template.elements, template.lattice, template.TF
    for index, item in enumerate(test_output):
        s = Structure(style=style, elements=elements, coords=Coordinates(frac_coords=item, lattice=lattice),
                      lattice=lattice, TF=TF)
        s.write_to_POSCAR(fname=f'{test_dir}/POSCAR_ML_{index + 1}')
