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

    mode = '110'
    kargs = {"style": "Slab+Mol",
             "mol_index": [48, 49],
             "anchor": 48}

    template_111 = Path(root_dir) / "examples/CeO2_111/POSCAR_template"
    template_110 = Path(root_dir) / "examples/CeO2_110/POSCAR_template"
    template = POSCAR(fname=template_110).to_structure(**kargs)
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

    if mode == '110':
        rotate = op.getRotate(mode)
        s_template_111 = POSCAR(fname=template_111).to_structure(style="Slab+Mol", mol_index=[36, 37], anchor=36)
        m_template_111 = s_template_111.create_mol()
        template_111_vcoord, _ = s_template_111.vcoord(m_template_111, 5.0, None)

        # search the transform vector
        temp = np.dot(test_input[0, :-2], rotate)
        index = []
        for index_i, i in enumerate(template_111_vcoord[:-2]):
            if index_i == 0:
                continue
            dist = [(np.linalg.norm(i-j), index_j) for index_j, j in enumerate(temp) if index_j != 0]
            min_dist = min(dist, key=lambda x:x[0])
            if min_dist[0] < 0.1:
                index.append((index_i, min_dist[1]))
        index_ref, index_cur = map(list, zip(*index))
        index_ref_miss = [i+1 for i in range(template_111_vcoord.shape[0]-3) if i+1 not in index_ref]
        index_cur_inv = [index_cur.index(i+1)+1 for i in range(len(index_cur))]

        # reconstruct the "test_input"
        new_test_input = np.zeros((test_input.shape[0], 10, 3))  # init with the zero-matrix
        new_test_input[:, 0] = test_input[:, 0]  # store the Ce-frac
        new_test_input[:, 1:8] = template_111_vcoord[1:8]
        new_test_input[:, index_ref] = np.dot(test_input[:, index_cur], rotate)  # respond the O-vector <110-surface to 111-surface>
        new_test_input[:, -2:] = test_input[:, -2:]  # store the CO-mcoord
        test_input = np.copy(new_test_input)
    else:
        rotate = None

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

    if mode == "110":
        test_output = np.delete(test_output, index_ref_miss, axis=1)  # delete the extra-column
        index_cur_inv = [0] + index_cur_inv + [7, 8]
        test_output = test_output[:, index_cur_inv]

    test_output = Model.decode_vcoord(ori_input_dm, test_output, orders, template.lattice, rotate)

    for index in range(test_output.shape[0]):
        test_output[index] -= trans_vectors[index]

    style, elements, lattice, TF = template.style, template.elements, template.lattice, template.TF
    for index, item in enumerate(test_output):
        s = Structure(style=style, elements=elements, coords=Coordinates(frac_coords=item, lattice=lattice),
                      lattice=lattice, TF=TF)
        s.write_to_POSCAR(fname=f'{test_dir}/POSCAR_ML_{index + 1}')
