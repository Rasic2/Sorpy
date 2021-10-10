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
    test_input, orders = ori_input_dm.vcoords()
    test_input = op.normalize_vcoord(test_input)
    test_input, trans_vectors = op.find_trans_vector(test_input, anchor=8)
    test_input = test_input.reshape((test_input.shape[0], 30))

    from keras.models import load_model

    model = load_model("results/intercoord_3layer.h5")
    test_output = model.predict([test_input[:, :24], test_input[:, 24:]])
    test_output = test_output.reshape(50, 10, 3)

    test_output = Model.decode_vcoord(ori_dir, test_output, orders, template.lattice)
    for index in range(test_output.shape[0]):
        test_output[index] -= trans_vectors[index]

    style, elements, lattice, TF = template.style, template.elements, template.lattice, template.TF
    for index, item in enumerate(test_output):
        s = Structure(style=style, elements=elements, coords=Coordinates(frac_coords=item, lattice=lattice),
                      lattice=lattice, TF=TF)
        s.write_to_POSCAR(fname=f'{test_dir}/POSCAR_ML_{index + 1}')
