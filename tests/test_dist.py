from pathlib import Path
import numpy as np
from common.manager import DirManager
from common.logger import root_dir
from common.io_file import POSCAR
from common.operate import Operator as op
from collections import Counter

if __name__ == "__main__":

    input_dir = Path(root_dir) / "train_set" / "input"
    output_dir = Path(root_dir) / "train_set" / "output"

    kargs = {"style": "Slab+Mol",
             "mol_index": [36, 37],
             "anchor": 36,
             "ignore_mol": True,
             'expand': {'expand_z': {'boundary': 0.2, 'expand_num': 2, 'ignore_index': [37]}}}

    template = POSCAR(fname=Path(root_dir) / "examples/CeO2_111/POSCAR_template").to_structure(**kargs)

    input_dm = DirManager(dname=input_dir, template=template, **kargs)
    output_dm = DirManager(dname=output_dir, template=template, **kargs)
    input_s = input_dm.structures
    output_s = output_dm.structures
    total_orders = []
    for si, sj in zip(input_s, output_s):
        dists = []
        for key, value in op.dist(si, sj).items():
            dists.append((key.order, list(value.items())[0][0].order, list(value.items())[0][1]))
        orders = [item[0] for item in sorted(dists, key=lambda x: x[2])[::-1][:5]]
        total_orders.append(orders)
    total_orders = sum(total_orders, [])
    print(Counter(total_orders))
    #ori_DM = DirManager("test_set/ori", "POSCAR", mol_index="37-38")
    #ML_DM = DirManager("test_set/ML-2", "POSCAR", mol_index="37-38")

    #ori_DM = DirManager("input-test_set", "POSCAR", mol_index="37-38")
    #ML_DM = DirManager("output-test_set", "CONTCAR", mol_index="37-38")

    #ori_coor = ori_DM.coords
    #ML_coor = ML_DM.coords

    #CT = CoorTailor(ori_coor, ML_coor, 2, intercoor_index=[])
    #CT._expand_xy()
    #CT.run(boundary=0, num=0)
    #ori_coor_2, ML_coor_2 = CT.input_arr, CT.output_arr

    #for i, j, k, l in zip(ori_coor, ML_coor, ori_coor_2, ML_coor_2):
    #    print(i[16], j[16])
    #    print(k[16], l[16])
    #    print()
        #dist = np.sqrt(np.sum((i-j)**2, axis=1))
        #print(np.sort(dist)[-1:-10:-1])
        #print(np.argsort(dist)[-1:-10:-1])
        #print()