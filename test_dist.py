import numpy as np

from main import DirManager, CoorTailor
from utils import distance

if __name__ == "__main__":
    #ori_DM = DirManager("test/ori", "POSCAR", mol_index="37-38")
    #ML_DM = DirManager("test/ML-2", "POSCAR", mol_index="37-38")

    ori_DM = DirManager("input-test", "POSCAR", mol_index="37-38")
    ML_DM = DirManager("output-test", "CONTCAR", mol_index="37-38")

    ori_coor = ori_DM.coords
    ML_coor = ML_DM.coords

    CT = CoorTailor(ori_coor, ML_coor, 2, intercoor_index=[])
    #CT._expand_xy()
    CT.run(boundary=0, num=0)
    ori_coor_2, ML_coor_2 = CT.input_arr, CT.output_arr

    for i, j, k, l in zip(ori_coor, ML_coor, ori_coor_2, ML_coor_2):
        print(i[16], j[16])
        print(k[16], l[16])
        print()
        #dist = np.sqrt(np.sum((i-j)**2, axis=1))
        #print(np.sort(dist)[-1:-10:-1])
        #print(np.argsort(dist)[-1:-10:-1])
        #print()