import numpy as np
from collections import defaultdict

from utils import distance

class Latt:

    def __init__(self, string):
        self.string = string
        assert isinstance(self.string, list)
        self.matrix = np.array([[float(ii) for ii in item.split()] for item in self.string])

class POSCAR:

    def __init__(self, fname):
        self.fname = fname
        self.__load()

    def __load(self):
        with open(self.fname, "r") as f:
            self.cfg = f.readlines()
        self.system = self.cfg[0].rstrip()
        self.factor = self.cfg[1].rstrip()
        self.latt = Latt(self.cfg[2:5]).matrix
        self.element = [(name, int(count)) for name, count in zip(self.cfg[5].split(), self.cfg[6].split())]
        self.selective = self.cfg[7].lower()[0] == "s"
        self.coor_type = self.cfg[8].rstrip()
        assert self.coor_type.lower()[0] == "d"
        self.frac_coors = self._frac_coors()
        self.cart_coors = self._cart_coors()

    @property
    def ele_count(self):
        return sum(n for _, n in self.element)

    def _frac_coors(self):
        if self.selective:
            return np.array([[float(ii) for ii in item.split()[:3]] for item in self.cfg[9:9+self.ele_count]])

    def _cart_coors(self):
        return np.dot(self._frac_coors(), self.latt)

    def nearest_neighbour_table(self):

        NNT = defaultdict(list)
        cut_radius = 3.0
        for ii in range(self.ele_count):
            for jj in range(self.ele_count):
                if jj != ii:
                    dis = distance(self.cart_coors[ii], self.cart_coors[jj])
                    if dis <= cut_radius:
                        NNT[ii].append((jj, dis))

        # sorted NNT
        sorted_NNT = defaultdict(list)
        for key, value in NNT.items():
            sorted_NNT[key] = sorted(value, key = lambda x : x[1])

        setattr(self, "NNT", sorted_NNT)
        return sorted_NNT

for ii in range(50):
    print(f"POSCAR_ML_{ii+1}")
    p = POSCAR(f"../test/ML-2/POSCAR_ML_{ii+1}")
    p.nearest_neighbour_table()
    print(36, p.NNT[36])
    print(37, p.NNT[37])

