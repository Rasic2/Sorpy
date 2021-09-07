import math
import numpy as np
from collections import defaultdict

from utils import distance

class Atom:

    def __init__(self, frac_coord):
        self.frac_coord = frac_coord

class Molecule:

    def __init__(self, coords, latt):

        assert type(coords) == np.ndarray # only accept the np array
        assert len(coords) == 2 # only consider the CO molecule

        self.coords = coords
        self.latt = latt
        self.Atoms = [Atom(coord) for coord in self.frac_coords]

    def __getitem__(self, index):
        return self.Atoms[index]

    @property
    def frac_coords(self):
        return None

    @property
    def cart_coords(self):
        return np.dot(self.frac_coords, self.latt)

    @property
    def vector(self):
        return self.cart_coords[1] - self.cart_coords[0]

    @property
    def bond_length(self):
        return np.linalg.norm(self.vector)

    @property
    def theta(self):
        Axisz=np.array([0, 0, 1])
        return math.degrees(math.acos(np.dot(Axisz, self.vector)/ \
                                      (np.linalg.norm(Axisz, ord=1) * np.linalg.norm(self.vector, ord=1))))

    @property
    def phi(self):
        x = self.vector[0]
        y = self.vector[1]
        return math.degrees(math.atan2(y, x))


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


if __name__ == "__main__":
    for ii in range(50):
        print(f"POSCAR_ML_{ii+1}")
        p = POSCAR(f"../test/ML-2/POSCAR_ML_{ii+1}")
        p.nearest_neighbour_table()
        print(36, p.NNT[36])
        print(37, p.NNT[37])

