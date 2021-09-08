import math
import itertools
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
        frac_vector = self.coords[1] -self.coords[0]
        self.coords[1] = np.where(frac_vector > 0.5, self.coords[1] - 1, self.coords[1])
        self.coords[1] = np.where(frac_vector < -0.5, self.coords[1] + 1, self.coords[1])
        return self.coords

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
        z = self.vector[2]
        r = self.bond_length
        return math.degrees(math.atan2(math.sqrt(r**2-z**2), z))

    @property
    def phi(self):
        x = self.vector[0]
        y = self.vector[1]
        return math.degrees(math.atan2(y, x))


class Latt:

    def __init__(self, matrix):
        """
        TODO inverse of matrix
        :param string:
        """
        assert type(matrix) == np.ndarray

        self.matrix = matrix

    @property
    def inverse(self):
        return np.linalg.inv(self.matrix)

    @staticmethod
    def read_from_string(string):
        """
        :param string:      POSCAR文件中的3行Lattice矢量
        :return:
        """

        assert isinstance(string, list)
        assert len(string) == 3

        matrix = np.array([[float(ii) for ii in item.split()] for item in string])
        return Latt(matrix)

class POSCAR:

    def __init__(self, fname, action="r"):
        self.fname = fname
        self.action = action
        self.__load() if self.action == "r" else None

    def __load(self):
        with open(self.fname, self.action) as f:
            self.cfg = f.readlines()
        self.system = self.cfg[0].rstrip()
        self.factor = self.cfg[1].rstrip()
        self.latt = Latt.read_from_string(self.cfg[2:5])
        self.elements = [(name, int(count)) for name, count in zip(self.cfg[5].split(), self.cfg[6].split())]
        self.selective = self.cfg[7].lower()[0] == "s"
        self.coor_type = self.cfg[8].rstrip()
        assert self.coor_type.lower()[0] == "d"
        self.frac_coors = self._frac_coors()
        self.cart_coors = self._cart_coors()
        self.TF = self._TF()

    @property
    def ele_count(self):
        return sum(n for _, n in self.elements)

    def _frac_coors(self):
        if self.selective:
            return np.array([[float(ii) for ii in item.split()[:3]] for item in self.cfg[9:9+self.ele_count]])
        else:
            return None

    def _cart_coors(self):
        if self.frac_coors is not None:
            return np.dot(self.frac_coors, self.latt.matrix)
        else:
            return None

    def _TF(self):
        if self.selective:
            return np.array([item.split()[3:6] for item in self.cfg[9:9+self.ele_count]])
        else:
            return None

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

    @staticmethod
    def mcoord_to_coord(array, latt, anchor, intercoor_index):

        assert type(array) == np.ndarray
        assert type(latt) == Latt

        total_index = list(range(len(array)))
        frac_index = list(set(total_index).difference(set(intercoor_index)))
        anchor_cart_coor = np.dot(array[anchor], latt.matrix)
        intercoor = array[intercoor_index] * [1, 180, 360] + [1.142, 0, 0] # TODO setting.yaml modify
        intercoor = intercoor.reshape(3)
        r, theta, phi = intercoor[0], math.radians(intercoor[1]), math.radians(intercoor[2])
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(theta) * math.sin(phi)
        z = r * math.cos(theta)
        mol_cart_coor = anchor_cart_coor + [x, y, z]
        mol_frac_coor = np.dot(mol_cart_coor, latt.inverse).reshape((1,3))
        return np.concatenate((array[frac_index], mol_frac_coor), axis=0)

    def write(self, template, coor):

        assert type(template) == POSCAR
        assert self.action == "w"

        with open(self.fname, self.action) as f:
            f.writelines(template.cfg[:9])
            for ii, jj in zip(coor, template.TF):
                item = f"{ii[0]:.6f} {ii[1]:.6f} {ii[2]:.6f} {jj[0]} {jj[1]} {jj[2]} \n"
                f.write(item)

    def align(self, template):
        indicator = 0 # 第二次对齐的原子
        repeat = 2 # supercell
        slab_coor = self.frac_coors[:36]
        mol_coor = self.frac_coors[36:38]
        model_region = np.array([1/repeat, 1/repeat])
        mass_center = np.mean(mol_coor, axis=0)[:2]
        search_vector = np.arange(-1, 1+1/repeat, 1/repeat)
        search_matrix = itertools.product(search_vector, search_vector)
        for ii in search_matrix:
            if 0 <= (mass_center + ii)[0] <= model_region[0] and 0 <= (mass_center + ii)[1] <= model_region[1]:
                vector_m = [ii[0], ii[1], 0]
                break
        coor_m = mol_coor + vector_m
        coor_first = np.concatenate((slab_coor, coor_m), axis=0)

        for ii, jj in zip(self.elements, template.elements):
            if ii[0] != jj[0] or ii[1] != jj[1]:
                raise NotImplementedError # TODO 3x3 or len(atoms) 不相等

        self.frac_coors = coor_first
        self.cart_coors = self._cart_coors() # Reset the cart_coords

        NNT = defaultdict(list)
        for ii in range(self.ele_count):
            for jj in range(template.ele_count):
                dis = distance(self.cart_coors[ii], self.cart_coors[jj])
                NNT[ii].append((jj, dis))

        sorted_NNT = defaultdict(list)
        for key, value in NNT.items():
            sorted_NNT[key] = sorted(value, key=lambda x: x[1])

        min_NNT = []
        for key, value in sorted_NNT.items():
            min_NNT.append((key, value[0][0], value[0][1]))

        for ii, jj, kk in sorted(min_NNT, key=lambda x: x[2]):
            print(ii, jj)
        #    if mass_center + model_region * i
        #print(model_region)

if __name__ == "__main__":

    for ii in range(50):
        print(f"POSCAR_ML_{ii+1}")
        p = POSCAR(f"../test/ML-2/POSCAR_ML_{ii+1}")
        p.nearest_neighbour_table()
        print(36, p.NNT[36])
        print(37, p.NNT[37])

