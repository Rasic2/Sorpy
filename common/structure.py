import math
import yaml
import itertools
import numpy as np
from collections import defaultdict, Counter

from utils import distance

yaml.warnings({'YAMLLoadWarning': False})


class Element:

    with open("element.yaml") as f:
        cfg = f.read()
    elements = yaml.load(cfg)

    def __init__(self, formula):
        self.formula = formula

    def __eq__(self, other):
        return self.number == other.number

    def __lt__(self, other):
        return self.number < other.number

    def __ge__(self, other):
        return self.number >= other.number

    def __hash__(self):
        return hash(self.number)

    def __repr__(self):
        return self.formula

    @property
    def number(self) -> int:
        return Element.elements[f'Element {self.formula}']['number']


class Atom:

    def __init__(self, element: Element= None, order: int = None, frac_coord=None, cart_coord=None):
        self.element = element
        self.order = order
        self.frac_coord = frac_coord
        self.cart_coord = cart_coord

    def __eq__(self, other):
        return self.element == other.element and self.order == other.order

    def __lt__(self, other):
        return self.element < other.element or self.order < other.order

    def __ge__(self, other):
        return self.element >= other.element or self.order >= other.order

    def __hash__(self):
        return hash(self.element) + hash(str(self.order))

    def __repr__(self):
        return f"<Atom {self.order}: {self.element}>"


class Lattice:

    def __init__(self, matrix):
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
        matrix = np.array([[float(ii) for ii in item.split()] for item in string])
        return Lattice(matrix)

    @staticmethod
    def read_from_POSCAR(fname):
        with open(fname) as f:
            cfg = f.readlines()
        return Lattice.read_from_string(cfg[2:5])


class AtomSetBase:

    def __init__(self, elements=None, frac_coords=None, cart_coords=None, **kargs):

        self.elements = elements if elements is not None else []
        self.frac_coords = frac_coords if frac_coords is not None else []
        self.cart_coords = cart_coords if cart_coords is not None else []

        for key, value in kargs.items():
            if getattr(self, key, None) is None:
                setattr(self, key, value)

    @property
    def atoms(self):
        return [Atom(element, order, frac_coord, cart_coord) \
                for element, order, frac_coord, cart_coord in \
                zip(self.elements, range(len(self.frac_coords)), self.frac_coords, self.cart_coords)]

    @property
    def atoms_total(self) -> int:
        return len(self.atoms)

    @property
    def atoms_formulas(self):
        return [element.formula for element in self.elements]

    @property
    def atoms_count(self) -> Counter:
        return Counter(list(self.atoms_formulas))

class Molecule(AtomSetBase):

    def __init__(self, elements=None, frac_coords=None, cart_coords=None, anchor=None, **kargs):

        super().__init__(elements=elements, frac_coords=frac_coords, cart_coords=cart_coords, **kargs)
        self.anchor = anchor if isinstance(anchor, (int, Atom)) else None

    def __getitem__(self, index):
        return self.atoms[index]

    @property
    def pair(self):
        pair_list = []
        for ii in itertools.product(self.atoms, self.atoms):
            if ii[0] != ii[1]:
                pair_list.append(ii)
        pair_list = (tuple(sorted(item)) for item in pair_list)

        return set(pair_list)

    @property
    def vector(self):
        """ vector in Cartesian format """
        return [(atom_i, atom_j, atom_j.cart_coord - atom_i.cart_coord) for atom_i, atom_j in self.pair]

    @property
    def dist(self):
        if self.anchor: # anchor Atom and Reset vector
            atom_j = self.atoms[self.anchor] if isinstance(self.anchor, int) else self.anchor
            vectors = [atom_i.cart_coord - atom_j.cart_coord for atom_i in self.atoms if atom_i != atom_j]
            return [(atom_j, atom_i, np.linalg.norm(vector)) \
                    for atom_i, vector in zip(self.atoms, vectors) if atom_i != atom_j]
        else:
            return [(atom_i, atom_j, np.linalg.norm(vector)) for atom_i, atom_j, vector in self.vector]

    @property
    def theta(self):
        if self.anchor: # anchor Atom and Reset vector
            atom_j = self.atoms[self.anchor] if isinstance(self.anchor, int) else self.anchor
            vectors = [atom_i.cart_coord - atom_j.cart_coord for atom_i in self.atoms if atom_i != atom_j]
            return [(atom_j, atom_i, math.degrees(math.atan2(math.sqrt(dist ** 2 - vector[2] ** 2), vector[2]))) \
                    for (vector), (_, atom_i, dist) in zip(vectors, self.dist) if atom_i != atom_j]
        else:
            return [(atom_i, atom_j, math.degrees(math.atan2(math.sqrt(dist ** 2 - vector[2] ** 2), vector[2]))) \
                    for (atom_i, atom_j, vector), (_, _, dist) in zip(self.vector, self.dist)]

    @property
    def phi(self):
        if self.anchor: # anchor Atom and Reset vector
            atom_j = self.atoms[self.anchor] if isinstance(self.anchor, int) else self.anchor
            vectors = [atom_i.cart_coord - atom_j.cart_coord for atom_i in self.atoms if atom_i != atom_j]
            return [(atom_j, atom_i, math.degrees(math.atan2(vector[1], vector[0]))) \
                    for (_, atom_i, _), (vector) in zip(self.theta, vectors)]
        else:
            return [(atom_i, atom_j, math.degrees(math.atan2(vector[1], vector[0]))) \
                    for (atom_i, atom_j, vector) in self.vector]

    @property
    def inter_coords(self):
        return [(atom_i, atom_j, [dist, theta, phi]) \
                for (atom_i, atom_j, dist), (_, _, theta), (_, _, phi) in zip(self.dist, self.theta, self.phi)]


class Slab(AtomSetBase):

    def __init__(self, elements=None, frac_coords=None, cart_coords=None, lattice: Lattice=None, **kargs):

        super().__init__(elements=elements, frac_coords=frac_coords, cart_coords=cart_coords, **kargs)
        self.lattice = lattice
        self.__set_coords() # Rewrite the frac_coords, cart_coords and atoms

        assert len(self.elements) == len(self.frac_coords) == len(self.cart_coords), \
            "The shape of <formulas>, <frac_coords>, <cart_coords> are not equal."

    def __set_coords(self):
        self.__set_cart_coords()
        self.__set_frac_coords()

    def __set_frac_coords(self):
        if self.cart_coords and self.lattice is not None and self.frac_coords is None:
            self.frac_coords = np.dot(self.cart_coords, self.lattice.inverse)

    def __set_cart_coords(self):
        if self.frac_coords and self.lattice is not None and self.cart_coords is None:
            self.cart_coords = np.dot(self.frac_coords, self.lattice.matrix)


m = Molecule(elements=[Element("C"), Element("O")], frac_coords=np.array([[0, 0, 0], [0, 0, 0.5]]),
             cart_coords=np.array([[0, 0, 0], [0, 0, 1.142]]))
#print(m.pair)
#print(m.vector)
#print(m.dist)
#print(m.theta)
#print(m.phi)
#print(m.frac_coords)
#print(m.cart_coords)
#print(m.inter_coords)
#print(m.atoms_count)
print(m.TF)
exit()




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
            return np.array([[float(ii) for ii in item.split()[:3]] for item in self.cfg[9:9 + self.ele_count]])
        else:
            return None

    def _cart_coors(self):
        if self.frac_coors is not None:
            return np.dot(self.frac_coors, self.latt.matrix)
        else:
            return None

    def _TF(self):
        if self.selective:
            return np.array([item.split()[3:6] for item in self.cfg[9:9 + self.ele_count]])
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
            sorted_NNT[key] = sorted(value, key=lambda x: x[1])

        setattr(self, "NNT", sorted_NNT)
        return sorted_NNT

    @staticmethod
    def mcoord_to_coord(array, latt, anchor, intercoor_index):

        assert type(array) == np.ndarray
        assert type(latt) == Latt

        total_index = list(range(len(array)))
        frac_index = list(set(total_index).difference(set(intercoor_index)))
        anchor_cart_coor = np.dot(array[anchor], latt.matrix)
        intercoor = array[intercoor_index] * [1, 180, 360] + [1.142, 0, 0]  # TODO setting.yaml modify
        intercoor = intercoor.reshape(3)
        r, theta, phi = intercoor[0], math.radians(intercoor[1]), math.radians(intercoor[2])
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(theta) * math.sin(phi)
        z = r * math.cos(theta)
        mol_cart_coor = anchor_cart_coor + [x, y, z]
        mol_frac_coor = np.dot(mol_cart_coor, latt.inverse).reshape((1, 3))
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
        indicator = 0  # 第二次对齐的原子
        repeat = 2  # supercell
        slab_coor = self.frac_coors[:36]
        mol_coor = self.frac_coors[36:38]
        model_region = np.array([1 / repeat, 1 / repeat])
        mass_center = np.mean(mol_coor, axis=0)[:2]
        search_vector = np.arange(-1, 1 + 1 / repeat, 1 / repeat)
        search_matrix = itertools.product(search_vector, search_vector)
        for ii in search_matrix:
            if 0 <= (mass_center + ii)[0] <= model_region[0] and 0 <= (mass_center + ii)[1] <= model_region[1]:
                vector_m = [ii[0], ii[1], 0]
                break
        coor_m = mol_coor + vector_m
        coor_first = np.concatenate((slab_coor, coor_m), axis=0)

        return vector_m, coor_first
        exit()

        for ii, jj in zip(self.elements, template.elements):
            if ii[0] != jj[0] or ii[1] != jj[1]:
                raise NotImplementedError  # TODO 3x3 or len(atoms) 不相等

        self.frac_coors = coor_first
        self.cart_coors = self._cart_coors()  # Reset the cart_coords

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
        # print(model_region)


if __name__ == "__main__":

    for ii in range(50):
        print(f"POSCAR_ML_{ii + 1}")
        p = POSCAR(f"../test/ML-2/POSCAR_ML_{ii + 1}")
        p.nearest_neighbour_table()
        print(36, p.NNT[36])
        print(37, p.NNT[37])
