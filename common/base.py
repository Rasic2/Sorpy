import copy
import pickle
import itertools
from pathlib import Path
from collections import Counter

import numpy as np
import yaml

from common.logger import root_dir
from common.utils import Format_defaultdict

yaml.warnings({'YAMLLoadWarning': False})


class Lattice:

    def __init__(self, matrix):
        self.matrix = matrix

    def __repr__(self):
        return f"{self.matrix}"

    def __eq__(self, other):
        return np.all(self.matrix == other.matrix)

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

    @property
    def to_strings(self):
        return "".join([" ".join([f"{ii:>9.6f}" for ii in item]) + "\n" for item in self.matrix])


class Coordinates:
    def __init__(self, frac_coords=None, cart_coords=None, lattice: Lattice = None):
        self.frac_coords = frac_coords if frac_coords is not None else np.array([])
        self.cart_coords = cart_coords if cart_coords is not None else np.array([])
        self.lattice = lattice
        self.index = 0
        self.__set_coords()

    def __getitem__(self, index):
        try:
            return Coordinates(self.frac_coords[index], self.cart_coords[index], self.lattice)
        except IndexError:
            return Coordinates(lattice=self.lattice)

    def __iter__(self):
        return self

    def __len__(self):
        assert len(self.frac_coords) == len(self.frac_coords)
        return len(self.frac_coords)

    def __next__(self):  # TODO "may produce bug"
        if self.index < len(self):
            parameter = (self.frac_coords[self.index], self.cart_coords[self.index], self.lattice)
            self.index += 1
            return Coordinates(*parameter)
        else:
            self.index = 0
            raise StopIteration

    def __repr__(self):
        return f"<Coordinates {self.frac_coords.shape}>"

    def __sub__(self, other):
        assert len(self) == len(other)
        assert self.frac_coords.shape == other.frac_coords.shape
        return self.frac_coords - other.frac_coords

    def __set_coords(self):
        self.__set_cart_coords()
        self.__set_frac_coords()

    def __set_frac_coords(self):
        if self.cart_coords.size > 0 and self.lattice is not None and self.frac_coords.size == 0:
            self.frac_coords = np.dot(self.cart_coords, self.lattice.inverse)

    def __set_cart_coords(self):
        if self.frac_coords.size > 0 and self.lattice is not None and self.cart_coords.size == 0:
            self.cart_coords = np.dot(self.frac_coords, self.lattice.matrix)

    def to_strings(self, ctype="frac"):
        if ctype == "frac":
            return "".join([" ".join([f"{ii:>9.6f}" for ii in item]) + "\n" for item in self.frac_coords]).rstrip()
        elif ctype == "cart":
            return "".join([" ".join([f"{ii:>9.6f}" for ii in item]) + "\n" for item in self.cart_coords]).rstrip()
        else:
            raise NotImplementedError

    @staticmethod
    def read_from_strings(strings=None, ctype="frac", lattice: Lattice=None):
        coords = np.array([[float(ii) for ii in item.split()[:3]] for item in strings])
        if ctype == "frac":
            return Coordinates(frac_coords=coords, lattice=lattice)
        elif ctype == "cart":
            return Coordinates(cart_coords=coords, lattice=lattice)
        else:
            raise ValueError("Invalid ctype parameter, should be <frac>, <cart>")


class Element:
    with open(Path(f"{root_dir}/config/element.yaml")) as f:
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
        return f"<Element {self.formula}>"

    @property
    def number(self) -> int:
        return Element.elements[f'Element {self.formula}']['number']

    @property
    def bonds(self):
        return {Element(bond['formula']): bond['bond length'] for bond in
                Element.elements[f'Element {self.formula}']['bonds']}

class Elements:

    def __new__(cls, *args, **kwargs):
        raise TypeError("Can't create the <class Elements> instance.")

    @staticmethod
    def read_from_strings(formulas, counts):
        elements = [(formula, int(count)) for formula, count in zip(formulas.split(), counts.split())]
        elements = sum([[formula] * count for (formula, count) in elements], [])
        return [Element(formula) for formula in elements]


class Atom:
    """Periodic System in Solid"""

    def __init__(self, element: Element = None, order: int = None, coord: Coordinates = None):
        self.element = element
        self.order = order
        self.coord = coord

    def __eq__(self, other):
        return self.element == other.element and self.order == other.order

    def __lt__(self, other):
        return self.element < other.element or self.order < other.order

    def __ge__(self, other):
        return self.element >= other.element or self.order >= other.order

    def __hash__(self):
        return hash(self.element) + hash(str(self.order))

    def __repr__(self):
        return f"(Atom {self.order} : {self.element} : {self.coord})"

    @property
    def frac_coord(self):
        return self.coord.frac_coords

    @property
    def cart_coord(self):
        return self.coord.cart_coords


class AtomSetBase:

    def __init__(self, elements=None, orders=None, coords: Coordinates = None, **kargs):

        self.elements = elements if elements is not None else np.array([])
        self.orders = orders
        self.coords = coords

        self._pseudo_bonds = None

        for key, value in kargs.items():
            if getattr(self, key, None) is None:
                setattr(self, key, value)

    def __len__(self):
        assert len(self.elements) == len(self.coords)
        return len(self.elements)

    def __contains__(self, item):
        if item is self.atoms:
            return True
        else:
            return False

    @property
    def frac_coords(self):
        return self.coords.frac_coords

    @frac_coords.setter
    def frac_coords(self, frac_coords):
        self.coords.frac_coords = frac_coords

    @property
    def cart_coords(self):
        return self.coords.cart_coords

    @property
    def atoms(self):
        return np.array([Atom(element, order, coord) for element, order, coord in
                         itertools.zip_longest(self.elements, self.orders, self.coords)])

    @property
    def atoms_total(self) -> int:
        return len(self)

    @property
    def atoms_formulas(self):
        return [element.formula for element in self.elements]

    @property
    def atoms_count(self):
        return sorted(Counter(self.atoms_formulas).items(), key=lambda x: Element(x[0]).number)

    @property
    def pseudo_bonds(self):
        if self._pseudo_bonds is None:
            NNT = Format_defaultdict(list)
            for atom_i in self.atoms:
                for atom_j in self.atoms:
                    if atom_j != atom_i:
                        atom_j_frac = np.copy(atom_j.frac_coord)  # Handle the PBC
                        atom_j_frac = np.where((atom_j_frac - atom_i.frac_coord) > 0.5, atom_j_frac - 1, atom_j_frac)
                        atom_j_frac = np.where((atom_j_frac - atom_i.frac_coord) < -0.5, atom_j_frac + 1, atom_j_frac)
                        atom_j_cart = np.dot(atom_j_frac, self.coords.lattice.matrix)
                        distance = np.linalg.norm(atom_j_cart - atom_i.cart_coord)
                        NNT[atom_i].append((atom_j, distance))

            sorted_NNT = Format_defaultdict(list)
            for key, value in NNT.items():
                sorted_NNT[key] = sorted(value, key=lambda x: x[1])

            setattr(self, "_pseudo_bonds", sorted_NNT)

        return self._pseudo_bonds

    @property
    def bonds(self):
        min_factor, max_factor = 0.8, 1.2
        bonds = Format_defaultdict(list)
        for atom_i, value in self.pseudo_bonds.items():
            for atom_j, dist in value:
                if atom_j.element in atom_i.element.bonds.keys() and min_factor <= dist / atom_i.element.bonds[atom_j.element] <= max_factor:
                    bonds[atom_i].append((atom_j, dist))
        return bonds
