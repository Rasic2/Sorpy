import yaml
import itertools
from collections import Counter
import numpy as np

from logger import current_dir

yaml.warnings({'YAMLLoadWarning': False})


class Lattice:

    def __init__(self, matrix):
        self.matrix = matrix

    def __repr__(self):
        return f"{self.matrix}"

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


class Element:
    with open(f"{current_dir}/common/element.yaml") as f:
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

    def __init__(self, elements=None, coords: Coordinates = None, **kargs):

        self.elements = elements if elements is not None else np.array([])
        self.coords = coords

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

    @property
    def cart_coords(self):
        return self.coords.cart_coords

    @property
    def atoms(self):
        return np.array([Atom(element, order, coord) for element, order, coord in
                         itertools.zip_longest(self.elements, range(len(self)), self.coords)])

    @property
    def atoms_total(self) -> int:
        return len(self)

    @property
    def atoms_formulas(self):
        return [element.formula for element in self.elements]

    @property
    def atoms_count(self):
        return sorted(Counter(self.atoms_formulas).items(), key=lambda x: Element(x[0]).number)
