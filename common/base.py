import copy
from collections import Counter
from pathlib import Path
from typing import List, Any

import numpy as np
import yaml

from common.logger import root_dir, logger

yaml.warnings({'YAMLLoadWarning': False})


class Lattice(object):

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def __repr__(self):
        return f"{self.matrix}"

    def __eq__(self, other):
        return np.all(self.matrix == other.matrix)

    @property
    def length(self):
        return np.power(np.sum(np.power(self.matrix, 2), axis=1), 0.5)

    @property
    def inverse(self):
        return np.linalg.inv(self.matrix)

    @staticmethod
    def from_string(string):
        """
        @parameter
            string:     three-line <string>,
                        e.g.,    7.707464  0.000000  0.000000
                                -3.853732  6.674860  0.000000
                                 0.000000  0.000000 28.319031
        """
        matrix = np.array([[float(ii) for ii in item.split()] for item in string])
        return Lattice(matrix)

    @staticmethod
    def read_from_POSCAR(fname):
        with open(fname) as f:
            cfg = f.readlines()
        return Lattice.from_string(cfg[2:5])

    @property
    def strings(self):
        return "".join([" ".join([f"{ii:>9.6f}" for ii in item]) + "\n" for item in self.matrix])


# class Coordinates(object):
#     def __init__(self, frac_coords=None, cart_coords=None, lattice: Lattice = None):
#         self.frac_coords = frac_coords if frac_coords is not None else np.array([])
#         self.cart_coords = cart_coords if cart_coords is not None else np.array([])
#         self.lattice = lattice
#         self.index = 0
#         self.__set_coords()
#
#     def __getitem__(self, index):
#         try:
#             return Coordinates(self.frac_coords[index], self.cart_coords[index], self.lattice)
#         except IndexError:
#             return Coordinates(lattice=self.lattice)
#
#     def __iter__(self):
#         return self
#
#     def __len__(self):
#         assert len(self.frac_coords) == len(self.cart_coords)
#         return len(self.frac_coords)
#
#     def __next__(self):  # TODO "may produce bug"
#         if self.index < len(self):
#             parameter = (self.frac_coords[self.index], self.cart_coords[self.index], self.lattice)
#             self.index += 1
#             return Coordinates(*parameter)
#         else:
#             self.index = 0
#             raise StopIteration
#
#     def __repr__(self):
#         return f"<Coordinates {self.frac_coords.shape}>"
#
#     def __sub__(self, other):
#         assert len(self) == len(other)
#         assert self.frac_coords.shape == other.frac_coords.shape
#         return self.frac_coords - other.frac_coords
#
#     def __set_coords(self):
#         self.__set_cart_coords()
#         self.__set_frac_coords()
#
#     def __set_frac_coords(self):
#         if self.cart_coords.size > 0 and self.lattice is not None and self.frac_coords.size == 0:
#             self.frac_coords = np.dot(self.cart_coords, self.lattice.inverse)
#
#     def __set_cart_coords(self):
#         if self.frac_coords.size > 0 and self.lattice is not None and self.cart_coords.size == 0:
#             self.cart_coords = np.dot(self.frac_coords, self.lattice.matrix)
#
#     def to_strings(self, ctype="frac"):
#         if ctype == "frac":
#             return "".join([" ".join([f"{ii:>9.6f}" for ii in item]) + "\n" for item in self.frac_coords]).rstrip()
#         elif ctype == "cart":
#             return "".join([" ".join([f"{ii:>9.6f}" for ii in item]) + "\n" for item in self.cart_coords]).rstrip()
#         else:
#             raise NotImplementedError
#
#     @staticmethod
#     def read_from_strings(strings=None, ctype="frac", lattice: Lattice = None):
#         coords = np.array([[float(ii) for ii in item.split()[:3]] for item in strings])
#         if ctype == "frac":
#             return Coordinates(frac_coords=coords, lattice=lattice)
#         elif ctype == "cart":
#             return Coordinates(cart_coords=coords, lattice=lattice)
#         else:
#             raise ValueError("Invalid ctype parameter, should be <frac>, <cart>")


class Atom(object):
    """
        `Atom class represent one atom in periodic solid system`

        @property
            formula:        chemical formula
            number:         atomic number
            period:         atomic period in element period table
            group:          atomic group in element period table
            color:          atomic color using RGB
            order:          atomic order in <Structure class>, default: 0
            frac_coord:     fractional coordinates  
            cart_coord:     cartesian coordinates
            bonds:          atomic default bond property {atom: bond-length}
        
        @func
            __initialize_attrs:     initialize the attributes from the element.yaml
    """
    _config_file = Path(f"{root_dir}/config/element.yaml")
    _attributes_mono = ['number', 'period', 'group', 'color']
    _attributes_list = ['frac_coord', 'cart_coord']
    _attributes_dict = ['bonds']
    _initialize = False
    _attrs = None

    def __new__(cls, *args, **kwargs):
        if not cls._initialize:
            with open(cls._config_file) as f:
                _cfg = f.read()
            cls._attrs = yaml.safe_load(_cfg)
            cls._initialize = True
        return super(Atom, cls).__new__(cls)

    def __init__(self, formula, order: (int, list) = 0, frac_coord=None, cart_coord=None):
        self.formula = formula
        self.order = order
        self.number, self.period, self.group, self.color, self.bonds = (None, None, None, None, [])
        self.frac_coord = np.array(frac_coord) if frac_coord is not None else None
        self.cart_coord = np.array(cart_coord) if cart_coord is not None else None

        self.__initialize_attrs()

    def __eq__(self, other):
        return self.number == other.number and self.order == other.order

    def __lt__(self, other):
        return self.number < other.number or self.order < other.order

    def __ge__(self, other):
        return self.number >= other.number or self.order >= other.order

    def __hash__(self):
        return hash(self.number) + hash(str(self.order))

    def __repr__(self):
        return f"(Atom {self.order} : {self.formula} : {self.cart_coord})"

    def __initialize_attrs(self):
        if isinstance(self.formula, str):  # <class Atom>
            for key, value in self._attrs[f'Element {self.formula}'].items():
                setattr(self, key, value)
        elif isinstance(self.formula, list):  # <class Atoms>
            for attr in self._attributes_mono + self._attributes_dict:
                setattr(self, attr, [self._attrs[f'Element {formula}'][attr] for formula in self.formula])

    def set_coord(self, lattice: Lattice):
        assert lattice is not None
        if (self.cart_coord is not None and None not in self.cart_coord) and (
                self.frac_coord is None or None in self.frac_coord):
            self.frac_coord = np.dot(self.cart_coord, lattice.inverse)
        elif (self.frac_coord is not None and None not in self.frac_coord) and (
                self.cart_coord is None or None in self.cart_coord):
            self.cart_coord = np.dot(self.frac_coord, lattice.matrix)

        return self

    @staticmethod
    def search_image(atom_i, atom_j) -> np.ndarray:
        if not isinstance(atom_i, Atom) or not isinstance(atom_j, Atom):
            SystemError("The parameters should be <class Atom>!")
        logger.debug(
            f"Start search the {atom_i.formula}{atom_i.order}-{atom_j.formula}{atom_j.order} neighbour in all images!")
        image_pos = np.where(atom_j.frac_coord - atom_i.frac_coord <= 0.5, 0, -1)
        image_neg = np.where(atom_j.frac_coord - atom_i.frac_coord >= -0.5, 0, 1)
        image = image_pos + image_neg
        COD_frac = np.all(atom_j.frac_coord + image - atom_i.frac_coord <= 0.5) and np.all(
            atom_j.frac_coord + image - atom_i.frac_coord >= -0.5)
        if not COD_frac:
            SystemExit(f"Transform Error, exit!")
        logger.debug(f"Search the image {image} successfully!")

        return image


class Atoms(Atom):
    """
        `Atoms class represent atom set in periodic solid system`

        @property
            formula:        chemical formula, <list>
            number:         atomic number, <list>
            period:         atomic period in element period table, <list>
            group:          atomic group in element period table, <list>
            color:          atomic color using RGB, <list>
            order:          atomic order in <Structure class>, default: <list(range(len(formula)))>
            frac_coord:     fractional coordinates, <list>
            cart_coord:     cartesian coordinates, <list>
            bonds:          atomic default bond property {atom: bond-length}

            count:          total atom number in atom set
            size:           list atom number according to their formula

        @func
            __initialize_attrs:     initialize the attributes from the element.yaml
            from_list:              construct the <class Atoms> from an Atom list, i.e., [Atom, Atom, Atom] --> Atoms
    """

    def __new__(cls, *args, **kwargs):
        return super(Atoms, cls).__new__(cls)

    def __init__(self, formula, order: (int, list) = 0, frac_coord=None, cart_coord=None):
        super(Atoms, self).__init__(formula, order, frac_coord, cart_coord)
        self.order = list(range(len(self.formula))) if isinstance(self.order, int) else self.order
        self.frac_coord = [None] * len(self.formula) if self.frac_coord is None else self.frac_coord
        self.cart_coord = [None] * len(self.formula) if self.cart_coord is None else self.cart_coord
        self.__index = 0

    def __len__(self) -> int:
        return len(self.formula)

    def __repr__(self):
        string = ""
        for order, formula, cart_coord in zip(self.order, self.formula, self.cart_coord):
            string += f"(Atom {order} : {formula} : {cart_coord}) \n"
        return string

    def __iter__(self):
        return copy.deepcopy(self)  # return deepcopy(instance), otherwise the __index will create count bug

    def __next__(self):
        if self.__index < len(self):
            self.__index += 1
            return self[self.__index - 1]
        else:
            self.__index = 0
            raise StopIteration

    def __contains__(self, atom):
        if atom in list(self):
            return True
        else:
            return False

    def __getitem__(self, index):
        return Atom(formula=self.formula[index], order=self.order[index],
                    frac_coord=self.frac_coord[index], cart_coord=self.cart_coord[index])

    @property
    def count(self) -> int:
        return len(self)

    @property
    def size(self):
        return Counter(self.formula)

    @staticmethod
    def from_list(atoms: list):
        formula = [atom.formula for atom in atoms]
        order: List[Any] = [atom.order for atom in atoms]
        frac_coord = [atom.frac_coord for atom in atoms]
        cart_coord = [atom.cart_coord for atom in atoms]
        return Atoms(formula=formula, order=order, frac_coord=frac_coord, cart_coord=cart_coord)

# class AtomSetBase:
#
#     def __init__(self, elements=None, orders=None, coords: Coordinates = None, **kargs):
#
#         self.elements = elements if elements is not None else np.array([])
#         self.orders = orders
#         self.coords = coords
#
#         for key, value in kargs.items():
#             if getattr(self, key, None) is None:
#                 setattr(self, key, value)
#
#     def __len__(self):
#         assert len(self.elements) == len(self.coords)
#         return len(self.elements)
#
#     def __contains__(self, item):
#         if item is self.atoms:
#             return True
#         else:
#             return False
#
#     @property
#     def frac_coords(self):
#         return self.coords.frac_coords
#
#     @property
#     def cart_coords(self):
#         return self.coords.cart_coords
#
#     @property
#     def atoms(self):
#         return np.array([Atom(element, order, coord) for element, order, coord in
#                          itertools.zip_longest(self.elements, self.orders, self.coords)])
#
#     @property
#     def atoms_total(self) -> int:
#         return len(self)
#
#     @property
#     def atoms_formulas(self):
#         return [element.formula for element in self.elements]
#
#     @property
#     def atoms_count(self):
#         return sorted(Counter(self.atoms_formulas).items(), key=lambda x: Element(x[0]).number)
#
#     @property
#     def bonds(self):
#         min_factor, max_factor = 0.8, 1.2
#         bonds = Format_defaultdict(list)
#         for atom_i in self.atoms:
#             for atom_j in self.atoms:
#                 if atom_i != atom_j and atom_j.element in atom_i.element.bonds.keys():
#                     bond_length = np.linalg.norm(
#                         self.coords[atom_j.order].cart_coords - self.coords[atom_i.order].cart_coords)
#                     if min_factor <= bond_length / atom_i.element.bonds[atom_j.element] <= max_factor:
#                         bonds[atom_i].append((atom_j, bond_length))
#
#         sorted_bonds = Format_defaultdict(list)
#         for key, value in bonds.items():
#             sorted_bonds[key] = sorted(value, key=lambda x: x[1])
#
#         return bonds
