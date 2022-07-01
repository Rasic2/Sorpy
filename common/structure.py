import copy
import itertools
from collections import defaultdict

import numpy as np

from common.base import Atom, Lattice, Atoms
from common.logger import logger


# class Molecule(AtomSetBase):
#
#     def __init__(self, elements=None, orders=None, coords: Coordinates = None, anchor=None, **kargs):
#
#         super().__init__(elements=elements, orders=orders, coords=coords, **kargs)
#         self.anchor = anchor if isinstance(anchor, (int, Atom)) else None
#         for index, atom in enumerate(self.atoms):
#             if self.anchor == atom.order:
#                 self.anchor = index
#
#     def __repr__(self):
#         return f"------------------------------------------------------------\n" \
#                f"<Molecule>                                                  \n" \
#                f"-Atoms-                                                     \n" \
#                f"{self.atoms}                                                \n" \
#                f"------------------------------------------------------------"
#
#     def __getitem__(self, index):
#         return self.atoms[index]
#
#     @property
#     def pair(self):
#         pair_list = []
#         for ii in itertools.product(self.atoms, self.atoms):
#             if ii[0] != ii[1]:
#                 pair_list.append(ii)
#         pair_list = (tuple(sorted(item)) for item in pair_list)
#
#         return set(pair_list)
#
#     @property
#     def vector(self):  # TODO PBC apply not considered important error
#         """ vector in Cartesian format """
#         lattice = self.coords.lattice
#         pair = set()
#         for atom_i, atom_j in self.pair:  # handle the PBC principle, Reset the molecule.atoms !!!
#             element = copy.deepcopy(atom_j.element)
#             order = copy.deepcopy(atom_j.order)
#             frac_coord = copy.deepcopy(atom_j.frac_coord)
#             frac_coord = np.where(frac_coord - atom_i.frac_coord > 0.5, frac_coord - 1, frac_coord)
#             frac_coord = np.where(frac_coord - atom_i.frac_coord < -0.5, frac_coord + 1, frac_coord)
#             coord = Coordinates(frac_coords=frac_coord, lattice=lattice)
#             atom_j = Atom(element=element, order=order, coord=coord)
#             pair.add((atom_i, atom_j))
#         return [(atom_i, atom_j, atom_j.cart_coord - atom_i.cart_coord) for atom_i, atom_j in pair]
#
#     @property
#     def dist(self):
#         if self.anchor:  # anchor Atom and Reset vector
#             atom_j = self.atoms[self.anchor] if isinstance(self.anchor, int) else self.anchor
#             vectors = [atom_i.cart_coord - atom_j.cart_coord for atom_i in self.atoms if atom_i != atom_j]
#             return [(atom_j, atom_i, np.linalg.norm(vector))
#                     for atom_i, vector in zip(self.atoms, vectors) if atom_i != atom_j]
#         else:
#             return [(atom_i, atom_j, np.linalg.norm(vector)) for atom_i, atom_j, vector in self.vector]
#
#     @property
#     def theta(self):
#         if self.anchor:  # anchor Atom and Reset vector
#             atom_j = self.atoms[self.anchor] if isinstance(self.anchor, int) else self.anchor
#             vectors = [atom_i.cart_coord - atom_j.cart_coord for atom_i in self.atoms if atom_i != atom_j]
#             return [(atom_j, atom_i, math.degrees(math.atan2(math.sqrt(dist ** 2 - vector[2] ** 2), vector[2])))
#                     for (vector), (_, atom_i, dist) in zip(vectors, self.dist) if atom_i != atom_j]
#         else:
#             return [(atom_i, atom_j, math.degrees(math.atan2(math.sqrt(dist ** 2 - vector[2] ** 2), vector[2])))
#                     for (atom_i, atom_j, vector), (_, _, dist) in zip(self.vector, self.dist)]
#
#     @property
#     def phi(self):
#         if self.anchor:  # anchor Atom and Reset vector
#             atom_j = self.atoms[self.anchor] if isinstance(self.anchor, int) else self.anchor
#             vectors = [atom_i.cart_coord - atom_j.cart_coord for atom_i in self.atoms if atom_i != atom_j]
#             return [(atom_j, atom_i, math.degrees(math.atan2(vector[1], vector[0])))
#                     for (_, atom_i, _), (vector) in zip(self.theta, vectors)]
#         else:
#             return [(atom_i, atom_j, math.degrees(math.atan2(vector[1], vector[0])))
#                     for (atom_i, atom_j, vector) in self.vector]
#
#     @property
#     def inter_coords(self):
#         return [(atom_i, atom_j, [dist, theta, phi])
#                 for (atom_i, atom_j, dist), (_, _, theta), (_, _, phi) in zip(self.dist, self.theta, self.phi)]
#
#
# class Slab(AtomSetBase):
#
#     def __init__(self, elements=None, orders=None, coords: Coordinates = None, lattice: Lattice = None, **kargs):
#         super().__init__(elements=elements, orders=orders, coords=coords, **kargs)
#         self.lattice = lattice
#
#         assert len(self.elements) == len(self.frac_coords) == len(self.cart_coords), \
#             "The shape of <formulas>, <frac_coords>, <cart_coords> are not equal."
#
#     def __repr__(self):
#         return f"------------------------------------------------------------\n" \
#                f"<Slab>                                                      \n" \
#                f"-Lattice-                                                   \n" \
#                f"{self.lattice.matrix}                                       \n" \
#                f"-Atoms-                                                     \n" \
#                f"{self.atoms}                                                \n" \
#                f"------------------------------------------------------------" \
#             if self.lattice is not None else f"<Slab object>"
#
#     @property
#     def mass_center(self):
#         return np.sum(self.coords.frac_coords, axis=0) / len(self)


class Structure():
    """TODO <class Coordinates including the frac, cart transfer>"""
    _styles = ("Crystal", "Slab", "Mol", "Slab+Mol")
    _extra_attrs = ("TF",)

    def __init__(self, style=None, atoms: Atoms = None, lattice: Lattice = None, mol_index=None, **kargs):
        """
        @parameter:
            style:              <Required> Indicate the system style: <"Crystal", "Slab", "Mol", "Slab+Mol">
            atoms:              <Required> atoms of the structure, <class Atoms>
            lattice:            <Required> Lattice vector
            mol_index:          <Optional> molecule index, if style=Mol or Slab+Mol
            kargs:              <Optional> <TF, anchor, ignore_mol, ignore_index>

        @property:
            neighbour_tabel:    neighbour table of structure, number of neighbour_atom default is 12

        @func:
            find_neighbour_tables(self, neighbour_num: int = 12, adj_matrix=None) --> self.neighbour_table
            to_POSCAR(self, fname, system=None, factor=1): output the structure into `POSCAR/CONTCAR` file

            from_POSCAR(fname, style=None, mol_index=None, **kargs) --> Structure
            from_adj_matrix(structure, adj_matrix, adj_matrix_tuple, bond_dist3d, known_first_order) --> Structure
        """
        self.style = style
        self.atoms = atoms
        self.lattice = lattice
        self.neighbour_table = None

        if self.style not in Structure._styles:
            raise AttributeError(f"The '{self.style}' not support in this version, optional style: {Structure._styles}")

        # orders = list(range(len(elements)))
        # super().__init__(elements=elements, orders=orders, coords=coords, **kargs)

        mol_index = mol_index if mol_index is not None else []
        # self.index = list(range(len(self.atoms)))
        self.mol_index = mol_index if isinstance(mol_index, (list, np.ndarray)) else [mol_index]
        self.slab_index = list(
            set(self.atoms.order).difference(set(self.mol_index))) if mol_index is not None else self.atoms.order

        for key, value in kargs.items():
            if key in Structure._extra_attrs:
                setattr(self, key, value)

        self.kargs = {attr: getattr(self, attr, None) for attr in Structure._extra_attrs}

    def __repr__(self):
        return f"------------------------------------------------------------\n" \
               f"<Structure>                                                 \n" \
               f"-Lattice-                                                   \n" \
               f"{self.lattice.matrix}                                       \n" \
               f"-Atoms-                                                     \n" \
               f"{self.atoms}                                                \n" \
               f"------------------------------------------------------------" \
            if self.lattice is not None else f"<Structure object>"

    @property
    def mass_center(self):
        self.ignore_index = getattr(self, "ignore_index", None)
        self.ignore_mol = getattr(self, "ignore_mol", None)
        if self.ignore_mol:
            logger.debug("<ignore_mol> set, Calculate the slab masss center")
            return self.slab.mass_center
        elif isinstance(self.ignore_index, list):
            index = list(set(self.index).difference(set(self.ignore_index)))
            logger.debug("<ignore_index> set, Calculate the structure mass center which excluding the ignore_index")
            return np.sum(self.coords.frac_coords[index], axis=0) / len(index)
        else:
            logger.debug("Calculate the structure mass center")
            return np.sum(self.coords.frac_coords, axis=0) / len(self)

    @property
    def slab(self):
        if self.style.startswith("Slab"):
            kargs = {key: np.array(value)[self.slab_index] for key, value in self.kargs.items() if value is not None}
            return Slab(elements=np.array(self.elements)[self.slab_index],
                        orders=self.slab_index,
                        coords=self.coords[self.slab_index],
                        lattice=self.lattice, **kargs)
        else:
            return None

    @property
    def molecule(self):
        self.anchor = getattr(self, "anchor", None)
        if self.style.endswith("Mol") and self.mol_index and set(self.index).difference(self.mol_index):
            kargs = {key: np.array(value)[self.mol_index] for key, value in self.kargs.items() if value is not None}
            return Molecule(elements=np.array(self.elements)[self.mol_index],
                            orders=self.mol_index,
                            coords=self.coords[self.mol_index],
                            lattice=self.lattice,
                            anchor=self.anchor, **kargs)
        else:
            return None

    def find_neighbour_table(self, neighbour_num: int = 12, adj_matrix=None):
        neighbour_table = NeighbourTable(list)
        for atom_i in self.atoms:
            neighbour_table_i = []
            atom_j_list = self.atoms if adj_matrix is None else [self.atoms[atom_j_order] for atom_j_order in
                                                                 adj_matrix[atom_i.order]]
            for atom_j in atom_j_list:
                if atom_i != atom_j:
                    image = Atom.search_image(atom_i, atom_j)
                    atom_j_image = Atom(formula=atom_j.formula, frac_coord=atom_j.frac_coord + image).set_coord(
                        lattice=self.lattice)
                    distance = np.linalg.norm(atom_j_image.cart_coord - atom_i.cart_coord)
                    logger.debug(f"distance={distance}")
                    neighbour_table_i.append((atom_j, distance, (atom_j_image.cart_coord - atom_i.cart_coord)))
            neighbour_table_i = sorted(neighbour_table_i,
                                       key=lambda x: x[1]) if adj_matrix is None else neighbour_table_i
            neighbour_table[atom_i] = neighbour_table_i[:neighbour_num]

        if adj_matrix is None:
            sorted_neighbour_table = NeighbourTable(list)
            for key, value in neighbour_table.items():
                sorted_neighbour_table[key] = sorted(value, key=lambda x: x[1])
            setattr(self, "neighbour_table", sorted_neighbour_table)
        else:
            setattr(self, "neighbour_table", neighbour_table)

    @staticmethod
    def from_POSCAR(fname, style=None, mol_index=None, **kargs):
        logger.debug(f"Handle the {fname}")
        with open(fname) as f:
            cfg = f.readlines()
        lattice = Lattice.read_from_string(cfg[2:5])

        formula = [(name, int(count)) for name, count in zip(cfg[5].split(), cfg[6].split())]
        formula = sum([[formula] * count for (formula, count) in formula], [])

        selective = cfg[7].lower()[0] == "s"
        if selective:
            coor_type = cfg[8].rstrip()
            coords = np.array(list([float(item) for item in coor.split()[:3]] for coor in cfg[9:9 + len(formula)]))

            frac_coord = coords if coor_type.lower()[0] == "d" else None
            cart_coord = coords if coor_type.lower()[0] == "c" else None
            TF = np.array(list([item.split()[3:6] for item in cfg[9:9 + len(formula)]]))
        else:
            raise NotImplementedError \
                ("The POSCAR file which don't have the selective seaction cant't handle in this version.")

        atoms = Atoms(formula=formula, frac_coord=frac_coord, cart_coord=cart_coord)
        atoms.set_coord(lattice)

        return Structure(style, mol_index=mol_index, atoms=atoms, lattice=lattice, TF=TF, **kargs)

    @staticmethod
    def from_adj_matrix(structure, adj_matrix, adj_matrix_tuple, bond_dist3d, known_first_order):
        """
        Construct a new structure from old structure's adj_matrix

        @parameter
            adj_matrix:         shape: (N, M)
            adj_matrix_tuple:   shape: (N, M, 2)
            bond_dist3d:        shape: (N, M, 3)
        """

        adj_matrix_tuple_flatten = adj_matrix_tuple.reshape(-1, 2)
        bond_dist3d_flatten = bond_dist3d.reshape(-1, 3)

        # construct the search-map
        known_order = []  # search-map, shape: (N-1, 2)
        known_index = [known_first_order]  # shape: (N,)
        known_index_matrix = []  # search-map corresponding to the index of adj_matrix_tuple
        for index, item in enumerate(adj_matrix_tuple_flatten):
            if item[0] not in known_index and item[1] in known_index:
                known_index.append(item[0])
                known_order.append((item[1], item[0]))
                real_index = item[1] * adj_matrix.shape[1] + np.where(adj_matrix[item[1]] == item[0])[0][0]
                known_index_matrix.append(real_index)
            if item[1] not in known_index and item[0] in known_index:
                known_index.append(item[1])
                known_order.append((item[0], item[1]))
                known_index_matrix.append(index)
            if len(known_index) == adj_matrix.shape[0]:
                break

        # calculate the coord from the search-map
        known_first_atom = structure.atoms[known_first_order]
        known_dist3d = bond_dist3d_flatten[known_index_matrix]  # diff matrix, shape: (N-1, 3)
        known_atoms = [known_first_atom]
        for item, diff_coord in zip(known_order, known_dist3d):
            atom_new = copy.deepcopy(structure.atoms[item[1]])  # unknown atom
            atom_new.frac_coord = None
            for atom_known in known_atoms:
                if atom_known.order == item[0]:
                    atom_new.cart_coord = atom_known.cart_coord + diff_coord
                    known_atoms.append(atom_new)
        assert len(known_atoms) == adj_matrix.shape[0], "Search-map construct failure, please check the code!"

        sorted_atoms = sorted(known_atoms, key=lambda atom: atom.order)
        sorted_atoms = [atom.set_coord(structure.lattice) for atom in sorted_atoms]
        atoms = Atoms.from_list(sorted_atoms)

        return Structure(style="Slab", atoms=atoms, lattice=structure.lattice, TF=structure.TF)

    def to_POSCAR(self, fname, system=None, factor=1):
        system = system if system is not None else " ".join(
            [f"{key} {value}" for key, value in self.atoms.size.items()])
        lattice = self.lattice.to_strings
        elements = [(key, str(len(list(value)))) for key, value in itertools.groupby(self.atoms.formula)]
        element_name, element_count = list(map(list, zip(*elements)))
        element_name, element_count = " ".join(element_name), " ".join(element_count)
        selective = getattr(self, "TF", None) is not None
        coords = "\n".join([" ".join([f"{item:15.12f}" for item in atom.frac_coord]) for atom in self.atoms])
        if selective:
            coords = "".join([coord + "\t" + "   ".join(TF) + "\n" for coord, TF in zip(coords.split("\n"), self.TF)])

        with open(fname, "w") as f:
            f.write(f"{system}\n")
            f.write(f"{factor}\n")
            f.write(lattice)
            f.write(f"{element_name}\n")
            f.write(f"{element_count}\n")
            if selective:
                f.write("Selective Dynamics\n")
            f.write("Direct\n")
            f.write(coords)

        logger.debug(f"{fname} write finished!")


class NeighbourTable(defaultdict):

    def __repr__(self):
        return " ".join([f"{key} <---> <{value[0]}> \n" for key, value in self.items()])

    @property
    def index(self):  # adj_matrix
        return np.array([[value[0].order for value in values] for key, values in self.items()])

    @property
    def index_tuple(self):  # adj_matrix_tuple
        return np.array([[(key.order, value[0].order) for value in values] for key, values in self.items()])

    @property
    def dist(self):
        return np.array([[value[1] for value in values] for _, values in self.items()])

    @property
    def dist3d(self):
        return np.array([[value[2] for value in values] for _, values in self.items()])
