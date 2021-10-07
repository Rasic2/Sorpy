import copy
import pickle
import math
import numpy as np
import itertools

from common.base import Element, Atom, AtomSetBase, Lattice, Coordinates
from common.utils import Format_defaultdict
from common.logger import logger



class Molecule(AtomSetBase):

    def __init__(self, elements=None, orders=None, coords: Coordinates = None, anchor=None, rotate=None, **kargs):

        super().__init__(elements=elements, orders=orders, coords=coords, **kargs)

        if isinstance(anchor, int) and anchor in self.orders:
            self.anchor = self.orders.index(anchor)
        elif isinstance(anchor, Atom):
            self.anchor = anchor
        else:
            self.anchor = None
        self.rotate = rotate if rotate is not None else np.identity(3)

    def __repr__(self):
        return f"------------------------------------------------------------\n" \
               f"<Molecule>                                                  \n" \
               f"-Atoms-                                                     \n" \
               f"{self.atoms}                                                \n" \
               f"------------------------------------------------------------"

    def __getitem__(self, index):
        return self.atoms[index]

    @property
    def pair(self):
        if self.anchor is not None:
            atom_j = self.atoms[self.anchor] if isinstance(self.anchor, int) else self.anchor
            pair_list = [(atom_j, atom_i) for atom_i in self.atoms if atom_i != atom_j]
            return pair_list
        else:
            pair_list = []
            for ii in itertools.product(self.atoms, self.atoms):
                if ii[0] != ii[1]:
                    pair_list.append(ii)
            pair_list = (tuple(sorted(item)) for item in pair_list)
            return set(pair_list)

    @property
    def vector(self):
        """ vector in Cartesian format """
        lattice = self.coords.lattice
        pair = []
        for atom_i, atom_j in self.pair:  # handle the PBC principle, Reset the molecule.atoms !!!
            element = atom_j.element
            order = atom_j.order
            frac_coord = np.copy(atom_j.frac_coord)
            frac_coord = np.where(frac_coord - atom_i.frac_coord > 0.5, frac_coord - 1, frac_coord)
            frac_coord = np.where(frac_coord - atom_i.frac_coord < -0.5, frac_coord + 1, frac_coord)
            coord = Coordinates(frac_coords=frac_coord, lattice=lattice)
            atom_j = Atom(element=element, order=order, coord=coord)
            pair.append((atom_i, atom_j))
        return [(atom_i, atom_j, np.dot(atom_j.cart_coord - atom_i.cart_coord, self.rotate)) for atom_i, atom_j in pair]

    @property
    def dist(self):
        return [(atom_i, atom_j, np.linalg.norm(vector)) for atom_i, atom_j, vector in self.vector]

    @property
    def theta(self):
        return [(atom_i, atom_j, math.degrees(math.atan2(math.sqrt(dist ** 2 - vector[2] ** 2), vector[2])))
                    for (atom_i, atom_j, vector), (_, _, dist) in zip(self.vector, self.dist)]

    @property
    def phi(self):
        return [(atom_i, atom_j, math.degrees(math.atan2(vector[1], vector[0])))
                    for (atom_i, atom_j, vector) in self.vector]

    @property
    def inter_coords(self):
        return [(atom_i, atom_j, [dist, theta, phi])
                for (atom_i, atom_j, dist), (_, _, theta), (_, _, phi) in zip(self.dist, self.theta, self.phi)]


class Slab(AtomSetBase):

    def __init__(self, elements=None, orders=None, coords: Coordinates = None, lattice: Lattice = None, **kargs):
        super().__init__(elements=elements, orders=orders, coords=coords, **kargs)
        self.lattice = lattice

        assert len(self.elements) == len(self.frac_coords) == len(self.cart_coords), \
            "The shape of <formulas>, <frac_coords>, <cart_coords> are not equal."

    def __repr__(self):
        return f"------------------------------------------------------------\n" \
               f"<Slab>                                                      \n" \
               f"-Lattice-                                                   \n" \
               f"{self.lattice.matrix}                                       \n" \
               f"-Atoms-                                                     \n" \
               f"{self.atoms}                                                \n" \
               f"------------------------------------------------------------" \
            if self.lattice is not None else f"<Slab object>"

    @property
    def mass_center(self):
        return np.sum(self.coords.frac_coords, axis=0) / len(self)


class Structure(AtomSetBase):
    """TODO <class Coordinates including the frac, cart transfer>"""
    styles = ("Crystal", "Slab", "Mol", "Slab+Mol")
    extra_attrs = ("TF",)

    def __init__(self, style=None, elements=None, coords: Coordinates = None, lattice: Lattice = None, mol_index=None,
                 **kargs):
        """
        :param style:           <Required> Indicate the system style: <"Crystal", "Slab", "Mol", "Slab+Mol">
        :param elements:        <Required> The system Elements list: [Element, Element, etc]
        :param coords:          <Required> The system Coordinates
        :param lattice:         <Required> The Lattice vector
        :param mol_index:       <Optional> The molecule index
        :param kargs:           <Optional> <TF, anchor, ignore_mol, ignore_index>
        """
        self.style = style
        if self.style not in Structure.styles:
            raise AttributeError(f"The '{self.style}' not support in this version, optional style: {Structure.styles}")

        orders = list(range(len(elements)))
        super().__init__(elements=elements, orders=orders, coords=coords, **kargs)
        self.lattice = lattice

        mol_index = mol_index if mol_index is not None else []
        self.index = list(range(len(self.atoms)))
        self.mol_index = mol_index if isinstance(mol_index, (list, np.ndarray)) else [mol_index]
        self.slab_index = list(set(self.index).difference(set(self.mol_index))) if mol_index is not None else self.index

        self.kargs = {attr: getattr(self, attr, None) for attr in Structure.extra_attrs}

    def __repr__(self):
        return f"------------------------------------------------------------\n" \
               f"<Structure>                                                 \n" \
               f"-Lattice-                                                   \n" \
               f"{self.lattice.matrix}                                       \n" \
               f"-Atoms-                                                     \n" \
               f"{self.atoms}                                                \n" \
               f"------------------------------------------------------------" \
            if self.lattice is not None else f"<Structure object>"

    def __sub__(self, other):
        return self.coords - other.coords

    @property
    def inter_coord(self):
        """inter_coords<molecule>"""
        return np.array([item[2] for item in self.molecule.inter_coords])

    @property
    def mcoord(self):
        """frac_coords<slab> + frac_coord<anchor> + inter_coords<molecule>"""
        if len(self.mol_index) > 0 and getattr(self, "anchor", None) is not None:
            mindex = list(set(self.mol_index).difference({self.anchor}))
            _mcoords = np.copy(self.frac_coords)
            _mcoords[mindex] = self.inter_coord # TODO: atoms over than 2, order !!!
            return _mcoords
        else:
            return None

    def vcoord(self, m_template, cut_radius, orders):
        from common.operate import Operator as op

        mol_CO = self.molecule
        mol_CO_coord = pickle.loads(pickle.dumps(mol_CO.frac_coords))
        mol_CO_coord[1:] = np.where(np.array(mol_CO.inter_coords[0][2]) < 0, np.array(mol_CO.inter_coords[0][2]) + 360,
                                    np.array(mol_CO.inter_coords[0][2])) / [1, 180, 360] - [1.142, 0, 0]
        # TODO nomalize the coord in the Model class

        mol_slab = self.create_mol(orders=orders, cut_radius=cut_radius)
        mol_slab_coord = pickle.loads(pickle.dumps(mol_slab.frac_coords))
        mol_slab = op.align_molecule(m_template, mol_slab)
        mol_slab_coord[1:] = (np.array([item[2] for item in mol_slab.vector]) / 2.356 + 1) / 2

        mol_coord = np.concatenate((mol_slab_coord, mol_CO_coord), axis=0)
        return mol_coord, mol_slab.orders

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

    def create_mol(self, orders=None, formula="Ce", cut_radius=5.0):
        if getattr(self, "NNT", None) is None:
            self.find_nearest_neighbour_table(cut_radius=cut_radius)

        max_length = cut_radius
        if orders is None:
            center = None
            for index in self.mol_index:
                for atom in self.NNT.index(index):
                    if atom[0].element.formula == formula and atom[1] <= max_length:
                        max_length = atom[1] # Iter the max_length to locate the nearest Ce atom for the molecule.
                        center = atom

            if center is None:
                raise ValueError(f"Can't find the {formula} element within the cut_radius.")

            elements = [atom[0].element for atom in self.bonds.index(center[0].order) if atom[0].order not in self.mol_index]
            orders = [atom[0].order for atom in self.bonds.index(center[0].order) if atom[0].order not in self.mol_index]
            coords = [atom[0].coord.frac_coords for atom in self.bonds.index(center[0].order) if atom[0].order not in self.mol_index]

            elements.insert(0, center[0].element)
            orders.insert(0, center[0].order)
            coords.insert(0, center[0].coord.frac_coords)
            coords = Coordinates(frac_coords=np.array(coords), lattice=center[0].coord.lattice)
            return Molecule(elements=elements, orders=orders, coords=coords, anchor=center[0].order)
        else:
            # print(type(orders[0]))
            # print(orders)
            elements = np.array(self.elements)[orders]
            coords = Coordinates(frac_coords=np.array(self.coords.frac_coords)[orders], lattice=self.lattice)
            return Molecule(elements=elements, orders=orders, coords=coords, anchor=orders[0])

    def find_nearest_neighbour_table(self, cut_radius=3.0):
        NNT = Format_defaultdict(list)
        for atom_i, value in self.pseudo_bonds.items():
            for atom_j, dist in value:
                if dist<= cut_radius:
                    NNT[atom_i].append((atom_j, dist))
        setattr(self, "NNT", NNT)

    @staticmethod
    def read_from_POSCAR(fname, style=None, mol_index=None, **kargs):
        logger.debug(f"Handle the {fname}")
        with open(fname) as f:
            cfg = f.readlines()
        lattice = Lattice.read_from_string(cfg[2:5])

        elements = [(name, int(count)) for name, count in zip(cfg[5].split(), cfg[6].split())]
        elements = sum([[formula] * count for (formula, count) in elements], [])
        elements = np.array([Element(formula) for formula in elements])

        selective = cfg[7].lower()[0] == "s"
        if selective:
            coor_type = cfg[8].rstrip()
            coords = np.array(list([float(item) for item in coor.split()[:3]] for coor in cfg[9:9 + len(elements)]))

            frac_coords = coords if coor_type.lower()[0] == "d" else None
            cart_coords = coords if coor_type.lower()[0] == "c" else None

            TF = np.array(list([item.split()[3:6] for item in cfg[9:9 + len(elements)]]))
        else:
            raise NotImplementedError \
                ("The POSCAR file which don't have the selective seaction cant't handle in this version.")
        coords = Coordinates(frac_coords=frac_coords, cart_coords=cart_coords, lattice=lattice)

        # if "expand" in kargs.keys():
        #    expand_func = list(kargs['expand'].keys())[0]
        #    if expand_func in Structure.ExpandFunc.keys():
        #        coords = Structure.ExpandFunc[expand_func](frac_coords, lattice=lattice)
        #        for coord in coords:
        #            yield Structure(style, mol_index=mol_index, elements=elements,
        #                            coords=coord, lattice=lattice, TF=TF, **kargs)
        #    else:
        #        logger.warning("Expand step may not happen.")

        return Structure(style, mol_index=mol_index,
                         elements=elements, coords=coords, lattice=lattice, TF=TF, **kargs)

    def write_to_POSCAR(self, fname, system=None, factor=1):
        system = system if system is not None else " ".join([item[0] + str(item[1]) for item in self.atoms_count])
        lattice = self.lattice.to_strings
        elements = [(key.formula, len(list(value))) for key, value in itertools.groupby(self.elements)]
        element_name = " ".join([item[0] for item in elements])
        element_count = " ".join(str(item[1]) for item in elements)
        selective = getattr(self, "TF", None) is not None
        coords = self.coords.to_strings(ctype="frac").split("\n")
        if selective:
            coords = "".join([coord + "\t" + "   ".join(TF) + "\n" for coord, TF in zip(coords, self.TF)])

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

# m_align = Molecule(elements=[Element("C"), Element("O")], frac_coords=np.array([[0, 0, 0], [0, 0, 0.5]]),
#             cart_coords=np.array([[0, 0, 0], [0, 0, 1.142]]))
# print(m_align.pair)
# print(m_align.vector)
# print(m_align.dist)
# print(m_align.theta)
# print(m_align.phi)
# print(m_align.frac_coords)
# print(m_align.cart_coords)
# print(m_align.inter_coords)
# print(m_align.atoms_count)
# print(m_align.TF)
# s = Structure("Slab+Mol", elements=[Element("C"), Element("O")], frac_coords=np.array([[0, 0, 0], [0, 0, 0.5]]),
#             cart_coords=np.array([[0, 0, 0], [0, 0, 1.142]]), mol_index=[0])
# exit()
# s = Structure.read_from_POSCAR(style="Slab+Mol", fname=f"{current_dir}/input/POSCAR_1-1", mol_index=[36, 37])
# p1 = POSCAR(fname=f"{current_dir}/input/POSCAR_1-1")
# p2 = POSCAR(fname=f"{current_dir}/output/CONTCAR_1-1")
# print(p1-p2)
##print(ctype(p[2]))
##s.find_nearest_neighbour_table()
##for key, value in s.NNT.items():
##    print(key, value[0])
##s = p.to_structure(style="Slab+Mol", mol_index=[36,37])
##print(s)
##print(s.style)
##print(s.slab)
##print(s.molecule)
##s.write_to_POSCAR(f"{current_dir}/POSCAR_test")
# exit()
#
#
#
#
# class POSCAR:
#
#    def __init__(self, fname, action="r"):
#        self.fname = fname
#        self.action = action
#        self.__load() if self.action == "r" else None
#
#    def __load(self):
#        with open(self.fname, self.action) as f:
#            self.cfg = f.readlines()
#        self.system = self.cfg[0].rstrip()
#        self.factor = self.cfg[1].rstrip()
#        self.latt = Latt.read_from_string(self.cfg[2:5])
#        self.elements = [(name, int(count)) for name, count in zip(self.cfg[5].split(), self.cfg[6].split())]
#        self.selective = self.cfg[7].lower()[0] == "s"
#        self.coor_type = self.cfg[8].rstrip()
#        assert self.coor_type.lower()[0] == "d"
#        self.frac_coors = self._frac_coors()
#        self.cart_coors = self._cart_coors()
#        self.TF = self._TF()
#
#    @property
#    def ele_count(self):
#        return sum(n for _, n in self.elements)
#
#    def _frac_coors(self):
#        if self.selective:
#            return np.array([[float(ii) for ii in item.split()[:3]] for item in self.cfg[9:9 + self.ele_count]])
#        else:
#            return None
#
#    def _cart_coors(self):
#        if self.frac_coors is not None:
#            return np.dot(self.frac_coors, self.latt.matrix)
#        else:
#            return None
#
#    def _TF(self):
#        if self.selective:
#            return np.array([item.split()[3:6] for item in self.cfg[9:9 + self.ele_count]])
#        else:
#            return None
#
#    def nearest_neighbour_table(self):
#
#        NNT = defaultdict(list)
#        cut_radius = 3.0
#        for ii in range(self.ele_count):
#            for jj in range(self.ele_count):
#                if jj != ii:
#                    dis = distance(self.cart_coors[ii], self.cart_coors[jj])
#                    if dis <= cut_radius:
#                        NNT[ii].append((jj, dis))
#
#        # sorted NNT
#        sorted_NNT = defaultdict(list)
#        for key, value in NNT.items():
#            sorted_NNT[key] = sorted(value, key=lambda x: x[1])
#
#        setattr(self, "NNT", sorted_NNT)
#        return sorted_NNT
#
#    @staticmethod
#    def mcoord_to_coord(array, latt, anchor, intercoor_index):
#
#        assert ftype(array) == np.ndarray
#        assert ftype(latt) == Latt
#
#        total_index = list(range(len(array)))
#        frac_index = list(set(total_index).difference(set(intercoor_index)))
#        anchor_cart_coor = np.dot(array[anchor], latt.matrix)
#        intercoor = array[intercoor_index] * [1, 180, 360] + [1.142, 0, 0]  # TODO setting.yaml modify
#        intercoor = intercoor.reshape(3)
#        r, theta, phi = intercoor[0], math.radians(intercoor[1]), math.radians(intercoor[2])
#        x = r * math.sin(theta) * math.cos(phi)
#        y = r * math.sin(theta) * math.sin(phi)
#        z = r * math.cos(theta)
#        mol_cart_coor = anchor_cart_coor + [x, y, z]
#        mol_frac_coor = np.dot(mol_cart_coor, latt.inverse).reshape((1, 3))
#        return np.concatenate((array[frac_index], mol_frac_coor), axis=0)
#
#    def write(self, template, coor):
#
#        assert ftype(template) == POSCAR
#        assert self.action == "w"
#
#        with open(self.fname, self.action) as f:
#            f.writelines(template.cfg[:9])
#            for ii, jj in zip(coor, template.TF):
#                item = f"{ii[0]:.6f} {ii[1]:.6f} {ii[2]:.6f} {jj[0]} {jj[1]} {jj[2]} \n"
#                f.write(item)
#
#    def align_structure(self, template):
#        indicator = 0  # 第二次对齐的原子
#        repeat = 2  # supercell
#        slab_coor = self.frac_coors[:36]
#        mol_coor = self.frac_coors[36:38]
#        model_region = np.array([1 / repeat, 1 / repeat])
#        mass_center = np.mean(mol_coor, axis=0)[:2]
#        search_vector = np.arange(-1, 1 + 1 / repeat, 1 / repeat)
#        search_matrix = itertools.product(search_vector, search_vector)
#        for ii in search_matrix:
#            if 0 <= (mass_center + ii)[0] <= model_region[0] and 0 <= (mass_center + ii)[1] <= model_region[1]:
#                vector_m = [ii[0], ii[1], 0]
#                break
#        coor_m = mol_coor + vector_m
#        coor_first = np.concatenate((slab_coor, coor_m), axis=0)
#
#        return vector_m, coor_first
#        exit()
#
#        for ii, jj in zip(self.elements, template.elements):
#            if ii[0] != jj[0] or ii[1] != jj[1]:
#                raise NotImplementedError  # TODO 3x3 or len(atoms) 不相等
#
#        self.frac_coors = coor_first
#        self.cart_coors = self._cart_coors()  # Reset the cart_coords
#
#        NNT = defaultdict(list)
#        for ii in range(self.ele_count):
#            for jj in range(template.ele_count):
#                dis = distance(self.cart_coors[ii], self.cart_coors[jj])
#                NNT[ii].append((jj, dis))
#
#        sorted_NNT = defaultdict(list)
#        for key, value in NNT.items():
#            sorted_NNT[key] = sorted(value, key=lambda x: x[1])
#
#        min_NNT = []
#        for key, value in sorted_NNT.items():
#            min_NNT.append((key, value[0][0], value[0][1]))
#
#        for ii, jj, kk in sorted(min_NNT, key=lambda x: x[2]):
#            print(ii, jj)
#        #    if mass_center + model_region * i
#        # print(model_region)
#
#
# if __name__ == "__main__":
#
#    for ii in range(50):
#        print(f"POSCAR_ML_{ii + 1}")
#        p = POSCAR(f"../test_set/ML-2/POSCAR_ML_{ii + 1}")
#        p.nearest_neighbour_table()
#        print(36, p.NNT[36])
#        p1 = POSCAR(fname=f"{current_dir}/input/POSCAR_1-1")
#        p2 = POSCAR(fname=f"{current_dir}/output/CONTCAR_1-1")
#        print(p1 - p2)
#        print(37, p.NNT[37])
