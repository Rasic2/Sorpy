import os
import sys
import numpy as np
#from pymatgen.io.vasp import Poscar

from common.structure import Lattice, Molecule
from common.io_file import POSCAR, CONTCAR
from logger import logger


class FileManager:
    """
        single POSCAR-like file Manager
    """
    files = {'POSCAR': POSCAR,
             'CONTCAR': CONTCAR
    }

    def __init__(self, fname: str, style=None, mol_index=None):
        """
        :param fname:   file name
        """
        self.fname = fname
        try:
            self.ftype = fname.split("_")[0].split("/")[-1]
            self.ftype = FileManager.files[self.ftype]
        except KeyError:
            logger.error(f"The '{self.ftype}' is not including in this version.")
            sys.exit()

        self.index = fname.split("_")[-1]

        if isinstance(mol_index, list):
            self.mol_index = mol_index
        elif isinstance(mol_index, str):
            if "-" in mol_index:
                self.mol_index = list(range(int(mol_index.split("-")[0]) - 1, int(mol_index.split("-")[1])))
            else:
                self.mol_index = [int(mol_index)]
        elif isinstance(mol_index, int):
            self.mol_index = [mol_index]
        else:
            self.mol_index = None
            logger.warning("The Molecule was not align.")

        self.style = style
        from collections import defaultdict
        self.atom_dict = defaultdict(list)

    def __eq__(self, other):
        return self.ftype == other.ftype and self.index == other.index

    def __le__(self, other):
        return self.ftype == other.ftype and self.index <= other.index

    def __gt__(self, other):
        return self.ftype == other.ftype and self.index > other.index

    def __repr__(self):
        return f"{self.ftype}: {self.index}"

    @property
    def file(self):
        return self.ftype(self.fname)

    @property
    def structure(self):
        return self.file.to_structure(style=self.style, mol_index=self.mol_index)
#
#    @property
#    def latt(self):
#        return Latt(self.structure.lattice.matrix)
#
#    @property
#    def sites(self):
#        return self.structure.sites
#
#    @property
#    def species(self):
#        return self.structure.species
#
#    @property
#    def coords(self):
#        return self.structure.frac_coords
#
#    @property
#    def atom_num(self):
#        return len(self.coords)
#
#    @property
#    def molecule(self):
#        if type(self.mol_index) == list and len(self.mol_index):
#            return [(ii + 1, site) for ii, site in zip(self.mol_index, np.array(self.structure)[self.mol_index])]
#        else:
#            return None
#
#    def _setter_slab_index(self):
#        if self.molecule:
#            self.slab_index = list(set(list(range(self.atom_num))).difference(set(self.mol_index)))
#        else:
#            self.slab_index = list(range(self.atom_num))
#
#    @property
#    def slab(self):
#        self._setter_slab_index()
#        return [(ii, site) for ii, site in zip(self.slab_index, np.array(self.structure)[self.slab_index])]
#
#    def align_the_element(self):
#
#        self._setter_slab_index()
#        for ii, item in enumerate(self.species):
#            if ii in self.slab_index:
#                self.atom_dict[item].append(ii)
#            elif ii in self.mol_index:
#                self.atom_dict["mol"].append(ii)
#
#    @property
#    def mcoords(self):
#        """
#        Cal Slab frac_coors + Mol <anchor_frac + bond length + theta + phi>
#        """
#        self._setter_slab_index()
#
#        if self.molecule:
#            m = Molecule(self.coords[self.mol_index], self.latt.matrix)
#            slab_coor = self.coords[self.slab_index]
#            m_anchor = m[0].frac_coord.reshape((1, 3))
#            m.phi_m = m.phi if m.phi >= 0 else 360 + m.phi
#            m_intercoor = np.array([m.bond_length, m.theta, m.phi_m]).reshape((1, 3))
#            m_intercoor = (m_intercoor/ [[1, 180, 360]] - [[1.142, 0, 0]])
#            return np.concatenate((slab_coor, m_anchor, m_intercoor), axis=0)
#
#
#class DirManager:
#    """
#        Input/Output directory manager
#    """
#
#    def __init__(self, dname: str, ftype: str, mol_index=None):
#        """
#        :param dname:       directory name
#        :param ftype:        determine which ctype of file including (e.g. POSCAR or CONTCAR)
#        """
#        self.dname = dname
#        self.type = ftype
#        self.mol_index = mol_index
#        if self.mol_index:
#            logger.info(f"Molecule was align to {self.mol_index} location.")
#
#    def one_file(self, fname):
#        """
#        The single file manager
#
#        :param fname:   file name
#        :return:        FileManager(fname)
#        """
#        return FileManager(f"{self.dname}/{fname}", mol_index=self.mol_index)
#
#
#    def __all_files(self):
#        for fname in os.listdir(self.dname):
#            if fname.startswith(self.type):
#                yield FileManager(f"{self.dname}/{fname}", mol_index=self.mol_index)
#
#    @property
#    def all_files(self):
#        return sorted(list(self.__all_files()))
#
#    @property
#    def count(self):
#        return len(self.all_files)
#
#    @property
#    def coords(self):
#        return np.array([file.coords for file in self.all_files])
#
#    def split_slab_mol(self):
#        for file in self.all_files:
#            file.align_the_element()
#            return file.atom_dict
#
#    @property
#    def mcoords(self):
#        return np.array([file.mcoords for file in self.all_files])