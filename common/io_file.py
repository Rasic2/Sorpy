from collections.abc import Iterable
from multiprocessing import Pool as ProcessPool

import numpy as np

from common.base import Lattice, Atoms
from common.structure import Structure


class VASPFile:

    def __init__(self, fname):
        self.fname = fname

    def __repr__(self):
        return f"<{self.ftype} '{self.fname}'>"

    @property
    def ftype(self):
        return self.__class__.__name__

    @property
    def strings(self):
        with open(self.fname) as f:
            cfg = f.readlines()
        return cfg


class POSCAR(VASPFile):

    def __init__(self, fname):
        super().__init__(fname)

    def __getitem__(self, index):
        return self.strings[index]

    def __sub__(self, other):
        self.structure = self.to_structure()
        other.structure = other.to_structure()
        if np.all(self.structure.lattice.matrix == other.structure.lattice.matrix):
            return self.structure - other.structure
        else:
            raise ArithmeticError(f"{self} and {other} not have the same lattice vector!")

    def to_structure(self):
        return Structure.from_POSCAR(self.fname)


class CONTCAR(POSCAR):
    def __init__(self, fname):
        super().__init__(fname=fname)


class XDATCAR(VASPFile):
    def __init__(self, fname, **kargs):
        super().__init__(fname)

        self.kargs = kargs
        self.system = self.strings[0].rstrip()
        self.factor = self.strings[1].rstrip()
        self.lattice = Lattice.from_string(self.strings[2:5])
        element_name = self.strings[5].split()
        element_count = [int(item) for item in self.strings[6].split()]
        self.element = sum([[name] * count for name, count in zip(element_name, element_count)], [])
        self.frames = [i for i in range(len(self.strings)) if self.strings[i].find("Direct") != -1]

        self._structures = []

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.structures[index]

    @property
    def structures(self):
        if len(self._structures) == 0:
            for frame in self.frames:
                frac_coord = np.array([[float(item) for item in line.split()] for line in
                                       self.strings[frame + 1:frame + 1 + len(self.element)]])
                atoms = Atoms(formula=self.element, frac_coord=frac_coord)
                self._structures.append(Structure(atoms=atoms, lattice=self.lattice, **self.kargs))
        return self._structures

    def split_file(self, index, fname, system=None, factor=1., num_workers=4):
        if isinstance(index, int):
            self[index].to_POSCAR(fname=fname, system=system, factor=factor)
        elif isinstance(index, (Iterable, slice)):
            pool = ProcessPool(processes=num_workers)
            for index_i, fname_i in zip(index, fname):
                pool.apply_async(self[index_i].to_POSCAR, args=(fname_i, system, factor))
            pool.close()
            pool.join()
