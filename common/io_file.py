import numpy as np
from common.structure import Structure


class POSCAR:

    def __init__(self, fname):
        self.fname = fname

    def __repr__(self):
        return f"<{self.ftype} '{self.fname}'>"

    def __getitem__(self, index):
        return self.to_strings[index]

    def __sub__(self, other):
        self.structure = self.to_structure(style="Slab")
        other.structure = other.to_structure(style="Slab")
        if np.all(self.structure.lattice.matrix == other.structure.lattice.matrix):
            return self.structure - other.structure
        else:
            raise ArithmeticError(f"{self} and {other} not have the same lattice vector!")

    @property
    def ftype(self):
        return self.__class__.__name__

    @property
    def to_strings(self):
        with open(self.fname) as f:
            cfg = f.readlines()
        return cfg

    def to_structure(self, style=None, mol_index=None, **kargs):
        return Structure.read_from_POSCAR(self.fname, style=style, mol_index=mol_index, **kargs)


class CONTCAR(POSCAR):
    def __init__(self, fname):
        super().__init__(fname=fname)
