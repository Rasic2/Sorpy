import os
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool

from common.base import Lattice#, Coordinates
from common.structure import Structure
from common.logger import logger


class VASPFile:

    def __init__(self, fname):
        self.fname = fname

    def __repr__(self):
        return f"<{self.ftype} '{self.fname}'>"

    @property
    def ftype(self):
        return self.__class__.__name__

    @property
    def to_strings(self):
        with open(self.fname) as f:
            cfg = f.readlines()
        return cfg


class POSCAR(VASPFile):

    def __init__(self, fname):
        super().__init__(fname)

    def __getitem__(self, index):
        return self.to_strings[index]

    def __sub__(self, other):
        self.structure = self.to_structure(style="Slab")
        other.structure = other.to_structure(style="Slab")
        if np.all(self.structure.lattice.matrix == other.structure.lattice.matrix):
            return self.structure - other.structure
        else:
            raise ArithmeticError(f"{self} and {other} not have the same lattice vector!")

    def to_structure(self, style=None, mol_index=None, **kargs):
        return Structure.from_POSCAR(self.fname, style=style, mol_index=mol_index, **kargs)


class CONTCAR(POSCAR):
    def __init__(self, fname):
        super().__init__(fname=fname)


class XDATCAR(VASPFile):
    def __init__(self, fname, **kargs):
        super().__init__(fname)
        self.kargs = kargs
        cfg = self.to_strings

        self.system = cfg[0].rstrip()
        self.factor = cfg[1].rstrip()
        self.lattice = Lattice.read_from_string(cfg[2:5])
        self.elements = Elements.read_from_strings(formulas=cfg[5], counts=cfg[6])
        self.frames = [i for i in range(len(cfg)) if cfg[i].find("Direct") != -1]

    def __len__(self):
        assert len(list(self.structures)) == len(self.frames)
        return len(self.frames)

    def __getitem__(self, index):
        return list(self.structures)[index]

    @property
    def structures(self):
        cfg = self.to_strings
        for frame in self.frames:
            coor = Coordinates.read_from_strings(strings=cfg[frame + 1: frame + 1 + len(self.elements)], ctype="frac",
                                                 lattice=self.lattice)
            yield Structure(elements=self.elements, coords=coor, lattice=self.lattice, **self.kargs)

    def to_POSCAR_thread(self, index, prefix, dname):
        fname_index = index + 1 if prefix.startswith("POSCAR") else index
        fname = prefix + os.path.basename(self.fname).split("_")[1] + "-" + f"{fname_index}"
        self[index].to_POSCAR(fname=dname / fname)

    def to_POSCAR(self, prefix="POSCAR_", dname=None, indexes=None):
        pool = ProcessPool(processes=4)
        for index in indexes:
            pool.apply_async(self.to_POSCAR_thread, args=(index, prefix, dname))

        pool.close()
        pool.join()
        logger.info(f"{os.path.basename(self.fname)} ---> {prefix.replace('_', '')} finished!")

    def to_CONCAR_thread(self, i, prefix, dname):
        fname = prefix + os.path.basename(self.fname).split("_")[1] + "-" + f"{i + 1}"
        self[-1].to_POSCAR(fname=dname / fname)

    def to_CONTCAR(self, prefix="CONTCAR_", dname=None, indexes=None):
        if len(indexes) == 1 and -1 in indexes:
            pool = ProcessPool(processes=4)
            for i in range(len(self)-1):
                pool.apply_async(self.to_CONCAR_thread, args=(i, prefix, dname))
            pool.close()
            pool.join()
            logger.info(f"{os.path.basename(self.fname)} ---> {prefix.replace('_', '')} finished!")
        else:
            self.to_POSCAR(prefix, dname, indexes)
