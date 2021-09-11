import re
import os
import numpy as np
from pathlib import Path

from common.io_file import POSCAR, CONTCAR
from logger import logger
from utils import Format_list


class FileManager:
    files = {'POSCAR': POSCAR,
             'CONTCAR': CONTCAR
    }

    def __new__(cls, *args, **kargs):
        ftype = args[0].name.split("_")[0]
        dname = args[0].parent.name
        if ftype in FileManager.files:
            return super().__new__(cls)
        else:
            logger.error(f"The '{ftype}' is excluding in the <{dname}> directory.")
            return None

    def __init__(self, fname: Path, style=None, mol_index=None):

        self.fname = fname

        self.ftype = fname.name.split("_")[0]
        self.ftype = FileManager.files[self.ftype]

        self.index = fname.name.split("_")[-1]
        try:
            self.num_index = int(self.index)
        except ValueError:
            self.num_index = [int(item) for item in re.split("[^0-9]", self.index)]
        except:
            self.num_index = self.index

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
        return self.ftype == other.ftype and self.num_index == other.num_index

    def __le__(self, other):
        return self.ftype == other.ftype and self.num_index <= other.num_index

    def __gt__(self, other):
        return self.ftype == other.ftype and self.num_index > other.num_index

    def __repr__(self):
        return f"{self.ftype}: {self.index}"

    @property
    def file(self):
        return self.ftype(self.fname)

    @property
    def structure(self):
        return self.file.to_structure(style=self.style, mol_index=self.mol_index)

class DirManager:

    def __init__(self, dname: Path, style=None, mol_index=None):

        self.dname = dname
        self.style = style
        self.mol_index = mol_index
        if self.mol_index:
            logger.info(f"Molecule was align to {self.mol_index} location.")

    def __len__(self):
        return len(self.all_files)

    def single_file(self, fname: Path):
        return FileManager(self.dname/fname, style=self.style, mol_index=self.mol_index)

    @property
    def all_files(self):
        all_files = [FileManager(self.dname/fname, style=self.style, mol_index=self.mol_index)
               for fname in os.listdir(self.dname)]
        all_files = [file for file in all_files if file is not None]
        return Format_list(sorted(all_files, key=lambda x : x))

    @property
    def coords(self):
        return Format_list([file.structure.coords for file in self.all_files])

    @property
    def frac_coords(self):
        return np.array([coord.frac_coords for coord in self.coords])

    @property
    def cart_coords(self):
        return np.array([coord.cart_coords for coord in self.coords])

    @property
    def inter_coords(self):
        return np.array([[inter_coord for _, _, inter_coord in file.structure.molecule.inter_coords]
                         for file in self.all_files])