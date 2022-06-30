import re
import os
import yaml
import numpy as np
from pathlib import Path
from multiprocessing import Pool as ProcessPool

from common.io_file import POSCAR, CONTCAR
# from common.operate import Operator as op
from common.logger import logger
from common.utils import Format_list

yaml.warnings({'YAMLLoadWarning': False})


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
            logger.warning(f"The '{ftype}' is excluding in the <{dname}> directory.")
            return None

    def __init__(self, fname: Path, style=None, mol_index=None, **kargs):

        self.fname = fname
        self.kargs = kargs

        self.ftype = fname.name.split("_")[0]
        self.ftype = FileManager.files[self.ftype]

        self.index = fname.name.split("_")[-1]
        self.num_index = self.index
        try:
            self.num_index = int(self.index)
        except ValueError:
            self.num_index = [int(item) for item in re.split("[^0-9]", self.index)]

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
            # logger.warning("The Molecule was not align.")

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
        return self.file.to_structure(style=self.style, mol_index=self.mol_index, **self.kargs)


class DirManager:

    def __init__(self, dname: Path, template=None, style=None, mol_index=None, **kargs):

        self.dname = dname
        self.template = template
        self.style = style
        self.mol_index = mol_index
        self.kargs = kargs
        if self.mol_index:
            logger.info(f"Molecule was align to {self.mol_index} location.")

        self._all_files = None

    def __getitem__(self, index):
        return self.all_files[index].structure

    def __len__(self):
        return len(self.all_files)

    def single_file(self, fname: Path):
        return FileManager(self.dname / fname, style=self.style, mol_index=self.mol_index, **self.kargs)

    @property
    def all_files(self):
        if self._all_files is None:
            all_files = [FileManager(self.dname / fname, style=self.style, mol_index=self.mol_index, **self.kargs)
                         for fname in os.listdir(self.dname)]
            all_files = [file for file in all_files if file is not None]
            self._all_files = Format_list(sorted(all_files, key=lambda x: x))
        return self._all_files

    @property
    def all_files_path(self):
        return [Path(file.fname) for file in self.all_files]

    @property
    def coords(self):
        logger.info("Align the structure to the template structure.")
        pool = ProcessPool(processes=os.cpu_count())

        results = [pool.apply_async(op.align, args=(self.template, file.structure)) for file in self.all_files]
        temp_coords = [result.get() for result in results]

        pool.close()
        pool.join()

        return temp_coords

    @property
    def mcoords(self):
        """frac_coords<slab> + frac_coord<anchor> + inter_coords<molecule>"""
        if len(self.mol_index) > 0 and "anchor" in self.kargs:
            mindex = list(set(self.mol_index).difference({self.kargs["anchor"]}))
            _mcoords = np.copy(self.frac_coords)
            _mcoords[:, mindex, :] = self.inter_coords[:, :, :]
            return _mcoords
        else:
            return None

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


class ParameterManager:
    _parameters = {'SpaceGroup': str,
                   'LatticeParameter': float,
                   'Species': list,
                   'Coordinates': list,
                   'MillerIndex': tuple,
                   'SlabThickness': float,
                   'VacuumHeight': float,
                   'supercell': tuple,
                   'z_height': float,
                   'TestNum': int,
                   }

    def __init__(self, filename):
        """
        TODO default value and ctype split!!!

        :param filename:            setting_110.yaml
        """
        self.fname = filename
        self.MillerIndex = None
        self.TestNum = None
        self.SpaceGroup = None
        self.z_height = None
        self.LatticeParameter = None
        self.Species = None
        self.SlabThickness = None
        self.Coordinates = None
        self.VacuumHeight = None
        self.supercell = None

        self.load()
        self.check_trans()

    def load(self):
        f = open(self.fname, "r", encoding='utf-8')
        cfg = f.read()
        parameters = yaml.load(cfg)
        f.close()
        for key, value in parameters.items():
            if key in ParameterManager._parameters.keys():
                setattr(self, key, value)
            else:
                logger.warning(f"'{key}' in {self.fname} is ignored.")

    def check_trans(self):
        for key, value in ParameterManager._parameters.items():
            if hasattr(self, key) and getattr(self, key, None) is not None and not isinstance(self.__dict__[key], value):
                if value == tuple:
                    self.__dict__[key] = tuple(eval(self.__dict__[key]))
                else:
                    raise IndexError
