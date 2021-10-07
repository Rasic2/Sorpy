import re
import os
import yaml
import numpy as np
from pathlib import Path
from multiprocessing import Pool as ProcessPool

from common.io_file import POSCAR, CONTCAR
from common.operate import Operator as op
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
            logger.warning("The Molecule was not align_structure.")

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
            logger.info(f"Molecule was align_structure to {self.mol_index} location.")

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
    def structures(self):
        logger.info("Align the structure to the template structure.")
        pool = ProcessPool(processes=os.cpu_count())

        results = [pool.apply_async(op.align_structure, args=(self.template, file.structure)) for file in self.all_files]
        temp_structures = [result.get() for result in results]

        pool.close()
        pool.join()

        return temp_structures

    @property
    def coords(self):
        return [structure.coords for structure in self.structures]

    @property
    def frac_coords(self):
        return np.array([coord.frac_coords for coord in self.coords])

    @property
    def cart_coords(self):
        return np.array([coord.cart_coords for coord in self.coords])

    @property
    def inter_coords(self):
        """inter_coords<molecule>"""
        return np.array([file.structure.inter_coord for file in self.all_files])

    @property
    def mcoords(self):
        """frac_coords<slab> + frac_coord<anchor> + inter_coords<molecule>"""
        return np.array([file.structure.mcoord for file in self.all_files])

    def vcoords(self, orders=None, cut_radius=5.0):
        """frac_coords<Ce1> + vector<O7> + frac_coord<anchor> + inter_coords<molecule>"""
        m_template = self.template.create_mol(cut_radius=cut_radius)
        pool = ProcessPool(processes=os.cpu_count())
        if orders is not None:
            results = [pool.apply_async(file.structure.vcoord, args=(m_template, cut_radius, order))
                       for file, order in zip(self.all_files, orders)]
        else:
            results = [pool.apply_async(file.structure.vcoord, args=(m_template, cut_radius, orders))
                       for file in self.all_files]

        temp_results = [result.get() for result in results]
        mcoords = [item for item, _ in temp_results]
        orders = [item for _, item in temp_results]

        pool.close()
        pool.join()

        return np.array(mcoords), orders


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
