import copy
import itertools
import numpy as np
from collections import defaultdict

from common.utils import Format_defaultdict
from common.logger import logger
from common.base import Coordinates
from common.structure import AtomSetBase, Structure


class Operator:
    def __init__(self):
        pass

    @staticmethod
    def dist(si, sj) -> defaultdict:
        assert isinstance(si, AtomSetBase) and isinstance(sj, AtomSetBase), \
            "The object A and B are not instance of the <class 'AtomSetBase'>"
        dists = defaultdict(dict)
        for ai in si.atoms:
            for aj in sj.atoms:
                dist = np.linalg.norm(aj.cart_coord - ai.cart_coord)
                dists[ai][aj] = dist

        sorted_dists = defaultdict(dict)
        for key, value in dists.items():
            sorted_dists[key] = {key: value for key, value in sorted(value.items(), key=lambda x: x[1])}

        return sorted_dists

    @staticmethod
    def __pbc_apply(template, for_pbc):
        """Apply the Periodic Boundary Condition <PBC_apply>"""
        index = np.where(np.abs(for_pbc.coords.frac_coords - template.coords.frac_coords) > 0.5)
        temp_index = [(i, j) for i, j in zip(index[0], index[1])]
        final_index = [item for item in temp_index if item[0] not in for_pbc.mol_index]
        index = (np.array([i for i, _ in final_index], dtype=np.int64), np.array([i for _, i in final_index], dtype=np.int64))

        new_frac_coords = np.copy(for_pbc.coords.frac_coords)
        logger.debug("PBC Apply")
        for (i, j) in zip(*index):
            iter_num = 0
            while True:
                iter_num += 1
                diff = new_frac_coords[i, j] - template.coords.frac_coords[i, j]
                increment = 1 if diff < 0 else - 1
                new_frac_coords[i, j] += increment
                diff = new_frac_coords[i, j] - template.coords.frac_coords[i, j]
                if abs(diff) < 0.5:
                    break
                if iter_num >= 10:
                    logger.error("Coordinates PBC-apply Error! Please check the input structure.")
                    raise StopIteration("Iter over than 10 times, Something wrong happens!")

        new_struct = copy.deepcopy(for_pbc)
        setattr(new_struct, "coords", Coordinates(frac_coords=new_frac_coords, lattice=for_pbc.lattice))

        return new_struct

    @staticmethod
    def __trans_mass_center(template, for_trans):
        """Translate the structure along the mass_center direction"""
        logger.debug("Trans the structure along the mass center direction")
        tmass = template.mass_center
        nmass = for_trans.mass_center
        new_frac_coords = for_trans.coords.frac_coords + tmass - nmass
        new_struct = copy.deepcopy(for_trans)
        setattr(new_struct, "coords", Coordinates(frac_coords=new_frac_coords, lattice=for_trans.lattice))

        return new_struct

    @staticmethod
    def __tailor_atom_order(template, for_tailor):
        """Tailor the atom order to achieve the atom mapping"""
        mapping_list = []
        dists = Operator.dist(template, for_tailor)
        logger.debug("Tailor the atom order to achieve the atom mapping")
        # If cart-dist < 0.2, mapping such atom as the same
        for atom_i, value in dists.items():
            for atom_j, dist in value.items():
                if dist < 0.2 and atom_i.element == atom_j.element: # Fix bug, test atom_i and atom_j is the same element
                    mapping_list.append([atom_i.order, atom_j.order])
                    break  # the shortest distance

        # For the atoms whose distance over than 0.2, the shortest distance atom will be mapping
        template_remain = set(template.orders).difference(set([i for (i, _) in mapping_list]))
        for_tailor_remain = set(for_tailor.orders).difference(set([i for (_, i) in mapping_list]))
        for index_i in template_remain:
            min_dist, min_index = 100, None
            for index_j in for_tailor_remain:
                if template.atoms[index_i].element == for_tailor.atoms[index_j].element:
                    dist = dists[template.atoms[index_i]][for_tailor.atoms[index_j]]
                    if dist < min_dist:
                        min_dist, min_index = dist, index_j
            mapping_list.append([index_i, min_index])
        index = [i for _, i in mapping_list]
        new_kargs = copy.deepcopy(for_tailor.__dict__)
        for key in ('elements', 'coords', 'TF'):
            if key in new_kargs.keys():
                try:
                    new_kargs[key] = for_tailor.__dict__[key][index]
                except:
                    print(f"key = {key}, index = {index}")
                    raise
        if "orders" in new_kargs.keys():
            del new_kargs["orders"]

        return Structure(**new_kargs)

    @staticmethod
    def align(template, for_align):
        if template is None and isinstance(for_align, Structure):
            return for_align

        assert isinstance(template, Structure) and isinstance(for_align, Structure), \
            f"The input parameters should be the instance of the <class Structure>, \n" \
            f"but your input parameters are {type(template)} and {type(for_align)}"

        assert template.lattice == for_align.lattice, \
            "The lattice vector of input structure is not consistent with the template structure. \n" \
            f"template.lattice \n {template.lattice} \n" \
            f"for_align.lattice \n {for_align.lattice}"

        fmass = template.mass_center
        nmass = for_align.mass_center
        new_struct = copy.deepcopy(for_align)
        count = 0
        while np.any(np.abs(fmass - nmass) > 10e-06):
            logger.debug(f"fmass = {fmass}")
            logger.debug(f"nmass = {nmass}")
            new_struct = Operator.__pbc_apply(template, new_struct)
            new_struct = Operator.__trans_mass_center(template, new_struct)
            new_struct = Operator.__tailor_atom_order(template, new_struct)
            fmass = template.mass_center
            nmass = new_struct.mass_center
            count += 1
            if count > 10:
                logger.error("Coordinates align Error! Please check the input structure.")
                raise StopIteration("Iter over than 10 times, Something wrong happens!")
        return new_struct

    @staticmethod
    def find_trans_vector(coord: np.ndarray):
        repeat = 2 # Make repeat to be a parameter in future
        ori_coord = copy.deepcopy(coord)

        anchor = ori_coord[:, 36, :2]
        model_region = np.array([1 / repeat, 1 / repeat])
        search_vector = np.arange(-1, 1 + 1 / repeat, 1 / repeat)
        trans_vectors = []
        for index, item in enumerate(anchor):
            search_matrix = itertools.product(search_vector, search_vector)
            for ii in search_matrix:
                if 0 <= (item + ii)[0] <= model_region[0] and 0 <= (item + ii)[1] <= model_region[1]:
                    vector_m = [ii[0], ii[1], 0]
                    ori_coord[index, 36, :] += vector_m
                    trans_vectors.append(vector_m)
                    break

        return ori_coord, trans_vectors
