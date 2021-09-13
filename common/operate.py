import copy
import numpy as np
from utils import Format_defaultdict

from logger import logger
from common.base import Coordinates
from common.structure import AtomSetBase, Structure


class Operator:
    def __init__(self):
        pass

    @staticmethod
    def dist(si, sj) -> Format_defaultdict:
        assert isinstance(si, AtomSetBase) and isinstance(sj, AtomSetBase), \
            "The object A and B are not instance of the <class 'AtomSetBase'>"
        dists = Format_defaultdict(dict)
        for ai in si.atoms:
            for aj in sj.atoms:
                dist = np.linalg.norm(aj.cart_coord - ai.cart_coord)
                dists[ai][aj] = dist

        sorted_dists = Format_defaultdict(dict)
        for key, value in dists.items():
            sorted_dists[key] = {key:value for key, value in sorted(value.items(), key=lambda x: x[1])}

        return sorted_dists

    @staticmethod
    def __pbc_apply(template, for_pbc):
        """Apply the Periodic Boundary Condition <PBC_apply>"""
        index = np.where(np.abs(for_pbc.coords.frac_coords - template.coords.frac_coords) > 0.5)
        new_frac_coords = np.copy(for_pbc.coords.frac_coords)

        for (i, j) in zip(*index):
            iter = 0
            while True:
                iter += 1
                diff = new_frac_coords[i, j] - template.coords.frac_coords[i, j]
                increment = 1 if diff < 0 else - 1
                new_frac_coords[i, j] += increment
                diff = new_frac_coords[i, j] - template.coords.frac_coords[i, j]
                if abs(diff) < 0.5:
                    break
                if iter >= 10:
                    logger.error("Coordinates PBC-apply Error! Please check the input structure.")
                    raise StopIteration("Iter over than 10 times, Something wrong happens!")

        new_struct = copy.deepcopy(for_pbc)
        setattr(new_struct, "coords", Coordinates(frac_coords=new_frac_coords, lattice=for_pbc.lattice))

        return new_struct

    @staticmethod
    def __trans_mass(template, for_trans):
        """Translate the structure along the mass_center direction"""
        tmass = template.mass_center
        nmass = for_trans.mass_center
        new_frac_coords = for_trans.coords.frac_coords + tmass - nmass
        new_struct = copy.deepcopy(for_trans)
        setattr(new_struct, "coords", Coordinates(frac_coords=new_frac_coords, lattice=for_trans.lattice))

        return new_struct

    @staticmethod
    def __tailor_atom_order(template, for_tailor):
        mapping_list = []
        dists = Operator.dist(template, for_tailor)

        # If cart-dist < 0.2, mapping such atom as the same
        for atom_i, value in dists.items():
            for atom_j, dist in value.items():
                if dist < 0.2:
                    mapping_list.append([atom_i.order, atom_j.order])
                    break # the shortest distance

        # For the atoms whose distance over than 0.2, the shortest distance atom will be mapping
        template_remain = set(template.orders).difference(set([i for (i, _) in mapping_list]))
        for_tailor_remain = set(for_tailor.orders).difference(set([i for (_,i) in mapping_list]))
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
                new_kargs[key] = for_tailor.__dict__[key][index]
        if "orders" in new_kargs.keys():
            del new_kargs["orders"]

        return Structure(**new_kargs)


    @staticmethod
    def align(template, for_align):
        """TODO 结构相差大如何位移"""
        if template is None and isinstance(for_align, Structure):
            return for_align

        assert isinstance(template, Structure) and isinstance(for_align, Structure), \
            "The input parameters should be the instance of the <class Structure>"

        assert template.lattice == for_align.lattice, \
            "The lattice vector of input structure is not consistent with the template structure. \n" \
            f"template.lattice \n {template.lattice} \n" \
            f"for_align.lattice \n {for_align.lattice}"

        fmass = template.mass_center
        nmass = for_align.mass_center
        new_struct = copy.deepcopy(for_align)
        count = 0
        while (np.any(fmass != nmass)):
            new_struct = Operator.__pbc_apply(template, new_struct)
            new_struct = Operator.__trans_mass(template, new_struct)
            new_struct = Operator.__tailor_atom_order(template, new_struct)
            fmass = template.mass_center
            nmass = new_struct.mass_center
            count += 1
            if count > 10:
                logger.error("Coordinates align Error! Please check the input structure.")
                raise StopIteration("Iter over than 10 times, Something wrong happens!")

        return new_struct