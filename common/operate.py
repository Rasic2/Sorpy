import numpy as np
from utils import Format_defaultdist

from logger import logger
from common.base import Coordinates
from common.structure import AtomSetBase


class Operator:
    def __init__(self):
        pass

    @staticmethod
    def dist(si, sj) -> Format_defaultdist:
        assert isinstance(si, AtomSetBase) and isinstance(sj, AtomSetBase), \
            "The object A and B are not instance of the <class 'AtomSetBase'>"
        dists = Format_defaultdist(list)
        for ai in si.atoms:
            for aj in sj.atoms:
                dist = np.linalg.norm(aj.cart_coord - ai.cart_coord)
                dists[ai].append((aj, dist))

        sorted_dists = Format_defaultdist(list)
        for key, value in dists.items():
            sorted_dists[key] = sorted(value, key=lambda x: x[1])

        return sorted_dists

    @staticmethod
    def align(template, for_align):
        """TODO 结构相差大如何位移"""
        if template is None and isinstance(for_align, Coordinates):
            return for_align

        assert isinstance(template, Coordinates) and isinstance(for_align, Coordinates), \
            "The input parameters should be the instance of the <class Coordinates>"

        assert template.lattice == for_align.lattice, \
            "The lattice vector of input structure is not consistent with the template structure. \n" \
            f"template.lattice \n {template.lattice} \n" \
            f"for_align.lattice \n {for_align.lattice}"

        index = np.where(np.abs(for_align.frac_coords - template.frac_coords) > 0.5)
        new_frac_coords = np.copy(for_align.frac_coords)

        for (i, j) in zip(*index):
            iter = 0
            while True:
                iter += 1
                diff = new_frac_coords[i, j] - template.frac_coords[i, j]
                increment = 1 if diff < 0 else - 1
                new_frac_coords[i, j] += increment
                diff = new_frac_coords[i, j] - template.frac_coords[i, j]
                if abs(diff) < 0.5:
                    break
                if iter >= 10:
                    logger.error("Coordinates align Error! Please check the input structure.")
                    raise StopIteration("Iter over than 10 times, Something wrong happens!")

        return Coordinates(frac_coords=new_frac_coords, lattice=for_align.lattice)