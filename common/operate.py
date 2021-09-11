import numpy as np
from utils import Format_defaultdist

from structure import AtomSetBase

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