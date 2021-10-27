import copy
import itertools
import numpy as np
from functools import partial
from collections import defaultdict

from common.logger import logger
from common.base import Coordinates
from common.structure import Structure, AtomSetBase, Molecule


class Operator:
    def __init__(self):
        pass

    @staticmethod
    def pbc(array):
        """
        Handle the PBC problem

        :param array: <np.array type>
        :return:
        """
        array = np.where(array > 0.5, array - 1, array)
        array = np.where(array < -0.5, array + 1, array)
        return array

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
    def __pbc_apply(template, s_pbc):
        """Apply the Periodic Boundary Condition <PBC_apply>"""
        index = np.where(np.abs(s_pbc.coords.frac_coords - template.coords.frac_coords) > 0.5)
        temp_index = [(i, j) for i, j in zip(index[0], index[1])]
        final_index = [item for item in temp_index if item[0] not in s_pbc.mol_index]

        np.array_int64 = partial(np.array, dtype=np.int64)
        index = (map(np.array_int64, zip(*final_index)))  # Use zip func unpack the <tuple list>

        new_frac_coords = np.copy(s_pbc.coords.frac_coords)
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

        new_struct = copy.deepcopy(s_pbc)
        setattr(new_struct, "coords", Coordinates(frac_coords=new_frac_coords, lattice=s_pbc.lattice))

        return new_struct

    @staticmethod
    def __trans_mass_center(template, s_trans):
        """Translate the structure along the mass_center direction"""
        logger.debug("Trans the structure along the mass center direction")
        tmass = template.mass_center
        nmass = s_trans.mass_center
        new_frac_coords = s_trans.coords.frac_coords + tmass - nmass
        new_struct = copy.deepcopy(s_trans)
        setattr(new_struct, "coords", Coordinates(frac_coords=new_frac_coords, lattice=s_trans.lattice))

        return new_struct

    @staticmethod
    def __tailor_atom_order(template, s_tailor):
        """Tailor the atom order to achieve the atom mapping"""
        mapping_list = []
        dists = Operator.dist(template, s_tailor)
        logger.debug("Tailor the atom order to achieve the atom mapping")
        # If cart-dist < 0.2, mapping such atom as the same
        for atom_i, value in dists.items():
            for atom_j, dist in value.items():
                # Fix bug, test atom_i and atom_j is the same element
                if dist < 0.2 and atom_i.element == atom_j.element:
                    mapping_list.append([atom_i.order, atom_j.order])
                    break  # the shortest distance

        # For the atoms whose distance over than 0.2, the shortest distance atom will be mapping
        template_remain = set(template.orders).difference(set([i for (i, _) in mapping_list]))
        for_tailor_remain = set(s_tailor.orders).difference(set([i for (_, i) in mapping_list]))
        for index_i in template_remain:
            min_dist, min_index = 100, None
            for index_j in for_tailor_remain:
                if template.atoms[index_i].element == s_tailor.atoms[index_j].element:
                    dist = dists[template.atoms[index_i]][s_tailor.atoms[index_j]]
                    if dist < min_dist:
                        min_dist, min_index = dist, index_j
            mapping_list.append([index_i, min_index])
        index = [i for _, i in mapping_list]
        new_kargs = copy.deepcopy(s_tailor.__dict__)
        for key in ('elements', 'coords', 'TF'):
            if key in new_kargs.keys():
                try:
                    new_kargs[key] = s_tailor.__dict__[key][index]
                except:
                    print(f"key = {key}, index = {index}")
                    raise
        if "orders" in new_kargs.keys():
            del new_kargs["orders"]

        return Structure(**new_kargs)

    @staticmethod
    def align_structure(template, s_align):
        """
        Align the structure to the template.

        :func: __pbc_apply
        :func: __tailor_atom_order
        :func: __trans_mass_center

        :param: template
        :param: s_lign
        :return: aligned structure
        """
        if template is None and isinstance(s_align, Structure):
            return s_align

        assert isinstance(template, Structure) and isinstance(s_align, Structure), \
            f"The input parameters should be the instance of the <class Structure>, \n" \
            f"but your input parameters are {type(template)} and {type(s_align)}"

        assert template.lattice == s_align.lattice, \
            "The lattice vector of input structure is not consistent with the template structure. \n" \
            f"template.lattice \n {template.lattice} \n" \
            f"for_align.lattice \n {s_align.lattice}"

        fmass = template.mass_center
        nmass = s_align.mass_center
        new_struct = copy.deepcopy(s_align)
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
                logger.error("Coordinates align_structure Error! Please check the input structure.")
                raise StopIteration("Iter over than 10 times, Something wrong happens!")
        return new_struct

    @staticmethod
    def align_molecule(template, m_align):  # TODO: the Ce is not consider in this func, <Ce1O7 system>
        """
        Align the Molecule to the template.

        :param: template
        :param: m_align
        :return: aligned Molecule
        """
        if template is None and isinstance(m_align, Molecule):
            return m_align

        assert isinstance(template, Molecule) and isinstance(m_align, Molecule), \
            f"The input parameters should be the instance of the <class Molecule>, \n" \
            f"but your input parameters are {type(template)} and {type(m_align)}"

        assert len(template) == len(m_align), \
            f"The atoms of the input template and molecule to be aligned is not same, \n" \
            f"len(template) = {len(template)}, len(m) = {len(m_align)} \n" \
            f"{m_align}"

        index = [item for item in m_align.inter_coords]  # m.index which remain to be sorted
        sorted_index = []
        for item_t in template.inter_coords:
            distance = [(item_m, np.linalg.norm(np.array(item_m[2]) - np.array(item_t[2]))) for item_m in index]
            min_dist = min(distance, key=lambda x: x[1])
            if min_dist[1] < 5.0:
                sorted_index.append((item_t[1].order, min_dist[0][1].order))  # template.index <--> m_align.index
                index.remove(min_dist[0])  # removing the sorted m_align.index

        if len(index):
            finish_align = [i for i, _ in sorted_index]  # template.index which have been sorted
            remain_align = [item for item in template.inter_coords if item[1].order not in finish_align]
            # template.index which remain to be sorted

            for item_t in remain_align:
                distance = [(item_m, np.linalg.norm(np.array(item_m[2]) - np.array(item_t[2]))) for item_m in index]
                min_dist = min(distance, key=lambda x: x[1])
                sorted_index.append((item_t[1].order, min_dist[0][1].order))  # template.index <--> m_align.index
                index.remove(min_dist[0])  # removing the sorted m_align.index

        sorted_index = sorted(sorted_index, key=lambda x: x[0])  # Make the order following the template.index

        # construct the new Molecule
        __orders = [m_align.orders.index(index) for _, index in sorted_index]
        __orders.insert(0, 0)
        elements = np.array(m_align.elements)[__orders]
        orders = [int(order) for order in np.array(m_align.orders)[__orders]]
        coords = Coordinates(frac_coords=np.array(m_align.coords.frac_coords)[__orders], lattice=m_align.coords.lattice)
        return Molecule(elements=elements, orders=orders, coords=coords, anchor=orders[0])

    @staticmethod
    def normalize_mcoord(data):
        """
        <mcoord normalization func>

        :param data:    shape = [:, 38, 3]
        :return: normalized data
        """
        data[:, 37, 2] = np.where(data[:, 37, 2] >= 0, data[:, 37, 2], 360 + data[:, 37, 2])
        data[:, 37, :] = data[:, 37, :] / [1, 180, 360] - [1.142, 0, 0]
        return data

    @staticmethod
    def normalize_vcoord(data):
        """
        <vcoord normalization func>

        :param data:    shape = [:, 10, 3] <Ce1O7 + CO>
        :return: normalized data
        """
        temp_data = np.copy(data)
        temp_data[:, 1:-2, :] = (temp_data[:, 1:-2, :] / 2.356 + 1) / 2
        temp_data[:, -1, :] = np.where(temp_data[:, -1, :] < 0, temp_data[:, -1, :] + 360, temp_data[:, -1, :]) / [1, 180, 360] - [1.142, 0, 0]
        return temp_data

    @staticmethod
    def find_trans_vector(coord: np.ndarray, anchor=36):
        repeat = 2  # Make repeat to be a parameter in future
        ori_coord = copy.deepcopy(coord)

        C_anchor = ori_coord[:, anchor, :2]
        model_region = np.array([1 / repeat, 1 / repeat])
        search_vector = np.arange(-1, 1 + 1 / repeat, 1 / repeat)
        trans_vectors = []
        for index, item in enumerate(C_anchor):
            search_matrix = itertools.product(search_vector, search_vector)
            for ii in search_matrix:
                if 0 <= (item + ii)[0] <= model_region[0] and 0 <= (item + ii)[1] <= model_region[1]:
                    vector_m = [ii[0], ii[1], 0]
                    ori_coord[index, anchor, :] += vector_m
                    trans_vectors.append(vector_m)
                    break

        return ori_coord, trans_vectors

    @staticmethod
    def getRotate(mode):
        """
        # 111-vector
            [ 1.92686215e+00,   1.11247889e+00,  -7.86646043e-01]           20
            [-1.92686985e+00,   1.11247889e+00,  -7.86646043e-01]           22
            [-3.85373200e-06,  -2.22495111e+00,  -7.86646043e-01]           23
            [ 0.00000000e+00,   0.00000000e+00,  -2.35990981e+00]           24
            [ 3.85373200e-06,   2.22495111e+00,   7.86646043e-01]           32  remove
            [ 1.92686985e+00,  -1.11247889e+00,   7.86646043e-01]           35
            [-1.92686215e+00,  -1.11247889e+00,   7.86646043e-01]           33

        # 110-vector                                              <111> - <110>
            [ 0.      ,  -1.3625,  -1.92685829]                       23-21
            [ 1.926866,  -1.3625,   0.        ]                       33-25
            [-1.926866,  -1.3625,   0.        ]                       35-27
            [ 1.926866,   1.3625,   0.        ]                       20-32
            [-1.926866,   1.3625,   0.        ]                       22-34
            [ 0.      ,   1.3625,  -1.92685829]]                      24-44
        """
        import math

        r = 2.35993
        theta = math.radians(109.4714)
        phi = math.radians(30)

        Base_111 = [[ 0.                             ,                               0., -r                ],
                    [ r*math.sin(theta)*math.cos(phi), -r*math.sin(theta)*math.sin(phi), -r*math.cos(theta)],
                    [-r*math.sin(theta)*math.cos(phi), -r*math.sin(theta)*math.sin(phi), -r*math.cos(theta)]]

       #[[ 0.00000000e+00,   0.00000000e+00,  -2.35990981e+00],
       # [ 1.92686985e+00,  -1.11247889e+00,   7.86646043e-01],
       # [-1.92686215e+00,  -1.11247889e+00,   7.86646043e-01]]  # three points in 111-surface

        theta = math.radians(144.7355)
        phi = math.radians(90)

        Base_110 = [[ 0.               ,  r*math.sin(theta)*math.sin(phi), r*math.cos(theta)],
                    [ r*math.cos(theta), -r*math.sin(theta)*math.sin(phi), 0.               ],
                    [-r*math.cos(theta), -r*math.sin(theta)*math.sin(phi), 0.               ]]

        #[[ 0.,         1.3625,  -1.92685829],
        # [-1.926866,  -1.3625,   0.        ],
        # [ 1.926866,  -1.3625,   0.        ]]  # three points in 110-surface

        if mode == '110':
            rotate = np.dot(np.linalg.inv(np.array(Base_110)), np.array(Base_111))
        else:
            rotate = None

        return rotate
