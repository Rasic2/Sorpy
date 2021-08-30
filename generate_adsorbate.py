#!/usr/bin/env python

import math
import warnings
import itertools

import numpy as np
from pymatgen.core import Structure, Lattice, Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import generate_all_slabs
from pymatgen.core.sites import Site
from pymatgen.io.vasp import Poscar

from _logger import *

CAL_DIR = os.path.join(current_dir, "cal")
surface_111_DIR = os.path.join(CAL_DIR, "111")
surface_110_DIR = os.path.join(CAL_DIR, "110")

warnings.filterwarnings("ignore") # Ignore the warning output

def surface_cleave( miller: tuple, latt_abc: float = 5.450, supercell: tuple = (2, 2, 1)):
    """
    Cleave the specified surface according to the Miller-index

    :param latt_abc:            晶格常数
    :param miller:              密勒指数，指定切哪个表面
    :param supercell:           扩胞参数
    :return:                    表面Slab
    """
    CeO2 = Structure.from_spacegroup("Fm-3m", Lattice.cubic(latt_abc), ["Ce", "O"], [[0, 0, 0], [0.25, 0.25, 0.25]])

    if miller == (1, 1, 1):
        slabs = generate_all_slabs(CeO2, max_index=1, min_slab_size=8.0, min_vacuum_size=15.0, center_slab=True,
                                   max_normal_search=1)
        CeO2_surf = [slab for slab in slabs if slab.miller_index == miller][1]
    else:
        slabs = generate_all_slabs(CeO2, max_index=1, min_slab_size=3.0, min_vacuum_size=15.0, center_slab=True,
                                   max_normal_search=1)
        CeO2_surf = [slab for slab in slabs if slab.miller_index == miller][0]

    CeO2_surf.make_supercell(supercell)
    asf_CeO2_surf = AdsorbateSiteFinder(CeO2_surf)

    return asf_CeO2_surf


def rotate_molecule(molecule, indices, theta: float = 0.0, phi: float = 0.0):
    """
    For the CO molecule, rotate it along the specified direction.

    :param molecule:        Pymatgen molecule instance
    :param indices:         indices of the molecule
    :param theta:           z-axis angle
    :param phi:             xy-surface angle
    :return:                rotated Molecule
    """
    radius = molecule.get_distance(0, 1)
    theta = math.radians(theta)
    phi = math.radians(phi)

    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)
    for ii in indices:
        site_i = molecule._sites[ii]
        if ii != 0:
            new_site = Site(site_i.species, np.array([x, y, z]), properties=site_i.properties)
            molecule._sites[ii] = new_site
    return molecule


def translate_molecule(molecule, vector):
    """
    For the CO molecule, translate to make it have the same surface height.

    :param molecule:            Pymatgen molecule instance
    :param vector:              the direction determine how to translate
    :return:                    translated molecule
    """
    indices = range(len(molecule))
    for ii in indices:
        site_i = molecule._sites[ii]
        new_site = Site(site_i.species, np.array(vector) + site_i.coords, properties=site_i.properties)
        molecule._sites[ii] = new_site
    return molecule


def orthogonal_molecule_getter():
    """
    For the CO molecule, after rotate and translate, six molecules can be acquired.

    :return:            Molecules list
    """
    Molecules_i = []
    for ii in itertools.product([0, 90, 180], [0, 90, 180, 270]):
        theta = ii[0]
        phi = ii[1]
        CO = Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.142]])
        CO_new = rotate_molecule(CO, list(range(len(CO.sites))), theta=theta, phi=phi)

        shiftz = 0
        for jj in range(len(CO_new)):
            site_i = CO_new._sites[jj]
            if site_i.coords[2] < 0 and math.fabs(site_i.coords[2]) > shiftz:
                shiftz = math.fabs(site_i.coords[2])

        CO_newt = translate_molecule(CO_new, [0, 0, shiftz])
        if len(Molecules_i) > 0 and CO_newt not in Molecules_i:
            Molecules_i.append(CO_newt)
        elif len(Molecules_i) == 0:
            Molecules_i.append(CO_newt)

    return Molecules_i

if __name__ == "__main__":

    if not os.path.exists(CAL_DIR):
        os.mkdir(CAL_DIR)

    side_ref = 7.707464 # 111 surface <latt_abc>
    miller = (1, 1, 0)
    z_height = 17.5

    #asf_CeO2_surf = surface_cleave((1, 1, 1))
    asf_CeO2_surf = surface_cleave(miller)
    latt = asf_CeO2_surf.slab.lattice.matrix[:2, :2]
    side_a = np.linspace(0, 0.5, num=math.ceil(10 * asf_CeO2_surf.slab.lattice.a / side_ref))
    side_b = np.linspace(0, 0.5, num=math.ceil(10 * asf_CeO2_surf.slab.lattice.b / side_ref))
    side_arr = np.linspace(0, 0.5, num=10)
    Molecules = orthogonal_molecule_getter()

    logger.info(f"Generate the VASP input file-POSCAR for the {miller} surface.")
    for i, item in enumerate(Molecules):
        count = 0
        for j in itertools.product(side_a, side_b):
            Mat2 = np.dot(j, latt)
            CO_ads = asf_CeO2_surf.add_adsorbate(item, [Mat2[0], Mat2[1], z_height])
            for site in CO_ads.sites:
                site.properties['selective_dynamics'] = [True, True, True]
            count += 1
            p = Poscar(CO_ads)
            p.write_file(f"{surface_110_DIR}/POSCAR_{i+1}-{count}")