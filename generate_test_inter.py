import os
import pickle
import numpy as np
from collections import Counter

from common.logger import root_dir
from common.io_file import POSCAR
from common.base import Coordinates, Element, Atom
from common.structure import Molecule, Structure

factor = [1, 2, 4]
# factor = [1, 1, 1]

def create_mol(s, cut_radius=5.0):
    max_length = cut_radius
    center = None
    # rotate = [[+0.433013, +0.250000, -0.866025],
    #           [-0.500000, +0.866025, +0.000000],
    #           [+0.750000, +0.433013, +0.500000]]
    rotate = None
    # PA = Atom(element=Element("PA"), order=-1, coord=Coordinates(lattice=s.lattice, cart_coords=np.array([10, 10, 10])))
    for index in s.mol_index:
        for atom in s.NNT.index(index):
            if atom[0].element.formula == "Ce" and atom[1] <= max_length:
                center = atom
                max_length = atom[1]

    if center is None:
        raise TypeError("Can't find the Ce element in the cut_radius.")

    elements = [atom[0].element for atom in s.bonds.index(center[0].order) if atom[0].order not in s.mol_index]
    orders = [atom[0].order for atom in s.bonds.index(center[0].order) if atom[0].order not in s.mol_index]
    coords = [atom[0].coord.frac_coords for atom in s.bonds.index(center[0].order) if atom[0].order not in s.mol_index]

    elements.insert(0, center[0].element)
    orders.insert(0, center[0].order)
    coords.insert(0, center[0].coord.frac_coords)
    coords = Coordinates(frac_coords=np.array(coords), lattice=center[0].coord.lattice)

    return Molecule(elements=elements, orders=orders, coords=coords, anchor=center[0].order, rotate=rotate)
    # return Molecule(elements=elements, orders=orders, coords=coords, anchor=PA, rotate=rotate)

def align(template, m):

    assert len(template) == len(m), f"len(template) = {len(template)}, len(m) = {len(m)}, {m}"
    index = [i for i in range(len(m)-1)]
    sorted_index = []
    for _index, (_, _, item_t) in enumerate(template.inter_coords):
        distance=[(i, np.linalg.norm(np.array(m.inter_coords[i][2])-np.array(item_t))) for i in index]
        min_dist = min(distance, key=lambda x: x[1])
        if min_dist[1] < 5.0:
            sorted_index.append((_index, min_dist[0]))
            index.remove(min_dist[0])

    if len(index):
        finish_align = [i for i, _ in sorted_index]
        remain_align = [i for i in range(len(m)-1) if i not in finish_align]

        for _index in remain_align:
            distance=[(i, np.linalg.norm(np.array(m.inter_coords[i][2])-np.array(template.inter_coords[_index][2]))) for i in index]
            min_dist = min(distance, key=lambda x: x[1])
            sorted_index.append((_index, min_dist[0]))
            index.remove(min_dist[0])

        # if len(index) == 1:
        #     sorted_index.append((remain_align[0], index[0]))
        # else:
        #     raise NotImplementedError("This function is not implemented now!")

    sorted_index = sorted(sorted_index, key=lambda x: x[0])
    return np.array([m.vector[index][2] for _, index in sorted_index])

def inter_coord(dname):
    cut_radius = 5.0
    template = POSCAR(fname=f"{root_dir}/examples/CeO2_111/POSCAR_template").to_structure(style="Slab+Mol", mol_index=[36,37])
    template.find_nearest_neighbour_table(cut_radius=cut_radius)
    m_template = create_mol(template)

    mcoords = []
    orders = []
    for file in os.listdir(dname):
        print(f"Handle the {file}")
        s1 = POSCAR(fname=f"{dname}/{file}").to_structure(style="Slab+Mol", mol_index=[36,37], anchor=36)
        s1.find_nearest_neighbour_table(cut_radius=cut_radius)
        mol_CO = s1.molecule
        mol_CO_coord = pickle.loads(pickle.dumps(mol_CO.frac_coords))
        mol_CO_coord[1] = np.where(np.array(mol_CO.inter_coords[0][2])<0, np.array(mol_CO.inter_coords[0][2])+360, np.array(mol_CO.inter_coords[0][2])) / [1, 180, 360] / factor - [1.142, 0, 0]
        # mol_CO_coord[1] = (mol_CO.vector[0][2]/1.142 + 1) / 2
        mol_slab= create_mol(s1)
        mol_slab_coord = pickle.loads(pickle.dumps(mol_slab.frac_coords))
        mol_slab_coord[1:8] = (align(m_template, mol_slab) / 2.356 + 1) / 2
        # mol_slab_coord[:] = np.where(align(m_template, mol_slab)<0, align(m_template, mol_slab)+360, align(m_template, mol_slab)) / [20, 180, 360] #- [2.356, 0, 0]
        mol_coord = np.concatenate((mol_slab_coord, mol_CO_coord), axis=0)
        mcoords.append(mol_coord)
        orders.append((mol_slab.orders+s1.mol_index))
    #print()

    return np.array(mcoords), orders

def reconstruct_coord(dname, test_output, orders, lattice):
    # rotate = [[+0.433013, +0.250000, -0.866025],
    #           [-0.500000, +0.866025, +0.000000],
    #           [+0.750000, +0.433013, +0.500000]]

    test_output = test_output.reshape(50, 10, 3)
    test_output[:, 9, :] = test_output[:, 9, :] * [1, 180, 360] * factor + [1.142, 0, 0]
    # test_output[:, 1:8, :] = test_output[:, 1:8, :] * [20, 180, 360] #+ [2.356, 0, 0]

    # handle Ce-O
    Ce_anchor_cart = np.dot(test_output[:, 0, :], lattice.matrix)
    # r, theta, phi = test_output[:, 0:8, 0], np.deg2rad(test_output[:, 0:8, 1]), np.deg2rad(test_output[:, 0:8, 2])
    # x = r * np.sin(theta) * np.cos(phi)
    # y = r * np.sin(theta) * np.sin(phi)
    # z = r * np.cos(theta)
    # Ce_xyz_cart = np.concatenate((x.reshape(50,8,1), y.reshape(50,8,1), z.reshape(50,8,1)), axis=2)
    Ce_xyz_cart = (test_output[:, 1:8, :] * 2 - 1 ) * 2.356
    # Ce_xyz_cart = np.dot(Ce_xyz_cart, np.linalg.inv(np.array(rotate)))
    test_output[:, 1:8, :] = np.dot((Ce_anchor_cart.reshape((50, 1, 3)) + Ce_xyz_cart), lattice.inverse)
    # test_output[:, 0:8, :] = np.dot((np.array([10, 10, 10]) + Ce_xyz_cart), lattice.inverse)

    #handle C-O
    C_anchor_cart = np.dot(test_output[:, 8, :], lattice.matrix)
    r, theta, phi = test_output[:, 9, 0], np.deg2rad(test_output[:, 9, 1]), np.deg2rad(test_output[:, 9, 2])
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    C_xyz_cart = np.concatenate((x.reshape(50,1,1), y.reshape(50,1,1), z.reshape(50,1,1)), axis=2)
    # C_xyz_cart = (test_output[:, 9, :] * 2 - 1 ) * 1.142
    test_output[:, 9, :] = np.dot((C_anchor_cart.reshape((50, 3)) + C_xyz_cart.reshape((50, 3))), lattice.inverse)

    outputs = []
    for file, order, coord in zip(os.listdir(dname), orders, test_output):
        s1 = POSCAR(fname=f"{dname}/{file}").to_structure(style="Slab+Mol", mol_index=[36, 37], anchor=36)
        output_frac = np.copy(s1.frac_coords)
        output_frac[order] = coord
        outputs.append(output_frac)
    return outputs


if __name__ == "__main__":

    template = POSCAR(fname=f"{root_dir}/examples/CeO2_111/POSCAR_template").to_structure(style="Slab+Mol",
                                                                                          mol_index=[36, 37])
    ori_dir = f"{root_dir}/test_set/guess/ori-2"
    test_dir = f"{root_dir}/test_set/guess/ML-test"
    test_input, orders = inter_coord(ori_dir)
    test_input = test_input.reshape((50, 30))

    from keras.models import load_model
    model = load_model("intercoord_3layer.h5")
    test_output = model.predict(test_input)
    #print(test_input[2].reshape((10, 3)))
    #print()
    #print(test_output[2].reshape((10, 3)))
    # print(Counter(np.where(test_output-test_input > 0.1)[1]))
    # exit()
    test_output= reconstruct_coord(ori_dir, test_output, orders, template.lattice)

    style, elements, lattice, TF = template.style, template.elements, template.lattice, template.TF
    for index, item in enumerate(test_output):
        s = Structure(style=style, elements=elements, coords=Coordinates(frac_coords=item, lattice=lattice),
                      lattice=lattice, TF=TF)
        s.write_to_POSCAR(fname=f'{test_dir}/POSCAR_ML_{index + 1}')
    #print(test_output)
    #print(orders)