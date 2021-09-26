import os
import copy

from common.operate import Operator as op
from common.io_file import POSCAR, CONTCAR
from common.manager import FileManager, DirManager
from common.model import Model
from common.structure import Molecule
from common.base import Coordinates
from common.utils import Format_list

from common.logger import root_dir
from pathlib import Path
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def create_mol(s, cut_radius=5.0):
    max_length = cut_radius
    center = None
    for index in s.mol_index:
        for atom in s.NNT.index(index):
            if atom[0].element.formula == "Ce" and atom[1] <= max_length:
                center = atom
                max_length = atom[1]

    if center is None:
        raise TypeError("Can't find the Ce element in the cut_radius.")

    elements = [atom[0].element for atom in s.bonds.index(center[0].order)]
    orders = [atom[0].order for atom in s.bonds.index(center[0].order)]
    coords = [atom[0].coord.frac_coords for atom in s.bonds.index(center[0].order)]

    elements.insert(0, center[0].element)
    orders.insert(0, center[0].order)
    coords.insert(0, center[0].coord.frac_coords)
    coords = Coordinates(frac_coords=np.array(coords), lattice=center[0].coord.lattice)

    return Molecule(elements=elements, orders=orders, coords=coords, anchor=center[0].order)

def align(template, m):

    assert len(template) == len(m)
    index = [i for i in range(len(m)-1)]
    sorted_index = []
    for _index, (_, _, item_t) in enumerate(template.inter_coords):
        distance=[(i, np.linalg.norm(np.array(m.inter_coords[i][2])-np.array(item_t))) for i in index]
        min_dist = min(distance, key=lambda x: x[1])
        if min_dist[1] < 5.0:
            sorted_index.append((_index, min_dist[0]))
            index.remove(min_dist[0])

    if len(index):
        finish_align = [i for _, i in sorted_index]
        remain_align = [i for i in range(len(m)-1) if i not in finish_align]
        if len(index) == 1:
            sorted_index.append((index[0], remain_align[0]))
        else:
            raise NotImplementedError("This function is not implemented now!")

    sorted_index = sorted(sorted_index, key=lambda x: x[0])
    return np.array([m.inter_coords[index][2] for _, index in sorted_index])

def test_NNT(dname):
    cut_radius = 5.0
    template = POSCAR(fname=f"{root_dir}/examples/CeO2_111/POSCAR_template").to_structure(style="Slab+Mol", mol_index=[36,37])
    template.find_nearest_neighbour_table(cut_radius=cut_radius)
    m_template = create_mol(template)

    mcoords = []
    for file in os.listdir(dname):
        print(file)
        s1 = POSCAR(fname=f"{dname}/{file}").to_structure(style="Slab+Mol", mol_index=[36,37], anchor=36)
        s1.find_nearest_neighbour_table(cut_radius=cut_radius)
        mol_CO = s1.molecule
        mol_CO_coord = copy.deepcopy(mol_CO.frac_coords)
        mol_CO_coord[1] = np.where(np.array(mol_CO.inter_coords[0][2])<0, np.array(mol_CO.inter_coords[0][2])+360, np.array(mol_CO.inter_coords[0][2])) / [1, 180, 360] - [1.142, 0, 0]
        mol_slab= create_mol(s1)
        mol_slab_coord = copy.deepcopy(mol_slab.frac_coords)
        mol_slab_coord[1:] = np.where(align(m_template, mol_slab)<0, align(m_template, mol_slab)+360, align(m_template, mol_slab)) / [1, 180, 360] - [2.356, 0, 0]
        mol_coord = np.concatenate((mol_slab_coord, mol_CO_coord), axis=0)
        mcoords.append(mol_coord)
    print()

    return np.array(mcoords)


if __name__ == "__main__":
    input_dir = f"{root_dir}/train_set/input"
    output_dir = f"{root_dir}/train_set/output"
    data_input = test_NNT(input_dir).reshape((600,30))
    data_output = test_NNT(output_dir).reshape((600, 30))

    from keras import models, layers, optimizers
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(10 * 3,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(30))
    model.compile(loss='mae', optimizer=optimizers.RMSprop(learning_rate=1E-05), metrics=['mae'])

    model.fit(data_input, data_output, epochs=30, batch_size=2, validation_split=0.1)