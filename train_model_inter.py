import os
import pickle
import h5py
import numpy as np
from pathlib import Path
from collections import Counter
from multiprocessing import Pool as ProcessPool

from common.io_file import POSCAR
from common.structure import Molecule
from common.base import Coordinates
from common.logger import logger, root_dir
from common.model import Ploter


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def create_mol(s, orders=None, cut_radius=5.0):
    max_length = cut_radius
    if orders is None:
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
    else:
        elements = np.array(s.elements)[orders]
        coords = Coordinates(frac_coords=np.array(s.coords.frac_coords)[orders], lattice=s.lattice)
        return Molecule(elements=elements, orders=orders, coords=coords, anchor=orders[0])

def align(template, m):
    assert len(template) == len(m), f"len(template) = {len(template)}, len(m) = {len(m)}, {m}"
    index = [i for i in range(len(m)-1)] # for align
    sorted_index = []
    for _index, (_, _, item_t) in enumerate(template.inter_coords):
        distance=[(i, np.linalg.norm(np.array(m.inter_coords[i][2])-np.array(item_t))) for i in index]
        min_dist = min(distance, key=lambda x: x[1])
        if min_dist[1] < 5.0:
            sorted_index.append((_index, min_dist[0]))
            index.remove(min_dist[0])

    if len(index):
        finish_align = [i for i, _ in sorted_index]
        remain_align = [i for i in range(len(m)-1) if i not in finish_align] # template

        for _index in remain_align:
            distance=[(i, np.linalg.norm(np.array(m.inter_coords[i][2])-np.array(template.inter_coords[_index][2]))) for i in index]
            min_dist = min(distance, key=lambda x: x[1])
            sorted_index.append((_index, min_dist[0]))
            index.remove(min_dist[0])

    sorted_index = sorted(sorted_index, key=lambda x: x[0])
    return np.array([m.inter_coords[index][2] for _, index in sorted_index])

def worker_inter_coord(dname, file, cut_radius, m_template, orders):
    s1 = POSCAR(fname=f"{dname}/{file}").to_structure(style="Slab+Mol", mol_index=[36, 37], anchor=36)
    s1.find_nearest_neighbour_table(cut_radius=cut_radius)
    mol_CO = s1.molecule
    mol_CO_coord = pickle.loads(pickle.dumps(mol_CO.frac_coords))
    mol_CO_coord[1] = np.where(np.array(mol_CO.inter_coords[0][2]) < 0, np.array(mol_CO.inter_coords[0][2]) + 360,
                               np.array(mol_CO.inter_coords[0][2])) / [1, 180, 360] - [1.142, 0, 0]
    mol_slab = create_mol(s1, orders=orders)
    mol_slab_coord = pickle.loads(pickle.dumps(mol_slab.frac_coords))
    mol_slab_coord[1:] = np.where(align(m_template, mol_slab) < 0, align(m_template, mol_slab) + 360,
                                  align(m_template, mol_slab)) / [1, 180, 360] - [2.356, 0, 0]
    mol_coord = np.concatenate((mol_slab_coord, mol_CO_coord), axis=0)
    return mol_coord, mol_slab.orders

def inter_coord(dname, orders=None):
    cut_radius = 5.0
    template = POSCAR(fname=f"{root_dir}/examples/CeO2_111/POSCAR_template").to_structure(style="Slab+Mol", mol_index=[36,37])
    template.find_nearest_neighbour_table(cut_radius=cut_radius)
    m_template = create_mol(template)

    pool = ProcessPool(processes=os.cpu_count())
    if orders is not None:
        results = [pool.apply_async(worker_inter_coord, args=(dname, file, cut_radius, m_template, order)) for file, order in zip(os.listdir(dname), orders)]
    else:
        results = [pool.apply_async(worker_inter_coord, args=(dname, file, cut_radius, m_template, orders)) for file in os.listdir(dname)]
    temp_results = [result.get() for result in results]
    mcoords = [item for item, _ in temp_results]
    orders = [item for _, item in temp_results]

    pool.close()
    pool.join()

    return np.array(mcoords), orders

def main():
    input_dir = f"{root_dir}/train_set/input"
    output_dir = f"{root_dir}/train_set/output"
    data_load_file = "data_train-test.h5"
    model_save_file = "intercoord_3layer.h5"
    plot_save_file = "intercoord_3layer.svg"
    data_load = "f"

    if data_load == "c":
        logger.info("Calculate the mcoords.")
        data_output, orders_o = inter_coord(output_dir)# TODO important bug: output 与 input Ce_anchor 不对应
        data_input, orders_i = inter_coord(input_dir, orders_o)
        shape = data_input.shape
        data_input, data_output = data_input.reshape((shape[0], shape[1]*shape[2])), data_output.reshape((shape[0], shape[1]*shape[2]))
        with h5py.File(Path(root_dir)/f"results/{data_load_file}", "w") as hf:
            hf.create_dataset("data_input", data=data_input)
            hf.create_dataset("data_output", data=data_output)
    elif data_load == "f":
        logger.info("Load data from the .h5 file.")
        with h5py.File(Path(root_dir)/f"results/{data_load_file}", "r") as hf:
            data_input = hf["data_input"][:]
            data_output = hf["data_output"][:]
    else:
        raise TypeError("Please indicate the load method of model train data, <'c' or 'f'>")


    data_output[:, 0:3] = np.where((data_output[:, 0:3] - data_input[:, 0:3] > 0.5), data_output[:, 0:3] - 1, data_output[:, 0:3])
    data_output[:, 0:3] = np.where((data_output[:, 0:3] - data_input[:, 0:3] < -0.5), data_output[:, 0:3] + 1, data_output[:, 0:3])

    data_output[:, 24:27] = np.where((data_output[:, 24:27] - data_input[:, 24:27] > 0.5), data_output[:, 24:27] - 1, data_output[:, 24:27])
    data_output[:, 24:27] = np.where((data_output[:, 24:27] - data_input[:, 24:27] < -0.5), data_output[:, 24:27] + 1, data_output[:, 24:27])

    #print(data_input[8].reshape((10, 3)))
    #print()
    #print(data_output[8].reshape((10, 3)))

    #data_input[:, 5] = 0
    #data_output[:, 5] = 0
    print(Counter(np.where(data_output - data_input > 0.5)[1]))
    exit()

    logger.info("Train the model.")
    from keras import models, layers
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10 * 3,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(30))
    model.compile(loss='mae', optimizer='rmsprop', metrics=['mae'])

    history = model.fit(data_input, data_output, epochs=50, batch_size=2, validation_split=0.1)
    predict = model.predict(data_input)

    model.save(model_save_file)

    p = Ploter(history)
    p.plot(fname=plot_save_file)

if __name__ == "__main__":
    main()

#profile.run('main()', 'result')
#p = pstats.Stats('result')
#p.strip_dirs().sort_stats('time').print_stats()