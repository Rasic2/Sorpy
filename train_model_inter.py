import os
import h5py
import numpy as np
from pathlib import Path
from collections import Counter

from common.io_file import POSCAR
from common.logger import logger, root_dir
from common.model import Ploter, Model
from common.manager import DirManager

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


if __name__ == "__main__":

    input_dir = Path(root_dir) / "train_set/input"
    output_dir = Path(root_dir) / "train_set/output"
    template_file = Path(root_dir) / "examples/CeO2_111/POSCAR_template"
    data_load_file = Path(root_dir) / "results/data_train-test.h5"
    model_save_file = Path(root_dir) / "results/intercoord_3layer.h5"
    plot_save_file = Path(root_dir) / "results/intercoord_3layer.svg"
    data_load = "c"

    if data_load == "c":

        kargs = {"style": "Slab+Mol",
                 "mol_index": [36, 37],
                 "anchor": 36}

        logger.info("Load the structure information.")
        template = POSCAR(fname=template_file).to_structure(**kargs)
        input_dm = DirManager(dname=input_dir, template=template, **kargs)
        output_dm = DirManager(dname=output_dir, template=template, **kargs)

        logger.info("Calculate the vcoords.")
        data_output, orders_o = output_dm.vcoords()
        logger.info("The vcoords of CONTCARs finished.")
        data_input, orders_i = input_dm.vcoords(orders=orders_o)

        logger.info("Write the data into the .h5 file.")
        with h5py.File(data_load_file, "w") as hf:
            hf.create_dataset("data_input", data=data_input)
            hf.create_dataset("data_output", data=data_output)
    elif data_load == "f":
        logger.info("Load data from the .h5 file.")
        with h5py.File(data_load_file, "r") as hf:
            data_input = hf["data_input"][:]
            data_output = hf["data_output"][:]
    else:
        raise TypeError("Please indicate the load method of model train data, <'c' or 'f'>")

    logger.info("Train the model.")
    from keras import models, layers, Input

    slab_input = Input(shape=(8 * 3,))
    mol_input = Input(shape=(2 * 3,))

    slab_model = layers.Dense(64, activation='relu')(slab_input)
    slab_model = layers.Dense(64, activation='relu')(slab_model)
    slab_model = models.Model(inputs=slab_input, outputs=slab_model)

    mol_model = layers.Dense(128, activation='relu')(mol_input)
    mol_model = layers.Dense(128, activation='relu')(mol_model)
    mol_model = layers.Dense(128, activation='relu')(mol_model)
    mol_model = models.Model(inputs=mol_input, outputs=mol_model)
    concatenated = layers.concatenate([slab_model.output, mol_model.output])

    # structure = layers.Dense(64, activation='relu')(concatenated)
    # structure = layers.Dense(30)(structure)
    structure = layers.Dense(30)(concatenated)
    model = models.Model(inputs=[slab_model.input, mol_model.input], outputs=structure)

    model.compile(loss='mae', optimizer='rmsprop', metrics=['mae'])
    # model.compile(loss='mae', optimizer=optimizers.RMSprop(learning_rate=1e-04), metrics=['mae'])

    train_model = Model(model, data_input, data_output, normalization="vcoord", expand=None)
    train_model.train_output[:, 0] = np.where(train_model.train_output[:, 0] - train_model.train_input[:, 0] > 0.5,
                                              train_model.train_output[:, 0] - 1, train_model.train_output[:, 0])
    train_model.train_output[:, 0] = np.where(train_model.train_output[:, 0] - train_model.train_input[:, 0] < -0.5,
                                              train_model.train_output[:, 0] + 1, train_model.train_output[:, 0])

    train_model.train_output[:, 0] = train_model.train_output[:, 0] - train_model.train_input[:, 0]
    train_model.train_input[:, 0] = 0.0

    # print(train_model.train_output[0])
    # print(np.where(train_model.train_output[:, 0] - train_model.train_input[:, 0]>0.5))
    # print(train_model.train_output[561, 0] - train_model.train_input[561, 0])
    # print(train_model.train_output[56, 8])
    # print()
    # print(train_model.train_input[56, 8])
    # exit()
    # print(np.where(train_model.train_input[:, 0]>0.1))
    print(train_model.train_input[5, 0])
    print(train_model.train_output[5, 0])
    # print(np.where(np.abs(train_model.train_output - train_model.train_input) > 0.1))
    logger.info(Counter(np.where(np.abs(train_model.train_output - train_model.train_input) > 0.1)[1]))
    exit()
    history = train_model("hold out", mname=model_save_file, epochs=60)

    p = Ploter(history)
    p.plot(fname=plot_save_file)
