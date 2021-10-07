import os
import h5py
import numpy as np
from pathlib import Path
from collections import Counter

from common.io_file import POSCAR
from common.logger import logger, root_dir
from common.model import Ploter
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
        shape = data_input.shape
        data_input, data_output = data_input.reshape((shape[0], shape[1] * shape[2])), \
                                  data_output.reshape((shape[0], shape[1] * shape[2]))

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

    logger.info(Counter(np.where(data_output-data_input>0.1)[1]))

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

    structure = layers.Dense(64, activation='relu')(concatenated)
    structure = layers.Dense(30)(structure)
    model = models.Model(inputs=[slab_model.input, mol_model.input], outputs=structure)

    model.compile(loss='mae', optimizer='rmsprop', metrics=['mae'])
    # model.compile(loss='mae', optimizer=optimizers.RMSprop(learning_rate=1e-04), metrics=['mae'])
    history = model.fit([data_input[:, :24], data_input[:, 24:]], data_output, epochs=60, batch_size=2,
                        validation_split=0.1)
    # predict = model.predict(data_input)

    model.save(model_save_file)

    p = Ploter(history)
    p.plot(fname=plot_save_file)
