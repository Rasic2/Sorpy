import os
import copy
import h5py
from pathlib import Path
from common.io_file import POSCAR
from common.manager import DirManager
from common.model import Model, Ploter

from common.logger import logger, root_dir

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
input_dir = Path(root_dir) / "train_set" / "xinput-m"
output_dir = Path(root_dir) / "train_set" / "xoutput-m"

if __name__ == "__main__":

    kargs = {"style": "Slab+Mol",
             "mol_index": [36, 37],
             "anchor": 36,
             "ignore_mol": True,
             'expand': {'expand_z': {'boundary': 0.2, 'expand_num': 2, 'ignore_index': [37]}}}

    template = POSCAR(fname=Path(root_dir) / "examples/CeO2_111/POSCAR_template").to_structure(**kargs)
    data_load = "f"

    logger.info("Load the structure information.")
    input_dm = DirManager(dname=input_dir, template=template, **kargs)
    output_dm = DirManager(dname=output_dir, template=template, **kargs)

    if data_load == "c":
        logger.info("Calculate the mcoords.")
        data_input, data_output = copy.deepcopy(input_dm.mcoords), copy.deepcopy(output_dm.mcoords)
        with h5py.File(Path(root_dir)/"results/data_train.h5", "w") as hf:
            hf.create_dataset("data_input", data=data_input)
            hf.create_dataset("data_output", data=data_output)
    elif data_load == "f":
        logger.info("Load data from the .h5 file.")
        with h5py.File(Path(root_dir)/"results/data_train.h5", "r") as hf:
            data_input = hf["data_input"][:]
            data_output = hf["data_output"][:]
    else:
        raise TypeError("Please indicate the load method of model train data, <'c' or 'f'>")

    from keras import models, layers
    model = models.Sequential()
    model.add(layers.Dense(1024, activation='relu', input_shape=(38 * 3,)))
    model.add(layers.Dense(114))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

    train_model = Model(model, data_input, data_output, normalization="mcoord", expand=kargs['expand'])
    history = train_model("hold out", mname=Path(root_dir)/"results/CeO2_111_CO_test.h5", epochs=3)

    p = Ploter(history)
    p.plot(fname=Path(root_dir)/"results/CeO2_111_history.svg")
