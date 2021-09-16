import os
import copy
from pathlib import Path
from common.io_file import POSCAR
from common.manager import DirManager
from common.model import Model, Ploter

from common.logger import logger, current_dir

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
input_dir = Path(current_dir) / "input"
output_dir = Path(current_dir) / "output"

if __name__ == "__main__":

    kargs = {"style": "Slab+Mol",
             "mol_index": [36, 37],
             "anchor": 36,
             "ignore_mol": True,
             'expand': {'expand_z': {'boundary': 0.2, 'expand_num': 2, 'ignore_index': [37]}}}

    template = POSCAR(fname=Path(current_dir) / "examples/CeO2_111/POSCAR_template").to_structure(**kargs)

    logger.info("Load the structure information.")
    input_dm = DirManager(dname=input_dir, template=template, **kargs)
    output_dm = DirManager(dname=output_dir, template=template, **kargs)

    logger.info("Calculate the mcoords")
    data_input, data_output = copy.deepcopy(input_dm.mcoords), copy.deepcopy(output_dm.mcoords)

    from keras import models, layers
    model = models.Sequential()
    model.add(layers.Dense(1024, activation='relu', input_shape=(38 * 3,)))
    model.add(layers.Dense(114))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

    train_model = Model(model, data_input, data_output, normalization="mcoord", expand=kargs['expand'])
    history = train_model("hold out", mname="CeO2_111_CO_test.h5")

    p = Ploter(history)
    p.plot(fname="CeO2_111_history.svg")
