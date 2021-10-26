import os
from pathlib import Path

from common.logger import root_dir, logger
from common.io_file import POSCAR, XDATCAR

if __name__ == "__main__":

    kargs = {"style": "Slab+Mol",
             "mol_index": [36, 37],
             "anchor": 36,
             "ignore_mol": True,
             'expand': {'expand_z': {'boundary': 0.2, 'expand_num': 2, 'ignore_index': [37]}}}
    template = POSCAR(fname=Path(root_dir) / "examples/CeO2_111/POSCAR_template").to_structure(**kargs)

    XDATCAR_dir = Path(Path(root_dir) / "train_set" / "XDATCAR")
    Xinput_mo_dir = Path(Path(root_dir) / "train_set" / "xinput-o")
    Xoutput_mo_dir = Path(Path(root_dir) / "train_set" / "xoutput-o")
    Path(Xinput_mo_dir).mkdir(exist_ok=True)
    Path(Xoutput_mo_dir).mkdir(exist_ok=True)

    count = 0
    for file in os.listdir(XDATCAR_dir):
        count += 1
        logger.info(f"Start handle the {file}, {len(os.listdir(XDATCAR_dir))-count} XDATCAR files remaining")
        x = XDATCAR(XDATCAR_dir / file, style="Slab+Mol", TF=template.TF)
        x.to_POSCAR(dname=Xinput_mo_dir, indexes=range(len(x)-1))
        x.to_CONTCAR(dname=Xoutput_mo_dir, indexes=[i+1 for i in range(len(x)-1)])