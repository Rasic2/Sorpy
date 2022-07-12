import random
from pathlib import Path

import numpy as np

from common.logger import root_dir
from common.manager import DirManager
from network.dataset import StructureDataset

xdat_dir = DirManager(dname=Path(f"{root_dir}/train_set/xdat"))
sample = random.sample(range(len(xdat_dir.sub_dir)), 1)
xdat_dir._sub_dir = np.array(xdat_dir.sub_dir, dtype=object)[sample]

energy_file = Path(f"{root_dir}/train_set/energy_summary")

dataset = StructureDataset(xdat_dir=xdat_dir, energy_file=energy_file)
print()
