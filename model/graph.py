from pathlib import Path

from common.io_file import POSCAR
from common.logger import root_dir

fname = Path(f'{root_dir}/train_set/input/POSCAR_1-1')
s1 = POSCAR(fname=fname).to_structure(style="Slab")
s1.find_nearest_neighbour_table(amplitude=0.2)
# atoms = s1.atoms
var = s1.NNT.data
print()