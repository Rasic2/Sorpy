from operate import Operator as op
from io_file import POSCAR
from logger import current_dir

p1 = POSCAR(fname=f"{current_dir}/input/POSCAR_1-1")
p2 = POSCAR(fname=f"{current_dir}/output/CONTCAR_1-1")
#print(p1 - p2)

#s1 = POSCAR(fname=f"{current_dir}/input/POSCAR_1-1").to_structure(style="Slab+Mol", mol_index=[36,37])
#s2 = POSCAR(fname=f"{current_dir}/output/CONTCAR_1-1").to_structure(style="Slab+Mol", mol_index=[36,37])

print(op.dist(p1, p2))



