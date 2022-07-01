from pathlib import Path

from common.io_file import POSCAR, XDATCAR
from common.logger import root_dir

if __name__ == '__main__':

    GENERATE = True
    if GENERATE:
        template = POSCAR(fname=Path(f"{root_dir}/train_set/input/POSCAR_1-1")).to_structure(style="Slab")
        XDATCAR_dir = Path(Path(root_dir) / "train_set" / "XDATCAR")
        x = XDATCAR(XDATCAR_dir / "XDATCAR_1-1", style="Slab")
        for structure in x.structures:
            setattr(structure, 'TF', template.TF)

        xdat_pos = [Path(root_dir) / "train_set" / "xdat" / "input" / f'POSCAR_{index + 1}' for index in
                    range(len(x) - 1)]
        xdat_con = [Path(root_dir) / "train_set" / "xdat" / "output" / f'CONTCAR_{index}' for index in range(1, len(x))]

        x.split_file(index=list(range(len(x) - 1)), fname=xdat_pos)
        x.split_file(index=[-1] * (len(x) - 1), fname=xdat_con)
