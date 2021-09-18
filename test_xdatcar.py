import os
from pathlib import Path

from common.base import Elements
from common.logger import ROOTDIR
from common.io_file import XDATCAR


XDATCAR_dir = Path(Path(ROOTDIR) / "train_set" / "XDATCAR")
x1 = XDATCAR(XDATCAR_dir / "XDATCAR_1-1", style="Slab+Mol")
print(x1)
