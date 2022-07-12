from collections import defaultdict

from common.logger import root_dir

TRAIN_SET = f"{root_dir}/train_set"
with open(f"{TRAIN_SET}/energy_summary") as f:
    cfg = f.readlines()

energy = defaultdict(list)
for line in cfg:
    if len(line.split()) == 1:
        key = line.split()[0]
    else:
        energy[key].append(line.split()[-1])

energy_sort = sorted(energy.items(), key=lambda x: (int(x[0].split("-")[0]), int(x[0].split("-")[1])))
energy_sort_dict = {item[0]:item[1] for item in energy_sort}

print()
