import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from common.logger import root_dir, logger
from common.manager import DirManager
from network.dataset import StructureDataset

model = torch.load(f"{root_dir}/model-energy.pth")
loss_fn = nn.MSELoss(reduction='mean')
predict_dir = DirManager(dname=Path(f"{root_dir}/train_set/xdat"))
sample = random.sample(range(len(predict_dir.sub_dir)), 1)
predict_dir._sub_dir = np.array(predict_dir.sub_dir, dtype=object)[sample]

energy_file = Path(f"{root_dir}/train_set/energy_summary")
energy_dict = StructureDataset.load_energy(energy_file)
energy_tensor = torch.from_numpy(np.array(sum([value for value in energy_dict.values()], [])))
energy_mean = energy_tensor.mean()
energy_var = energy_tensor.var()
energy_max = energy_tensor.max()
energy_min = energy_tensor.min()

dataset = StructureDataset(xdat_dir=predict_dir, energy_file=energy_file)
new_energy = (dataset.data[-1] -  energy_min) / (energy_max - energy_min)
# new_energy = new_energy / energy_var.pow(0.5)
dataset.data = (*dataset.data[:-1], new_energy)

batch_size = 1
predict_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

energy_predict_result = []
energy_target_result = []
model.eval()
for step, data in enumerate(predict_dataloader):
    atom_feature, bond_dist3d, adj_matrix, _, energy = data

    if torch.cuda.is_available():
        atom_feature = atom_feature.cuda()
        adj_matrix = adj_matrix.cuda()
        bond_dist3d = bond_dist3d.cuda()
        energy = energy.cuda()

    energy_predict = model(atom_feature, bond_dist3d, adj_matrix)
    energy_predict_result.append(energy_predict.cpu().detach().item())
    energy_target_result.append(energy.cpu().detach().item())

loss = loss_fn(torch.Tensor(energy_predict_result), torch.Tensor(energy_target_result))
logger.info(f"loss =  {loss.item()}")
x = np.linspace(-0.1, 0.5, 100)
y = x
plt.plot(energy_target_result, energy_predict_result, "o")
plt.plot(x, y, '-')
plt.xlabel("Target energy / eV")
plt.ylabel("Predict energy / eV")
plt.xlim([-0.1, 0.5,])
plt.ylim([-0.1, 0.5,])
plt.show()
# plt.savefig("result.png")

print()
