import itertools
import math
import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

from common.io_file import POSCAR
from common.logger import root_dir, logger
from common.manager import DirManager
from network.dataset import StructureDataset
from network.model_energy import Model


def data_prepare(batch_data):
    atom_feature, bond_dist3d_input, adj_matrix, adj_matrix_tuple, energy = batch_data

    if torch.cuda.is_available():
        atom_feature = atom_feature.cuda()
        bond_dist3d_input = bond_dist3d_input.cuda()
        adj_matrix = adj_matrix.cuda()
        adj_matrix_tuple = adj_matrix_tuple.cuda()
        energy = energy.cuda()

    return atom_feature, bond_dist3d_input, adj_matrix, adj_matrix_tuple, energy


xdat_dir = DirManager(dname=Path(f"{root_dir}/train_set/xdat"))
sample = random.sample(range(len(xdat_dir.sub_dir)), 10)
xdat_dir._sub_dir = np.array(xdat_dir.sub_dir, dtype=object)[sample]

energy_file = Path(f"{root_dir}/train_set/energy_summary")

data = torch.load("../dataset-energy.pth")
# data = None
dataset = StructureDataset(xdat_dir=xdat_dir, energy_file=energy_file, data=data)
# torch.save(dataset.data, "../dataset-energy.pth")

TRAIN = math.floor(len(dataset) * 0.8)
train_dataset = dataset[:TRAIN]
test_dataset = dataset[TRAIN:]
batch_size = 1
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
train_size = len(train_dataloader)
test_size = len(test_dataloader)

# atom_type
structure = POSCAR(fname=Path(f"{root_dir}/train_set/input/POSCAR_1-1")).to_structure()
structure.find_neighbour_table(neighbour_num=12)
atom_type = structure.atoms.atom_type
atom_type_index = [(index, item) for index, item in enumerate(atom_type)]
atom_type_sort_index = sorted(atom_type_index, key=lambda x: x[1])
atom_type_group = [list(item) for key, item in itertools.groupby(atom_type_sort_index, key=lambda x: x[1])]
atom_type_group_name = [group[0][1] for group in atom_type_group]
atom_type_group_index = [[item[0] for item in group] for group in atom_type_group]

model = Model(atom_type=(atom_type_group_name, atom_type_group_index),
              atom_in_fea_num=64,
              atom_out_fea_num=25,
              bond_in_fea_num=3,
              bond_out_fea_num=3,
              bias=True)
# parameters = [(name, param) for name, param in model.named_parameters()]
loss_fn = nn.L1Loss(reduction='mean')
initial_lr = 5e-04
optimizer = optim.SGD(model.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)
if torch.cuda.is_available():
    model = model.cuda()
    loss_fn = loss_fn.cuda()

train_loss_result = []
test_loss_result = []
test_min_loss = 100
threshold = 1000

for epoch in range(50):
    model.train()
    total_train_loss = 0.
    total_test_loss = 0.
    for step, data in enumerate(train_dataloader):
        atom_feature, bond_dist3d_input, adj_matrix, adj_matrix_tuple, energy = data_prepare(data)
        energy_predict = model(atom_feature, bond_dist3d_input, adj_matrix)
        train_loss = loss_fn(energy_predict, energy)
        total_train_loss += train_loss.detach()  # detach the loss to monitor its value
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    # scheduler.step()

    model.eval()
    for step, data in enumerate(test_dataloader):
        atom_feature, bond_dist3d_input, adj_matrix, adj_matrix_tuple, energy = data_prepare(data)
        energy_predict = model(atom_feature, bond_dist3d_input, adj_matrix)
        test_loss = loss_fn(energy_predict, energy)
        total_test_loss += test_loss.detach()
        # print(energy_predict.item(), energy.item())

    if torch.cuda.is_available():
        train_loss_result.append((total_train_loss / train_size).cpu().numpy())
        test_loss_result.append((total_test_loss / test_size).cpu().numpy())
    else:
        train_loss_result.append(total_train_loss / train_size)
        test_loss_result.append(total_test_loss / test_size)

    logger.info(
        f"Training {(epoch + 1)}: train_loss = {total_train_loss / train_size:15.12f}, "
        f"test_loss = {total_test_loss / test_size:15.12f}, "
        f"learning_rate = {optimizer.state_dict()['param_groups'][0]['lr']:15.12f}")

    if (total_test_loss / test_size) <= test_min_loss:
        test_min_loss = total_test_loss / test_size
    elif (total_test_loss / test_size) > test_min_loss * threshold:
        logger.info(f"test_loss exceed {test_min_loss:9.6f} * {threshold}")
        break

# plot the loss-curve
plt.plot(train_loss_result, '-o')
plt.plot(test_loss_result, '-o')
plt.savefig(f"{root_dir}/loss.svg")

# model save
torch.save(model, "../model-energy.pth")
