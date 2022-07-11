import itertools
import math
import random
from pathlib import Path

import torch.cuda
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import Linear, Sequential, Tanh, Conv2d
from torch.utils.data import DataLoader

from common.io_file import POSCAR
from common.logger import root_dir, logger
from common.manager import DirManager
from common.structure import Structure
from network.dataset import StructureDataset


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.sequential1 = Sequential(
            Conv2d(in_channels=3, out_channels=16, kernel_size=(1, 3), padding=(0, 1)),
            Conv2d(in_channels=16, out_channels=3, kernel_size=(1, 3), padding=(0, 1)),
        )

        self.sequential2 = Sequential(
            Linear(in_features=3, out_features=16),
            Tanh(),
            # Linear(in_features=16, out_features=64),
            # Tanh(),
            # Linear(in_features=64, out_features=128),
            # Tanh(),
            # Linear(in_features=128, out_features=64),
            # Tanh(),
            # Linear(in_features=64, out_features=16),
            # Tanh(),
            Linear(in_features=16, out_features=3),
        )

    def forward(self, bond):
        bond = torch.permute(input=bond, dims=(0, 3, 1, 2))
        bond_update = self.sequential1(bond)
        bond_update = torch.permute(input=bond_update, dims=(0, 2, 3, 1))
        bond_update = self.sequential2(bond_update)

        return bond_update


def data_prepare(structure, batch_data):
    atom_feature, bond_dist3d_input, adj_matrix, adj_matrix_tuple, bond_dist3d_output = batch_data

    # apply the PBC
    diff_frac = torch.matmul((bond_dist3d_output - bond_dist3d_input), torch.Tensor(structure.lattice.inverse))
    diff_frac = torch.where(diff_frac >= 0.99, diff_frac - 1, diff_frac)
    diff_frac = torch.where(diff_frac <= -0.99, diff_frac + 1, diff_frac)
    diff_cart = torch.matmul(diff_frac, torch.Tensor(structure.lattice.matrix))
    bond_dist3d_output = bond_dist3d_input + diff_cart

    if torch.cuda.is_available():
        atom_feature = atom_feature.cuda()
        bond_dist3d_input = bond_dist3d_input.cuda()
        bond_dist3d_output = bond_dist3d_output.cuda()
        adj_matrix = adj_matrix.cuda()
        adj_matrix_tuple = adj_matrix_tuple.cuda()

    return atom_feature, bond_dist3d_input, bond_dist3d_output, adj_matrix, adj_matrix_tuple


def main():
    # logger.setLevel(logging.DEBUG)
    torch.set_printoptions(profile='full')
    logger.info("---------------Start----------------")

    input_dir = DirManager(dname=Path(f'{root_dir}/train_set/input'))
    output_dir = DirManager(dname=Path(f'{root_dir}/train_set/output'))
    dataset_path = f"{root_dir}/dataset-AtomType.pth"

    if not Path(dataset_path).exists():
        dataset = StructureDataset(input_dir, output_dir)
        torch.save(dataset.data, dataset_path)
        logger.info("-----All Files loaded successful-----")
    else:
        data = torch.load(dataset_path)
        dataset = StructureDataset(input_dir, output_dir, data=data)
        logger.info("-----All Files loaded from dataset successful-----")

    short_dataset = dataset
    TRAIN = math.floor(len(short_dataset) * 0.8)
    train_dataset = short_dataset[:TRAIN]
    test_dataset = short_dataset[TRAIN:]
    batch_size = 1
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    train_size = len(train_dataloader)
    test_size = len(test_dataloader)

    # atom_type
    structure = POSCAR(fname=Path(f"{root_dir}/train_set/input/POSCAR_1-1")).to_structure()
    structure.find_neighbour_table(neighbour_num=12)

    model = Model()
    parameters = [(name, param) for name, param in model.named_parameters()]
    loss_fn = nn.MSELoss(reduction='mean')
    initial_lr = 0.1
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    train_loss_result = []
    test_loss_result = []
    test_min_loss = 100
    threshold = 1000

    for epoch in range(100):
        model.train()
        total_train_loss = 0.
        total_test_loss = 0.
        for step, data in enumerate(train_dataloader):
            atom_feature, bond_dist3d_input, bond_dist3d_output, adj_matrix, adj_matrix_tuple = data_prepare(structure,
                                                                                                             data)

            # input: POSCAR, output: CONTCAR, loss: (out - CONTCAR); bond_update.grad = 1. / (batch*N*M*3)
            bond_update = model(bond_dist3d_input[:, -2:])
            # bond_update.retain_grad()
            train_loss = loss_fn(bond_update, bond_dist3d_output[:, -2:])
            total_train_loss += train_loss
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # if step == 0:
            #     print("AtomType_C1c.linear.weight: ", torch.max(parameters[0][1].grad))
        scheduler.step()

        model.eval()
        for step, data in enumerate(test_dataloader):
            atom_feature, bond_dist3d_input, bond_dist3d_output, adj_matrix, adj_matrix_tuple = data_prepare(structure,
                                                                                                             data)

            bond_update = model(bond_dist3d_input[:, -2:])
            test_loss = loss_fn(bond_update, bond_dist3d_output[:, -2:])
            total_test_loss += test_loss

        if torch.cuda.is_available():
            train_loss_result.append((total_train_loss / train_size).cpu().detach().numpy())
            test_loss_result.append((total_test_loss / test_size).cpu().detach().numpy())
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

    logger.info("---------------End---------------")


if __name__ == '__main__':
    main()