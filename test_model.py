import math
import random
from pathlib import Path

import torch.cuda
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

from common.io_file import POSCAR
from common.logger import root_dir, logger
from common.manager import DirManager
from common.structure import Structure
from network.dataset import StructureDataset
from network.model import Model


def main():
    # logger.setLevel(logging.DEBUG)
    torch.set_printoptions(profile='full')
    logger.info("---------------Start----------------")

    input_dir = DirManager(dname=Path(f'{root_dir}/train_set/input-20'))
    output_dir = DirManager(dname=Path(f'{root_dir}/train_set/output-20'))
    dataset_path = "dataset-mini.pth"

    if not Path(dataset_path).exists():
        dataset = StructureDataset(input_dir, output_dir)
        # torch.save(dataset.data, dataset_path)
        logger.info("-----All Files loaded successful-----")
    else:
        data = torch.load(dataset_path)
        dataset = StructureDataset(input_dir, output_dir, data=data)
        logger.info("-----All Files loaded from dataset successful-----")

    short_dataset = dataset[:20]
    TRAIN = math.floor(len(short_dataset) * 0.8)
    train_dataset = short_dataset[:TRAIN]
    test_dataset = short_dataset[TRAIN:]
    batch_size = 1
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    train_size = len(train_dataloader)
    test_size = len(test_dataloader)
    initial_lr = 0.1
    model = Model(25, 25, 3, 3, bias=True)
    loss_fn = nn.L1Loss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    train_loss_result = []
    test_loss_result = []
    test_min_loss = 100
    threshold = 1.3
    for epoch in range(100):
        model.train()
        total_train_loss = 0.
        total_test_loss = 0.
        for step, data in enumerate(train_dataloader):
            atom_feature, bond_dist3d_input, adj_matrix, adj_matrix_tuple, bond_dist3d_output = data
            if torch.cuda.is_available():
                atom_feature = atom_feature.cuda()
                bond_dist3d_input = bond_dist3d_input.cuda()
                bond_dist3d_output = bond_dist3d_output.cuda()
                adj_matrix = adj_matrix.cuda()
                adj_matrix_tuple = adj_matrix_tuple.cuda()

            # input: POSCAR, output: CONTCAR, loss: (out - CONTCAR)
            atom_update, bond_update = model(atom_feature, bond_dist3d_input, adj_matrix, adj_matrix_tuple)
            train_loss = loss_fn(bond_update, bond_dist3d_output)
            total_train_loss += train_loss
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        for step, data in enumerate(test_dataloader):
            atom_feature, bond_dist3d_input, adj_matrix, adj_matrix_tuple, bond_dist3d_output = data
            if torch.cuda.is_available():
                atom_feature = atom_feature.cuda()
                bond_dist3d_input = bond_dist3d_input.cuda()
                bond_dist3d_output = bond_dist3d_output.cuda()
                adj_matrix = adj_matrix.cuda()
                adj_matrix_tuple = adj_matrix_tuple.cuda()

            atom_update, bond_update = model(atom_feature, bond_dist3d_input, adj_matrix, adj_matrix_tuple)
            test_loss = loss_fn(bond_update, bond_dist3d_output)
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
    plt.savefig("loss.svg")

    # model save
    torch.save(model, "model.pth")

    # model predict
    atom_feature, bond_dist3d_input, adj_matrix, adj_matrix_tuple, bond_dist3d_output = test_dataset.data
    index = random.choice(list(range(len(atom_feature))))
    structure = POSCAR(fname=Path(f"{root_dir}/train_set/input/POSCAR_1-1")).to_structure()
    structure_target = Structure.from_adj_matrix(structure, adj_matrix[index].cpu().detach().numpy(),
                                                 adj_matrix_tuple[index].cpu().detach().numpy(),
                                                 bond_dist3d_output[index].cpu().detach().numpy(), 0)
    structure_target.to_POSCAR("CONTCAR_target")

    if torch.cuda.is_available():
        atom_feature = atom_feature.cuda()
        bond_dist3d_input = bond_dist3d_input.cuda()
        adj_matrix = adj_matrix.cuda()
        adj_matrix_tuple = adj_matrix_tuple.cuda()

    predict = model(atom_feature, bond_dist3d_input, adj_matrix, adj_matrix_tuple)
    structure_predict = Structure.from_adj_matrix(structure, adj_matrix[index].cpu().detach().numpy(),
                                                  adj_matrix_tuple[index].cpu().detach().numpy(),
                                                  predict[1][index].cpu().detach().numpy(), 0)
    structure_predict.to_POSCAR(f"CONTCAR_predict")
    logger.info("---------------End---------------")


if __name__ == '__main__':
    main()
