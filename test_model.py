import math
from pathlib import Path

import torch.cuda
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

from common.logger import root_dir, logger
from common.manager import DirManager
from network.dataset import StructureDataset
from network.model import Model


def main():
    # logger.setLevel(logging.DEBUG)
    torch.set_printoptions(profile='full')
    logger.info("---------------Start----------------")

    input_dir = DirManager(dname=Path(f'{root_dir}/train_set/input'))
    output_dir = DirManager(dname=Path(f'{root_dir}/train_set/output'))

    if not Path("dataset.pth").exists():
        dataset = StructureDataset(input_dir, output_dir)
        torch.save(dataset.data, "dataset.pth")
        logger.info("-----All Files loaded successful-----")
    else:
        data = torch.load("dataset.pth")
        dataset = StructureDataset(input_dir, output_dir, data=data)
        logger.info("-----All Files loaded from dataset successful-----")

    TRAIN = math.floor(len(dataset) * 0.8)
    train_dataset = dataset[:TRAIN]
    test_dataset = dataset[TRAIN:]
    batch_size = 1
    train_dataloader = DataLoader(train_dataset,  batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    train_size = len(train_dataloader)
    test_size = len(test_dataloader)
    initial_lr = 0.1
    model = Model(25, 25, 3, 3, bias=True)
    loss_fn = nn.L1Loss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** (epoch / 10))
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    loss_result = []
    for epoch in range(10):
        model.train()
        total_train_loss = 0.
        total_test_loss = 0.
        for step, data in enumerate(train_dataloader):
            atom_feature, bond_dist3d, adj_matrix, adj_matrix_tuple, bond_dist3d_target = data
            if torch.cuda.is_available():
                atom_feature = atom_feature.cuda()
                bond_dist3d = bond_dist3d.cuda()
                bond_dist3d_target = bond_dist3d_target.cuda()
                adj_matrix = adj_matrix.cuda()
                adj_matrix_tuple = adj_matrix_tuple.cuda()

            out = model(atom_feature, bond_dist3d, adj_matrix, adj_matrix_tuple)
            train_loss = loss_fn(bond_dist3d_target, out[1])
            total_train_loss += train_loss
            loss_result.append(train_loss.cpu().detach().numpy())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        for step, data in enumerate(test_dataloader):
            atom_feature, bond_dist3d, adj_matrix, adj_matrix_tuple, bond_dist3d_target = data
            if torch.cuda.is_available():
                atom_feature = atom_feature.cuda()
                bond_dist3d = bond_dist3d.cuda()
                bond_dist3d_target = bond_dist3d_target.cuda()
                adj_matrix = adj_matrix.cuda()
                adj_matrix_tuple = adj_matrix_tuple.cuda()

            out = model(atom_feature, bond_dist3d, adj_matrix, adj_matrix_tuple)
            test_loss = loss_fn(bond_dist3d_target, out[1])
            total_test_loss += test_loss

        logger.info(
                    f"Training {(epoch + 1)}: train_loss = {total_train_loss/train_size:15.12f}, "
                    f"test_loss = {total_test_loss/test_size:15.12f}, "
                    f"learning_rate = {optimizer.state_dict()['param_groups'][0]['lr']:15.12f}")

    logger.info("---------------End---------------")
    plt.plot(loss_result)
    # plt.show()


if __name__ == '__main__':
    main()
