from pathlib import Path

import numpy as np
import torch.cuda
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from common.io_file import POSCAR, CONTCAR
from common.logger import root_dir, logger
from common.manager import DirManager
from model.nn_model import Model

PERIOD = 7
GROUP = 18


class StructureDataset(Dataset):
    def __init__(self, input_dir: DirManager, output_dir: DirManager, data=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.data = self.load_data() if data is None else data

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index], self.data[2][index], self.data[3][index], self.data[4][index]

    def __len__(self):
        return self.data[0].shape[0]

    def load_data(self):
        atom_feature_input = []
        bond_dist3d_input = []
        bond_dist3d_output = []
        adj_matrix_input = []
        adj_matrix_tuple_input = []
        count = 0
        for fname_input, fname_out in zip(self.input_dir.all_files_path, self.output_dir.all_files_path):

            # load structures
            structure, atom_feature, bond_dist3d, adj_matrix, adj_matrix_tuple = self._load_input(
                fname=fname_input)
            bond_dist3d_target = self._load_output(fname=fname_out, adj_matrix=adj_matrix)

            # align the input and output
            diff = (bond_dist3d_target - bond_dist3d) / (0.5 * structure.lattice.length)
            diff_around = np.around(diff)
            diff_around = diff_around * 0.5 * structure.lattice.length
            bond_dist3d_target -= diff_around

            count += 1
            if count % 100 == 0:
                logger.info(f"-----{count} files have been loaded!-----")

            # put the structure into the dataset
            atom_feature_input.append(atom_feature)
            bond_dist3d_input.append(bond_dist3d)
            adj_matrix_input.append(adj_matrix)
            adj_matrix_tuple_input.append(adj_matrix_tuple)
            bond_dist3d_output.append(bond_dist3d_target)

        return torch.Tensor(np.array(atom_feature_input)), torch.Tensor(np.array(bond_dist3d_input)), \
               torch.LongTensor(np.array(adj_matrix_input)), torch.LongTensor(np.array(adj_matrix_tuple_input)), \
               torch.Tensor(np.array(bond_dist3d_output))

    def _load_input(self, fname):
        structure = POSCAR(fname=fname).to_structure(style="Slab")
        structure.find_neighbour_table(neighbour_num=12)

        atom_feature_period = F.one_hot(torch.LongTensor(structure.atoms.period), num_classes=PERIOD)
        atom_feature_group = F.one_hot(torch.LongTensor(structure.atoms.group), num_classes=GROUP)
        atom_feature = torch.cat((atom_feature_period, atom_feature_group), dim=1).numpy()
        adj_matrix = structure.neighbour_table.index
        adj_matrix_tuple = structure.neighbour_table.index_tuple
        bond_dist3d = structure.neighbour_table.dist3d

        return structure, atom_feature, bond_dist3d, adj_matrix, adj_matrix_tuple

    def _load_output(self, fname, adj_matrix):
        structure = CONTCAR(fname=fname).to_structure(style="Slab")
        structure.find_neighbour_table_from_index(adj_matrix=adj_matrix)

        return structure.neighbour_table.dist3d


def main():
    # logger.setLevel(logging.DEBUG)
    logger.info("---------------Start----------------")

    input_dir = DirManager(dname=Path(f'{root_dir}/train_set/input'))
    output_dir = DirManager(dname=Path(f'{root_dir}/train_set/output'))

    if not Path("dataset.pth").exists():
        dataset = StructureDataset(input_dir, output_dir)
        torch.save(dataset.data,"dataset.pth")
        logger.info("-----All Files loaded successful-----")
    else:
        data = torch.load("dataset.pth")
        dataset = StructureDataset(input_dir, output_dir, data=data)
        logger.info("-----All Files loaded from dataset successful-----")

    dataloader = DataLoader(dataset, batch_size=4)
    initial_lr = 0.1
    model = Model(25, 25, 3, 3)
    loss_fn = nn.L1Loss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** (epoch / 10))
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    model.train()
    for epoch in range(2):
        for step, data in enumerate(dataloader):
            atom_feature, bond_dist3d, adj_matrix, adj_matrix_tuple, bond_dist3d_target = data
            if torch.cuda.is_available():
                atom_feature = atom_feature.cuda()
                bond_dist3d = bond_dist3d.cuda()
                bond_dist3d_target = bond_dist3d_target.cuda()
                adj_matrix = adj_matrix.cuda()
                adj_matrix_tuple = adj_matrix_tuple.cuda()

            out = model(atom_feature, bond_dist3d, adj_matrix, adj_matrix_tuple)
            loss = loss_fn(bond_dist3d_target, out[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 10 == 0:
                logger.info(
                    f"Training {(epoch + 1)}-{(step + 1)}: loss = {loss:15.12f}, "
                    f"learning_rate = {optimizer.state_dict()['param_groups'][0]['lr']:15.12f}")

    logger.info("---------------End---------------")


if __name__ == '__main__':
    main()
