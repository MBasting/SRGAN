import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from Discriminator import Discriminator
from Load_dataset import RockDataset, show_rock_samples
from SRCNN import SRCNN
from torchvision.models import vgg19

m_seed = 1  # or use random.randint(1, 10000) for random reed
random.seed(m_seed)
torch.manual_seed(m_seed)


def try_gpu():
    """
    If GPU is available, return torch.device as cuda:0; else return torch.device
    as cpu.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


def train(gen, disc, vgg):

    # Specify hyperparameters Generator (SRCNN)
    epochs_gen = 100
    nr_of_iterations = 1000
    mini_batch_size = 16  # Size 192x192
    lr_generator = 1e-4
    alpha = 1e-5
    beta = 5e-3
    optimizer_gen = torch.optim.Adam(gen.parameters(), lr_generator)

    # Specify hyperparameters Discriminator
    lr_disc = 1e-4
    optimizer_disc = torch.optim.Adam(disc.parameters(), lr_disc)
    epoch_both = 150

    # Load Dataset
    rock_s_4_train = RockDataset("DeepRockSR-2D/shuffled2D/shuffled2D_train_LR_unknown_X4", "DeepRockSR-2D/shuffled2D/shuffled2D_train_HR", True)
    rock_s_4_valid = RockDataset("DeepRockSR-2D/shuffled2D/shuffled2D_valid_LR_unknown_X4", "DeepRockSR-2D/shuffled2D/shuffled2D_valid_HR")
    rock_s_4_test = RockDataset("DeepRockSR-2D/shuffled2D/shuffled2D_test_LR_unknown_X4", "DeepRockSR-2D/shuffled2D/shuffled2D_test_HR")

    # Configure Data Loaders
    mini_batch_size_test = 8 # These are higher resolution images and thus might not fit into batches of size 16!
    mini_batch_size_valid = 8
    rock_data_loader_train = DataLoader(rock_s_4_train, batch_size=mini_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    rock_data_loader_valid = DataLoader(rock_s_4_valid, batch_size=mini_batch_size_valid, shuffle=True, num_workers=0)
    rock_data_loader_test = DataLoader(rock_s_4_test, batch_size=mini_batch_size_test, shuffle=True, num_workers=0)

    # Training of SRCNN (Generator)
    for phase, epochs in enumerate([epochs_gen, epoch_both]):
        for epoch in tqdm(range(epochs), position=0, desc='Epoch', leave=True):

            # Specify Inner progressbar which keeps track of training inside epoch
            inner = tqdm(total=nr_of_iterations, desc='Batch', position=1, leave=False)
            # Necessary because batch_size * len(data_loader) < nr_iterations
            # So we loop over the data another time
            iteration = 0
            stop = False
            while(not stop):
                for i_batch, sample_batch in enumerate(rock_data_loader_train):

                    # Stop when number of iterations has been reached
                    if iteration >= nr_of_iterations:
                        stop = True
                        break

                    # Training should take place here



                    # Update progressbar and iteration var
                    iteration += 1
                    inner.update(1)

            # Validation should take place here




if __name__ == '__main__':
    device = try_gpu()
    img = torch.randn((1, 1, 50, 50), dtype=torch.float32).to(device)
    model = SRCNN(1)
    disc = Discriminator(1)
    model.to(device)
    disc.to(device)
    z = model(img)
    y = disc(z)
    print(y)

    # VGG uses Coloured images originally so need to duplicate channels or something?
    vgg_original = vgg19(pretrained=True)
    vgg_cut = vgg_original.features[:-1] # Use all Layers before fully connected layer and before max pool layer
    vgg_cut.to(device)

    train(model, disc, vgg_cut)

    # PSNR metric (equation 3)

    # Loss functions:
    l1_loss = nn.L1Loss()
    adv_loss = ...

    Total_D_loss = nn.BCELoss()  # Still needs to be averaged over number of samples