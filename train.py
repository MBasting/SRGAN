import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from Discriminator import Discriminator
from Load_dataset import RockDataset, show_rock_samples
from SRCNN import SRCNN
from torchvision.models import vgg19
from DLoss import DLoss
from SRCNN_Loss import L1loss, L2loss, PSNR
from VGG19Loss import VGG19_Loss

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
    rock_data_loader_train = DataLoader(rock_s_4_train, batch_size=mini_batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)  # Number of workers needs some figgeting to increase speedup
    rock_data_loader_valid = DataLoader(rock_s_4_valid, batch_size=mini_batch_size_valid, shuffle=True, num_workers=0)
    rock_data_loader_test = DataLoader(rock_s_4_test, batch_size=mini_batch_size_test, shuffle=True, num_workers=0)

    # torch.backends.cudnn.benchmark = True  # Could lead to some speedup later according to a blogpost

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
                    # with autocast():
                    #     print("okay")


                    # Update progressbar and iteration var
                    iteration += 1
                    inner.update(1)

            with torch.no_grad(): # Since we are evaluating no gradients need to calculated -> speedup
                inner = tqdm(total=len(rock_data_loader_valid), desc='Validation', position=1, leave=False)
                for i_batch, sample_batch in enumerate(rock_data_loader_valid):
                    inner.update(1)
                    if (i_batch) > 10:
                        break
                    # # Validation should take place here
                    # break

    with torch.no_grad(): # No gradient calculation needed
        inner = tqdm(total=len(rock_data_loader_test), desc='Testing', position=1, leave=False)
        for t_batch in rock_data_loader_test:
            # Any sort of testing should take place here
            break



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

    # # VGG uses Coloured images originally so need to duplicate channels or something?
    # vgg_original = vgg19(pretrained=True)
    # vgg_cut = vgg_original.features[:-1] # Use all Layers before fully connected layer and before max pool layer
    # vgg_cut.to(device)

    train(model, disc, None)

    # # Loss functions:
    # l1_loss = L1loss(SR, HR)
    # l2_loss = L2loss(SR, HR)
    #
    # # PSNR metric (equation 3)
    # # is a standard function in pytorch that can be attached to the network,
    # # but the authors use I=2 which we cannot set using the pytorch way.
    # # what do we want?
    # psnr = PSNR(l2_loss) # our implementation where I=2.
    #
    # vgg19_loss = VGG19_Loss(SR_image, HR_image)
    #
    # adv_loss = ...
    #
    # Total_D_loss = DLoss(YLabel, OutputDiscrim)