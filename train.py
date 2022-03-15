import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from Discriminator import Discriminator
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
    Epochs_generator = 100
    iterations = 1000
    mini_batch_size = 16  # Size 192x192
    lr_generator = 1e-4
    optimizer_gen = torch.optim.Adam(gen.parameters(), lr_generator)

    # Specify hyperparameters Discriminator
    alpha = 1e-5
    beta = 5e-3
    lr_disc = 1e-4
    optimizer_disc = torch.optim.Adam(disc.parameters(), lr_disc)
    epoch_both = 150

    print("oka")


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
    vgg_cut = vgg_original.features
    vgg_cut.to(device)

    train(model, disc, vgg_cut)

    # PSNR metric (equation 3)

    # Loss functions:
    l1_loss = nn.L1Loss()
    adv_loss = ...

    Total_D_loss = nn.BCELoss()  # Still needs to be averaged over number of samples