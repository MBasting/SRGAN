import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from Discriminator import Discriminator
from SRCNN import SRCNN
from torchvision.models import vgg19



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


def train(model, img):
    print(img)
    y = model(img)
    print(y)
    pass


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
