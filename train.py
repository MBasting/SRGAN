import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


from SRCNN import SRCNN


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

def train(model, img, device=try_gpu()):
    print(img)
    y = model(img)
    print(y)
    pass

if __name__ == '__main__':
    img = x = torch.randn((1, 1, 50, 50), dtype=torch.float32)
    model = SRCNN(1)
    train(model, img)