import torch.nn as nn


class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, hidden_channel, kernel, stride, alpha=0.2):
        super(DiscriminatorBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channel, kernel, stride)
        self.batchnorm = nn.BatchNorm2d(hidden_channel)
        self.lrelu = nn.LeakyReLU(alpha)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.lrelu(x)
        return x

class Discriminator(nn.Module):

    def __init__(self, in_channels, hidden_channels=None, stride=None, kernel_size=3, alpha=0.2):

        super(Discriminator, self).__init__()

        if stride is None:
            stride = [1, 2, 1, 2, 1, 2, 1, 2]
        if hidden_channels is None:
            hidden_channels = [64, 64, 128, 128, 256, 256, 512, 512]
        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0], kernel_size, stride[0])
        self.lrelu = nn.LeakyReLU(alpha)

        self.disc_blocks = nn.ModuleList([DiscriminatorBlock(hidden_channels[i-1], hidden_channels[i], kernel_size, stride[i]) for i in range(1, len(hidden_channels))])

        self.flatten = nn.Flatten()
        # TODO: verify Input size (9x9x512)
        self.fc1 = nn.Linear(41472, 1024)

        self.fc2 = nn.Linear(1024, 1)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        x = self.lrelu(self.conv1(x))

        for layer in self.disc_blocks:
            x = layer(x)
        x = self.flatten(x)
        x = self.lrelu(self.fc1(x))
        pred = self.sigmoid(self.fc2(x))
        return pred




