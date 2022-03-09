import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual Block module containing:
        - Convolution layer
        - PReLU
        - Convolution Layer
        - Addition/ Residual connection
    """

    def __init__(self, in_channels, hidden_channels, kernel_size, padding):
        """
        Initialize Layer

        :param in_channels: number of input channels
        :param hidden_channels: List of numbers containing the number of hidden features
        :param kernel_size: Kernel size
        :param padding: Padding
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0], kernel_size, padding=padding)
        self.PReLU = nn.PReLU()
        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size, padding=padding)

    def forward(self, x):
        """
        Forward pass
        :param x: Input
        :return: Output of forward pass
        """
        res = self.conv1(x)
        x = self.PReLU(x)
        x = self.conv2(x)
        x = x + res
        return x


class UpscaleLayer(nn.Module):
    """
    Upscale/ Upsample module containing:
        - Convolution Layer
        - Upsample/ Shuffle Layer
        - PreLU
    """

    def __init__(self, in_channels, hidden_channels, kernel_size, upscale_factor, padding):
        """
        Initialize Layer
        :param in_channels: Number of input channels
        :param hidden_channels: List of numbers containing the number of hidden features
        :param kernel_size: Kernel size
        :param upscale_factor: Factor of upsample
        :param padding: Padding
        """
        super(UpscaleLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.upscale = nn.PixelShuffle(upscale_factor)
        self.PReLU = nn.PReLU()

    def forward(self, x):
        """
        :param x: Input
        :return: Output of forward pass
        """
        x = self.conv1(x)
        x = self.upscale(x)
        x = self.PReLU(x)
        return x


class SRCNN(nn.Module):
    """
    ## TODO: LINK TO PAPER
    SRCNN implementation
    """

    def __init__(self, in_channels, hidden_channels=[64, 64, 64, 64, 256, 256, 1]):
        """
        Initialize SRCNN network with correct layers and dimensions
        :param in_channels: Size of the input image
        :param hidden_channels: List of numbers containing the number of hidden features
        """
        super(SRCNN, self).__init__()

        # TODO: Verify padding mode
        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, padding=1)
        self.residual = ResidualBlock(hidden_channels[0], [hidden_channels[1], hidden_channels[2]], kernel_size=3,
                                      padding=1)

        self.conv2 = nn.Conv2d(hidden_channels[2], hidden_channels[3], kernel_size=3, padding=1)

        # Note that the output of the upscale layers are 2xheight and 2xwidth and hidden_channels/4 !
        self.upscale_1 = UpscaleLayer(hidden_channels[3], hidden_channels[4], 3, 2, padding=1)
        self.upscale_2 = UpscaleLayer(hidden_channels[3], hidden_channels[5], 3, 2, padding=1)

        self.conv3 = nn.Conv2d(hidden_channels[3], hidden_channels[6], kernel_size=1)

    def forward(self, x):
        """
        Forward pass

        :param x: Input Image (Low resolution)
        :return: Generated Super resolution Image
        """
        x = self.conv1(x)
        residual = x

        x = self.residual(x)

        x = self.conv2(x)
        x = x + residual

        x = self.upscale_1(x)
        x = self.upscale_2(x)

        x = self.conv3(x)

        return x
