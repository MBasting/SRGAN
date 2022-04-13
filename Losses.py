import math
import torch.nn as nn
import torch
from torchvision import transforms


def DLoss(YLabel, OutputDiscrim):
    """
    Function that calculates the binary cross entropy loss according to eq. 5 of the paper.

    YLabel: The groundtruth value for the images, either 1.0 or 0.0.(must be floats)
    OutputDiscrim: Value between 0 and 1 to classify the image.
    """
    lossFunc = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    Dloss = lossFunc(OutputDiscrim, YLabel)
    return Dloss


def L1loss(SR, HR):
    """
    Calculates the L1 loss as in eq. 2

    SR: super resolution image
    HR: High resolution image
    """
    loss = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
    L1loss = loss(SR, HR)
    return L1loss


def L2loss(SR, HR):
    """
    Calculates the L2 loss as in eq. 1

    SR: super resolution image
    HR: High resolution image
    """
    loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    L2loss = loss(SR, HR)
    return L2loss


def PSNR(L2loss, I=2):
    """
    Function that calculates the PSNR metric according to eq. 3
    L2loss: a float
    I: set to 2 since the paper assumes the HR and SR pixel values are between [-1,1]
    """
    x = I ** 2 / L2loss  # calculating the argument for the log
    psnr = 10 * math.log10(x)  # calculating the psnr as in eq. 3
    return psnr


def ADVloss(PSR, device):
    """
    Function that calculates the adversarial loss term according to eq. 7 in the paper.

    PSR: a 1D tensor that holds the probability that the picture is a super resolution picture, p(SR).
    note: p(SR) should not be 0 or 1 since the log function is undefined for these values.
    """
    lossFunc = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    ADVloss = lossFunc(PSR, torch.ones(PSR.size(), device=device))  # binary cross entropy function with y set to one.
    return ADVloss


def VGG19_Loss(SR_image, HR_image, vgg_cut):
    """
    Function that calculates the vgg19 loss by processing the two input images with the vgg19 network.
    SR_image: The first input image
    HR_image: The second input image
    """
    # Required preprocessing for the vgg network see: https://pytorch.org/hub/pytorch_vision_vgg/
    SR_image = torch.cat((SR_image, SR_image, SR_image), 1)
    HR_image = torch.cat((HR_image, HR_image, HR_image), 1)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # get images in the right format to be processed by the vgg network.
    SRinput_tensor = preprocess(SR_image)  # execute preprocessing

    HRinput_tensor = preprocess(HR_image)

    input_batch = torch.cat((SRinput_tensor, HRinput_tensor), 0)  # concatenate input batches.

    # evaluate the images on the network.
    vgg_cut.eval()  # turns of layers not needed for evaluation
    with torch.no_grad():  # stops gradient calculation since not needed for evaluation.
        output = vgg_cut(input_batch)

    # calculate the loss
    # the first part of the batch are the sr images the secon part are the hr images.
    # the images are concatenated to be able to go through the vgg19 network at ones.
    vggloss = L2loss(output[0:SRinput_tensor.size()[0], :, :, :],
                     output[SRinput_tensor.size()[0]:input_batch.size()[0], :, :, :])  #

    return vggloss
