import torch.nn as nn

def bicubic_interpolation(image_LR, scale_factor):
    """ 
    Creates a bicupic interpolated image 

    Input is tensor image with size [3, 125, 125], all three channels are identical, 
    interpolation is therefore only performed on one channel and consequently expanded
    to original three channels. 
    Output image depends on scale factor (2x or 4x)
    """

    image_LR = image_LR[0]

    up = nn.Upsample(scale_factor=scale_factor, mode='bicubic')
    image_LR = image_LR.view(1, 1, image_LR.shape[0], image_LR.shape[1])
    upscale_image_LR = up(image_LR).view(1, image_LR.shape[2]*4, image_LR.shape[3]*4)
    return upscale_image_LR