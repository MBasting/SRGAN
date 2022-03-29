"""
VGG19 is a pretrained neural network that expects coloured images. 
The output of the fourth cnn layer, prior to the fifth max-pooling layer of this network
is used as the output feature map. 

The HR and SR images are fed to the vgg19 network and the feature maps are then fed into the L2 loss function 
to obtain the VGG19loss. 
"""

import torch
from torchvision import transforms
from torchvision.models import vgg19
from PIL import Image
from SRCNN_Loss import L2loss

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
    SRinput_tensor = preprocess(SR_image)           # execute preprocessing

    HRinput_tensor = preprocess(HR_image)

    input_batch = torch.cat((SRinput_tensor, HRinput_tensor), 0)    # concatenate input batches.

    # evaluate the images on the network.
    vgg_cut.eval()                      # turns of layers not needed for evaluation
    with torch.no_grad():               # stops gradient calculation since not needed for evaluation.
        output = vgg_cut(input_batch)
    
    # calculate the loss
    # the first part of the batch are the sr images the secon part are the hr images. 
    # the images are concatenated to be able to go through the vgg19 network at ones.
    vggloss = L2loss(output[0:SRinput_tensor.size()[0],:,:,:], output[SRinput_tensor.size()[0]:input_batch.size()[0],:,:,:]) # 
    
    return vggloss

#vggloss = VGG19_Loss(SR_image, HR_image)
#print(vggloss)