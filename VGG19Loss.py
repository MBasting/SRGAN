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
from train import try_gpu

# test images
SR_image = Image.open("./ChessBord.jpg")
HR_image = Image.open("./ChesBord2.jpg")

def VGG19_Loss(SR_image, HR_image):
    """
    Function that calculates the vgg19 loss by processing the two input images with the vgg19 network.
    SR_image: The first input image
    HR_image: The second input image 
    """
    # Required preporsessing for the vgg network see: https://pytorch.org/hub/pytorch_vision_vgg/
    preprocess = transforms.Compose([           
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # get images in the right format to be processed by the vgg network.
    SRinput_tensor = preprocess(SR_image)           # execute preprocessing
    SRinput_batch = SRinput_tensor.unsqueeze(0)     # create imput batch as expected from network.

    HRinput_tensor = preprocess(HR_image)
    HRinput_batch = HRinput_tensor.unsqueeze(0)

    input_batch = torch.cat((SRinput_batch, HRinput_batch), 0)     # concatenate input batches.

    # create the vgg19 network
    vgg_original = vgg19(pretrained=True)   # Load the pretrained network
    vgg_cut = vgg_original.features[:-1]    # Use all Layers before fully connected layer and before max pool layer
    device = try_gpu()
    vgg_cut.to(device)

    # evaluate the images on the network.
    vgg_cut.eval()                      # turns of layers not needed for evaluation
    with torch.no_grad():               # stops gradient calculation since not needed for evaluation.
        output = vgg_cut(input_batch)
    
    # calculate the loss
    vggloss = L2loss(output[0,:,:,:], output[1,:,:,:])
    
    return vggloss.item()

#vggloss = VGG19_Loss(SR_image, HR_image)
#print(vggloss)