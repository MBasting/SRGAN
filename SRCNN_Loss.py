""""
calculates the loss function to train the generator, as well as the PSNR metric.

according to the paper the input image will be a tensor of shape (Ny, Nx, Nc) with Ny and Nx the coordinates of the picture
and Nc the color channels. 
The output will then consist of a tensor with shape (4Ny, 4Nx, Nc), the SR image. 
This will then be compared to the HR image that should have the same shape.

The training is done with minibatches of 16 cropped images, not sure if that should be implemented here as well
"""
import torch
import math

# test images
SR = torch.randn((200, 200, 3), dtype=torch.float32) 
HR = torch.randn((200, 200, 3), dtype=torch.float32)

def Lloss(SR, HR):
    """
    Function to calculate the L1 and L2 loss as in equation 2 and 1 of the paper
    SR: Square image of size [Nx, Ny, Nc] 
    HR: Square image of size [Nx, Ny, Nc] 
    """
    
    diff = torch.sub(SR, HR)    #take difference elementwise
    
    #calculating L1 loss
    absDiff = torch.abs(diff)   # take the absolurte value elementwise
    sum1 = torch.sum(absDiff)    # sums all elements in the tensor, meaning that all color channels are summed as well.
    L1 = sum1 / (SR.size()[0]*SR.size()[0])        # devide by the number of pixels(=!elements) in the tensor. since every color channel has all pixels. 
    
    # calculating L2 loss
    square = torch.pow(diff, 2) # square the input elementwise
    sum2 = torch.sum(square)    
    L2 = sum2 / (SR.size()[0]*SR.size()[0])  # devide by the number of pixels(=!elements) in the tensor. since every color channel has all pixels. 

    return L1.item(), L2.item()

loss1, loss2 = Lloss(SR, HR)
print(loss1, loss2)

def PSNR(L2loss, I=2):
    """
    Function that calculates the PSNR metric according to eq. 3
    L2loss: a float
    I: set to 2 since the paper assumes the HR and SR pixel values are between [-1,1]
    """
    x = I**2 / L2loss           # calculating the argument for the log
    psnr = 10 * math.log10(x)   # calculating the psnr as in eq. 3
    return psnr

psnr = PSNR(loss2)
print(psnr)