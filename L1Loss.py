""""
recreates the second equation of the paper.

The absolute difference between the HR picture and SR picture is pixelwise taken.
Then all values are summed and devided by the amound of pixels in the image. 

"""
import torch

SR = torch.randn((50, 50), dtype=torch.float32)
HR = torch.randn((50, 50), dtype=torch.float32)
print(SR.size())
#print(SR-HR)

def L1Loss(SR, HR):
    """"
    Function expects a 2D tensor of 200 by 200, but the problem is that we will probably train with batch sizes.
    meaning that the loss needs to be calculated for 3D or 4D tensors. 
    """
    diff = torch.sub(SR, HR)    #take difference work for any dimension tensor
    absDiff = torch.abs(diff)   # take the absolurte value pointwise, works for any dimension

    sum = torch.sum(absDiff)    # sums all elements in the tensor only gives the desired outcome with a 2D tensor
    loss = sum / 200*200        # devide by the number of elements in the tensor. 
    return loss