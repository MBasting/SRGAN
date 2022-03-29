import torch.nn

'''
https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html 
documentatation example from pytorch. 
from the example it can be seen that the first term in the loss function is the x value in the log.
the second term in the loss function is the target, which represents the y value. 

>>> m = nn.Sigmoid()
>>> loss = nn.BCELoss()
>>> input = torch.randn(3, requires_grad=True)
>>> target = torch.empty(3).random_(2)
>>> output = loss(m(input), target)
>>> output.backward()
'''

def ADVloss(PSR, device):
    """
    Function that calculates the adversarial loss term according to eq. 7 in the paper.

    PSR: a 1D tensor that holds the probability that the picture is a super resolution picture, p(SR).
    note: p(SR) should not be 0 or 1 since the log function is undefined for these values.
    """
    lossFunc = torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    ADVloss = lossFunc(PSR, torch.ones(PSR.size(), device=device))     # binary cross entropy function with y set to one.
    return ADVloss
