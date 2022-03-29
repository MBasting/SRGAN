"""
Calculates the Dloss according to eq. 5 in the paper

discriminator outputs a single value between [0,1], where 1=real and 0=fake. 
The discriminator output is discretized(rounded) to classify an image as either real or fake.

for training the images are labeled as real or fake: Yhr=1 Ysr=0.

implementation of the authors:

with tf.variable_scope('Discriminator_loss'):
                # the cross entropy should approach zero for perfect discrimination. random discrimination should be 0.5 for both, summing to 1           
    #            d_loss = (d_loss_real + d_loss_fake)
                
                d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
                d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
                
                d_loss = d_loss1 + d_loss2
            # here, fake is 0 if the discriminator is good.
            # the generator aims to raise the label values towards 1
"""
import torch.nn

# test data for fucntion
#YLabel = torch.tensor([1., 1., 0., 1.])
#OutputDiscrim = torch.randn(4)

def DLoss(YLabel, OutputDiscrim):
    """
    Function that calculates the binary cross entropy loss according to eq. 5 of the paper.

    YLabel: The groundtruth value for the images, either 1.0 or 0.0.(must be floats) 
    OutputDiscrim: Value between 0 and 1 to classify the image.
    """
    lossFunc = torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    Dloss = lossFunc(OutputDiscrim, YLabel)
    return Dloss

#loss = DLoss(YLabel, OutputDiscrim)
#print(loss)