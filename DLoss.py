"""
Calculates the Dloss according to eq. 5 in the paper

discriminator outputs a single value between [0,1], where 1=real and 0=fake. 
The discriminator output is discretized(rounded) to classify an image as either real or fake.

for training the images are labeled as real or fake: Yhr=1 Ysr=0.

I do not know how to implement the DLoss function, in eq. 4 the BXEsr has a term that is always 0 namely Ysr. Therefor BXEsr=0
The second problem is that both the BXEhr and BXEsr sum over N, yet there is no N behind the sum. I do not know what to sum over.

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

def DLoss():
    