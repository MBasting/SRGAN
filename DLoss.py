"""
Calculates the Dloss according to eq. 5 in the paper

discriminator outputs a single value between [0,1], where 1=real and 0=fake. 
The discriminator output is discretized(rounded) to classify an image as either real or fake.

for training the images are labeled as real or fake: Yhr=1 Ysr=0.

I do not know how to implement the DLoss function, in eq. 4 the BXEsr has a term that is always 0 namely Ysr. Therefor BXEsr=0
The second problem is that both the BXEhr and BXEsr sum over N, yet there is no N behind the sum. I do not know what to sum over.
"""

def DLoss():
    