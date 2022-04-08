# Reproducing Boosting Resolution and Recovering Texture of 2D and 3D Micro-CT Images with Deep Learning

## Introduction 

In this blog we will describe our attempt and result in reproducing the following paper: Boosting Resolution and Recovering Texture of 2D and 3D Micro-CT Images with Deep Learning (Wang et al., 2019). In this paper the authors explain and showcase the Enhanced Deep Super Resolution Generative Adversarial Network (EDSRGAN) they designed. This model is designed to take low-resolution(LR) images and superresolve a high-resolution(HR) image from it. These images can than be used for image-based analysis techniques. The model the authors designed is able to handle 2D and 3D images, but for the scope if this project the choice was made to focus only on reproducing the 2D images.

As the code that was given with this paper uses a lot of old libraries and has a lot of old dependencies, running this model and training with it becomes quite hard. However the results in the paper are promising and we would like to see if these hold up. With this some of the papers results seem questionable such as the lower PSNR values in figure 3. 

We recreated the network as described in the paper and trained it with the same data set that was used for the model within the paper. We then used these trained models to recreate the figures 2 and 3 from the paper to see how these compare.

## Implementation

### Loading the data

In order to load the rock data samples efficiently (from [https://www.digitalrocksportal.org/projects/215](https://www.digitalrocksportal.org/projects/215)), there is a need for an efficient Data Loader that can transfer the samples to the GPU in batches and Dataset class that holds the location to the images.  The Dataset additionally can apply transformation on the samples if needed, in our case this is Random Cropping on both the low resolution and high resolution data samples to 64 x 64 (LR) and 192 x 192 (HR) and converting the samples to 1 colour channel as all samples in the dataset are grey images represented by 3 identical colour channels. An important note here is that these crops are linked as the SRGAN should reproduce the HR images from LR images as best as possible using upscaling. The Torchvision transforms module has the `get_params` method that helps with that. 

```python
# Transform operations
t_random_crop = T.RandomCrop(48)

# Get random location parameters for Low Res image
params = t_random_crop.get_params(image_LR, (48, 48))

# Get same locations in the High res image
params_HR = list([i * 4 for i in params])

# Apply cropping
image_LR = T.functional.crop(image_LR, *params)
image_HR = T.functional.crop(image_HR, *params_HR)
```

Loading the corresponding LR and HR image can be done by removing "x4" from the LR filename, since the LR image and HR image have corresponding names.  A helpful resource in writing your own custom dataset and dataset loader can be found [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html). 

### Architecture

The next step was to get the SRCNN - generator and Discriminator, represented by Figure 1 in the paper, working in PyTorch. This proved to be slightly more difficult than expected because it was unclear which exact module SubPixel Convolution represented in PyTorch, why an input size of 48 x 48 was used, which type of padding was used in the convolution layers, what the backward arrow from the first residual connection to the second convolution layers means, and how exactly we should read the diagram in general. A clearer representation of the Generator and Discriminator can be found below, printed using the torchsummary. We assume that the backward connection is actually a mistake as we couldn't find information on it or a similar use case where such an arrow is used. 

```
SRGAN - Generator
----------------------------------------------------------------
        Layer (type)               Output Shape         Param # 
================================================================
            Conv2d-1           [-1, 64, 48, 48]             640 
            Conv2d-2           [-1, 64, 48, 48]          36,928 
             PReLU-3           [-1, 64, 48, 48]               1 
            Conv2d-4           [-1, 64, 48, 48]          36,928 
     ResidualBlock-5           [-1, 64, 48, 48]               0 
            Conv2d-6           [-1, 64, 48, 48]          36,928 
            Conv2d-7          [-1, 256, 48, 48]         147,712 
      PixelShuffle-8           [-1, 64, 96, 96]               0
             PReLU-9           [-1, 64, 96, 96]               1
     UpscaleLayer-10           [-1, 64, 96, 96]               0
           Conv2d-11          [-1, 256, 96, 96]         147,712
     PixelShuffle-12         [-1, 64, 192, 192]               0
            PReLU-13         [-1, 64, 192, 192]               1
     UpscaleLayer-14         [-1, 64, 192, 192]               0
           Conv2d-15          [-1, 1, 192, 192]              65
================================================================
Total params: 406,916
Trainable params: 406,916
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 97.03
Params size (MB): 1.55
Estimated Total Size (MB): 98.59
----------------------------------------------------------------
None

================================================================

Discriminator
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 190, 190]             640
         LeakyReLU-2         [-1, 64, 190, 190]               0
            Conv2d-3           [-1, 64, 94, 94]          36,928
       BatchNorm2d-4           [-1, 64, 94, 94]             128
         LeakyReLU-5           [-1, 64, 94, 94]               0
DiscriminatorBlock-6           [-1, 64, 94, 94]               0
            Conv2d-7          [-1, 128, 92, 92]          73,856
       BatchNorm2d-8          [-1, 128, 92, 92]             256
         LeakyReLU-9          [-1, 128, 92, 92]               0
DiscriminatorBlock-10          [-1, 128, 92, 92]               0
           Conv2d-11          [-1, 128, 45, 45]         147,584
      BatchNorm2d-12          [-1, 128, 45, 45]             256
        LeakyReLU-13          [-1, 128, 45, 45]               0
DiscriminatorBlock-14          [-1, 128, 45, 45]               0
           Conv2d-15          [-1, 256, 43, 43]         295,168
      BatchNorm2d-16          [-1, 256, 43, 43]             512
        LeakyReLU-17          [-1, 256, 43, 43]               0
DiscriminatorBlock-18          [-1, 256, 43, 43]               0
           Conv2d-19          [-1, 256, 21, 21]         590,080
      BatchNorm2d-20          [-1, 256, 21, 21]             512
        LeakyReLU-21          [-1, 256, 21, 21]               0
DiscriminatorBlock-22          [-1, 256, 21, 21]               0
           Conv2d-23          [-1, 512, 19, 19]       1,180,160
      BatchNorm2d-24          [-1, 512, 19, 19]           1,024
        LeakyReLU-25          [-1, 512, 19, 19]               0
DiscriminatorBlock-26          [-1, 512, 19, 19]               0
           Conv2d-27            [-1, 512, 9, 9]       2,359,808
      BatchNorm2d-28            [-1, 512, 9, 9]           1,024
        LeakyReLU-29            [-1, 512, 9, 9]               0
DiscriminatorBlock-30            [-1, 512, 9, 9]               0
          Flatten-31                [-1, 41472]               0
           Linear-32                 [-1, 1024]      42,468,352
        LeakyReLU-33                 [-1, 1024]               0
           Linear-34                    [-1, 1]           1,025
          Sigmoid-35                    [-1, 1]               0
================================================================
Total params: 47,157,313
Trainable params: 47,157,313
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.14
Forward/backward pass size (MB): 118.61
Params size (MB): 179.89
Estimated Total Size (MB): 298.64
----
```

Some extra information on the summary:

- ResidualBlock is a nn.Module with a convolution layer followed by a PReLu activation function and another convolution layer. The output of the first convolution layer is then summed to the second convolution layer to get the residual connection. 
- UpscaleLayer, is a nn.Module consisting of a convolution layer followed by a PixelShuffle layer and PReLu activation. 
- DiscriminatorBlock, is a nn.Module consisting of a convolution layer followed by a BatchNorm2d layer and LeakyReLU activation. 

The VGG model mentioned in the paper uses the fully convolutional output from the sixteenth layer of the 19-layer VGG network, however the provided code does make use of this max-pooling layer. But since this is a reproducibility project and we are not sure whether their code actually works we assume that the paper is correct. The VGG model can be imported using:

```python
from torchvision.models import vgg19

vgg_original = vgg19(pretrained=True)  # Load the pretrained network
vgg_cut = vgg_original.features[:-1]  # Use all Layers before fully connected layer and before max pool layer
```

This results in an additional 20,024,384 parameters. Summing these parameters does not result in the same number of parameters described in the paper, but we could not figure out what the reason of this discrepancy was. 

### Training Procedure

Before jumping in the explanation of the training loop, we briefly give a short explanation on the loss functions used. All loss functions can be found in `Losses.Py`. 

- Discriminator Loss => `nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')`
- L1loss => `nn.L1Loss(size_average=None, reduce=None, reduction='mean')`
- L2loss => `nn.MSELoss(size_average=None, reduce=None, reduction='mean')`
- Adversarial loss (ADVloss) => takes as input probability of discriminator that input image is a Superresolution image so $1-ouput$, and then again uses `nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')`
- VGG19_Loss => Still needs some pre-processing before it can be run through the VGG19 model but uses  `nn.MSELoss(size_average=None, reduce=None, reduction='mean')` in the end. 

The training procedure closely follows the paper representation as best as possible, but instead of running 1000 iterations per epoch, which equates to 16k samples even though the dataset is only 9.6k, the model is trained for 600 iterations per epoch and the number of epochs to train is lengthened to keep the number of total iterations the same. Thus the model is trained for 166 epochs in phase 1 and 250 epochs in phase 2 using the minibatch size of 16. 

The training loop follows the following structure:

```
for each phase in phases:
	Initialize generator optimizer
	Initialize discriminator optimizer
	Check to skip phase 1 //(if we use pretrained generator weights from previous training)
	Check to skip phase 2 //(if we use pretrained generator and discriminator weights from previous training)
	If phase = 2 : Save Model weights of trained generator and compute PSNR on validation and test set
	for each epoch:
		
		for each batch in dataset:
            Prepare data
            Reset parameter gradients

            Forward pass generator to generate Super Resolution Image
            Calculate GLoss = L1 Loss
            Calculate L2 and PSNR loss training samples

            if phase = 2:
                Calculate p(HR) using forward pass discriminator
                Calculate DLoss

                Calculate P(SR) = 1 - p(HR) to compute ADVloss
                Calculate VGG19_loss

                Perform backward and optimizer step Discriminator

                Add ADVloss and VGG19_Loss to GLoss => g_loss += alpha * VGG19_Loss + beta (ADVloss)

       	Perform backward and optimizer step Generator
       	Keep track of loss function
       	
Calculate PSNR on validation and test set (for generation of figure 3)
Save weights of trained generator and discriminator

```

We chose to use `tqdm` to have more insight in the training procedure by giving an estimated time but also showing the current loss values for that batch or epoch. 

```python
from tqdm import tqdm

inner = tqdm(total=len(loader), desc='Batch', position=1, leave=False)
for i_batch, sample_batch in enumerate(loader):
	# Perform forward, backpass etc
	inner.update(1)
	inner.inner.set_postfix(g_loss=g_loss.item(), loss_l2=l2_Loss.item())
```

It is important during phase 2 to detach the generated Super Resolution image otherwise it will be added to the computational graph of the Discriminator and be cleared when the backward pass is performed! The same holds for calculating p(SR) = 1 - p(HR) as p(SR) is used for the adversarial loss. 



## Results



## Discussion













