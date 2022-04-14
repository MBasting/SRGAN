# SRGAN Reproducibility Project
This repository contains a pytorch implementation of the paper 'Boosting Resolution and Recovering Texture of 2D and 3D
Micro-CT Images with Deep Learning' that can be found [here](https://agupubs-onlinelibrary-wiley-com.tudelft.idm.oclc.org/doi/full/10.1029/2019WR026052). 

## Downloading Dataset
Can be done from https://www.digitalrocksportal.org/projects/215/images/159816/download/
## Configure Environment
Run `conda create --name SRGAN python=3.9` to create a conda environment called SRGAN. 
Now the necessary requirements can be installed using `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch` for installation on GPU or `conda install pytorch torchvision torchaudio cpuonly -c pytorch
` for CPU, and the other dependencies using `conda install  tqdm matplotlib scikit-image`.

Before running the code the results.zip and weights.zip need to be extracted. These files contain the pre-trained weights of a run and resulting psnr values of that run. **If pre-trained weights are not necessary, a results and weights folder need to be created** .