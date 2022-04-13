SRGAN

# Downloading Dataset
Can be done from https://www.digitalrocksportal.org/projects/215/images/159816/download/
# Configure Environment
Run `conda create --name SRGAN python=3.9` to create a conda environment called SRGAN. 
Now the necessary requirements can be installed using `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch` for installation on GPU or `conda install pytorch torchvision torchaudio cpuonly -c pytorch
` for CPU, and the other dependencies using `conda install  tqdm matplotlib scikit-image`.