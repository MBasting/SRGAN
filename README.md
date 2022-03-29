SRGAN

# Downloading Dataset
Can be done from https://www.digitalrocksportal.org/projects/215/images/159816/download/
# Configure Environment Linux
Run `conda create --name SRGAN --file SRGAN_env.txt` to create a conda environment which downloads the necessary dependencies. 

Before running `python train.py` don't forget to activate the environment using `conda activate SRGAN`

# Configure Environment Windows
Run `conda create --name SRGAN python=3.9` to create a conda environment called SRGAN. Now the necessary requirements can be installed using `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch tqdm matplotlib scikit-image`