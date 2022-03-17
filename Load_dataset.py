import glob
import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision import utils
import torchvision.transforms as T



class RockDataset(Dataset):
    """ DeepRockSR-2D dataset."""

    def __init__(self, dir_LR, dir_HR, crop=None):
        self.dir_LR = dir_LR
        self.dir_HR = dir_HR

        self.image_list_LR = [name for name in os.listdir(self.dir_LR) if
                              os.path.isfile(os.path.join(self.dir_LR, name))]
        self.image_list_HR = [name.replace("x4", "") for name in self.image_list_LR]

        self.crop = crop

    def __len__(self):
        return len(self.image_list_LR)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_LR = os.path.join(self.dir_LR, self.image_list_LR[idx])
        image_LR = io.imread(img_name_LR)


        img_name_HR = os.path.join(self.dir_HR, self.image_list_HR[idx])
        image_HR = io.imread(img_name_HR)

        if self.crop:

            # Transform operations
            toPil = T.ToPILImage()
            t_random_crop = T.RandomCrop(48)
            to_Tensor = T.ToTensor()

            # To apply randomcropping tensor needs to be converted to PIL
            image_LR = toPil(image_LR)
            image_HR = toPil(image_HR)

            # Get random location parameters for Low Res image
            params = t_random_crop.get_params(image_LR, (48, 48))

            # Get same locations in the High res image
            params_HR = list([i * 4 for i in params])

            # Apply cropping
            image_LR = T.functional.crop(image_LR, *params)
            image_HR = T.functional.crop(image_HR, *params_HR)

            # Convert back to tensor
            image_LR = to_Tensor(image_LR)
            image_HR = to_Tensor(image_HR)

        return {"LR": image_LR, "HR": image_HR}


def show_rock_samples(sample_batch, nr_to_show=4):
    image_LR_batch = sample_batch['LR']
    image_HR_batch = sample_batch['HR']

    for image_batch in [image_LR_batch, image_HR_batch]:
        img_to_show = (image_batch[:nr_to_show])

        grid_img_LR = utils.make_grid(img_to_show)
        plt.imshow(grid_img_LR.permute(1, 2, 0))
        plt.show()


def show_image(dataset, i):
    sample = dataset[i]

    plt.imshow(sample["LR"])
    plt.show()

    plt.imshow(sample["HR"])
    plt.show()


def load_dataset():

    rock_s_4_train = RockDataset("DeepRockSR-2D/shuffled2D/shuffled2D_train_LR_unknown_X4", "DeepRockSR-2D/shuffled2D/shuffled2D_train_HR", True)
    rock_s_4_valid = RockDataset("DeepRockSR-2D/shuffled2D/shuffled2D_valid_LR_unknown_X4", "DeepRockSR-2D/shuffled2D/shuffled2D_valid_HR")
    rock_s_4_test = RockDataset("DeepRockSR-2D/shuffled2D/shuffled2D_test_LR_unknown_X4", "DeepRockSR-2D/shuffled2D/shuffled2D_test_HR")

    return rock_s_4_train, rock_s_4_valid, rock_s_4_test

