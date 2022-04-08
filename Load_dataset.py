import glob
import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision import utils
import torchvision.transforms as T


# Mostly adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class RockDataset(Dataset):
    """ DeepRockSR-2D dataset."""

    def __init__(self, dir_LR, dir_HR, crop=None):
        # Save directories of LR and HR images
        self.dir_LR = dir_LR
        self.dir_HR = dir_HR

        # Get List of image for LR images
        self.image_list_LR = [name for name in os.listdir(self.dir_LR) if
                              os.path.isfile(os.path.join(self.dir_LR, name))]

        # Crop flag
        self.crop = crop

    def __len__(self):
        return len(self.image_list_LR)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        to_Tensor = T.ToTensor()
        toPil = T.ToPILImage()
        to_Gray = T.Grayscale(num_output_channels=1)

        image_LR_name = self.image_list_LR[idx]
        image_HR_name = image_LR_name.replace("x4", "")

        # Get the full path to the images and load
        img_name_LR = os.path.join(self.dir_LR, image_LR_name)
        img_name_HR = os.path.join(self.dir_HR, image_HR_name)

        image_LR = io.imread(img_name_LR)
        image_HR = io.imread(img_name_HR)

        # # Convert to grayScale
        image_LR = to_Gray(toPil(image_LR))
        image_HR = to_Gray(toPil(image_HR))


        if self.crop:

            # Transform operations
            t_random_crop = T.RandomCrop(48)

            # Get random location parameters for Low Res image
            params = t_random_crop.get_params(image_LR, (48, 48))

            # Get same locations in the High res image
            params_HR = list([i * 4 for i in params])

            # Apply cropping
            image_LR = T.functional.crop(image_LR, *params)
            image_HR = T.functional.crop(image_HR, *params_HR)

        image_LR = to_Tensor(image_LR)
        image_HR = to_Tensor(image_HR)
        return {"LR": image_LR, "HR": image_HR}


def show_rock_samples(sample_batch, nr_to_show=4):
    """

    :param sample_batch:
    :param nr_to_show:
    :return:
    """
    image_LR_batch = sample_batch['LR']
    image_HR_batch = sample_batch['HR']

    for image_batch in [image_LR_batch, image_HR_batch]:
        img_to_show = (image_batch[:nr_to_show])

        grid_img_LR = utils.make_grid(img_to_show)
        plt.imshow(grid_img_LR.permute(1, 2, 0))
        plt.show()


def show_image(dataset, i):
    sample = dataset[i]
    print(sample["LR"])

    plt.imshow(sample["LR"], cmap='gray')
    plt.show()

    plt.imshow(sample["HR"], cmap='gray')
    plt.show()




def load_dataset():
        rock_s_4_train = RockDataset("DeepRockSR-2D/shuffled2D/shuffled2D_train_LR_unknown_X4", "DeepRockSR-2D/shuffled2D/shuffled2D_train_HR", True)
        rock_s_4_valid_carbonate = RockDataset("DeepRockSR-2D/carbonate2D/carbonate2D_valid_LR_unknown_X4", "DeepRockSR-2D/carbonate2D/carbonate2D_valid_HR")
        rock_s_4_valid_coal = RockDataset("DeepRockSR-2D/coal2D/coal2D_valid_LR_unknown_X4", "DeepRockSR-2D/coal2D/coal2D_valid_HR")
        rock_s_4_valid_sandstone = RockDataset("DeepRockSR-2D/sandstone2D/sandstone2D_valid_LR_unknown_X4", "DeepRockSR-2D/sandstone2D/sandstone2D_valid_HR")
        rock_s_4_test_carbonate = RockDataset("DeepRockSR-2D/carbonate2D/carbonate2D_test_LR_unknown_X4",
                                               "DeepRockSR-2D/carbonate2D/carbonate2D_test_HR")
        rock_s_4_test_coal = RockDataset("DeepRockSR-2D/coal2D/coal2D_test_LR_unknown_X4",
                                          "DeepRockSR-2D/coal2D/coal2D_test_HR")
        rock_s_4_test_sandstone = RockDataset("DeepRockSR-2D/sandstone2D/sandstone2D_test_LR_unknown_X4",
                                               "DeepRockSR-2D/sandstone2D/sandstone2D_test_HR")

        return rock_s_4_train, rock_s_4_valid_carbonate, rock_s_4_valid_coal, rock_s_4_valid_sandstone, rock_s_4_test_carbonate, rock_s_4_test_coal, rock_s_4_test_sandstone
