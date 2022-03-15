import glob
import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision import utils


class RockDataset(Dataset):
    """ DeepRockSR-2D dataset."""

    def __init__(self, dir_LR, dir_HR, transform=None):
        self.dir_LR = dir_LR
        self.dir_HR = dir_HR

        self.image_list_LR = [name for name in os.listdir(self.dir_LR) if
                              os.path.isfile(os.path.join(self.dir_LR, name))]
        self.image_list_HR = [name.replace("x4", "") for name in self.image_list_LR]

        self.transform = transform

    def __len__(self):
        return len(self.image_list_LR)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_LR = os.path.join(self.dir_LR, self.image_list_LR[idx])
        image_LR = io.imread(img_name_LR)

        img_name_HR = os.path.join(self.dir_HR, self.image_list_HR[idx])
        image_HR = io.imread(img_name_HR)

        if self.transform:
            image_LR = self.transform(image_LR)
            image_HR = self.transform(image_HR)

        return {"LR": image_LR, "HR": image_HR}


def show_rock_samples(sample_batch, nr_to_show=4):
    image_LR_batch = sample_batch['LR']
    image_HR_batch = sample_batch['HR']

    for image_batch in [image_LR_batch, image_HR_batch]:
        img_to_show = (image_batch[:nr_to_show]).permute(0, 3, 1, 2)

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
    rock_s_4_train = RockDataset("DeepRockSR-2D/Train/s_train_LR_u_X4", "DeepRockSR-2D/Train/s_train_HR")
    rock_s_4_test = RockDataset("DeepRockSR-2D/Test/s_test_LR_u_X4", "DeepRockSR-2D/Test/s_test_HR")
    rock_s_4_valid = RockDataset("DeepRockSR-2D/Valid/s_valid_LR_u_X4", "DeepRockSR-2D/Valid/s_valid_HR")

    rock_data_loader = DataLoader(rock_s_4_train, batch_size=4, shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(rock_data_loader):
        if i_batch == 0:
            show_rock_samples(sample_batched)
        else:
            break


if __name__ == '__main__':
    load_dataset()
