import matplotlib.pyplot as plt

from Load_dataset import RockDataset
import torchvision.transforms as T
from Bicubic import bicubic_interpolation
from SRCNN import SRCNN
from train import load_weights

to_Tensor = T.ToTensor()
toPil = T.ToPILImage()

# get data, only from validation set 
sandstone_4_valid = RockDataset("DeepRockSR-2D/sandstone2D/sandstone2D_train_LR_default_X4",
                                "DeepRockSR-2D/sandstone2D/sandstone2D_train_HR")
carbonate_4_valid = RockDataset("DeepRockSR-2D/carbonate2D/carbonate2D_train_LR_default_X4",
                                "DeepRockSR-2D/carbonate2D/carbonate2D_train_HR")
coal_4_valid = RockDataset("DeepRockSR-2D/coal2D/coal2D_valid_LR_default_X4", "DeepRockSR-2D/coal2D/coal2D_valid_HR")

gen_1 = SRCNN(1)
gen_2 = SRCNN(1)

load_weights(gen_1, None, "weights/model_weights_gen_1648742795.0276928.pth", None)
load_weights(gen_2, None, "weights/model_weights_gen_2_1648763675.3885765.pth", None)
gen_1.eval()
gen_2.eval()

valid_dict = {"sandstone": sandstone_4_valid, "carbonate": carbonate_4_valid, "coal": coal_4_valid}

fig, axs = plt.subplots(3, 5, figsize=(10, 5))
fig2, axs2 = plt.subplots(3, 3, figsize=(10, 9))
cm1 = "Greys_r"
cm = "Greys"
for index, key in enumerate(valid_dict):
    # Load the correct Sample also used in Figure 2 of the paper
    if key == "carbonate":
        sample_batch = carbonate_4_valid[5]
    elif key == "coal":
        sample_batch = coal_4_valid[3484 - 3201]
    else:
        sample_batch = sandstone_4_valid[1153]

    # Loading of images
    image_LR = sample_batch["LR"][0]
    image_HR = sample_batch["HR"]
    image_LR_2 = image_LR.view(1, 1, image_LR.shape[0], image_LR.shape[1]) # Map to Batch representation
    image_SR_bicupic = bicubic_interpolation(image_LR_2, 4)

    # Generate SRCNN and SRGAN images and map to batch representation
    image_SR_SRCNN = gen_1(image_LR_2).detach().view(1, image_HR.size()[-1], image_HR.size()[-1])
    image_SR_SRGAN = gen_2(image_LR_2).detach().view(1, image_HR.size()[-1], image_HR.size()[-1])

    axs[index, 0].imshow(toPil(image_LR), cmap=cm1)
    axs[index, 0].set(ylabel=key)
    axs[index, 1].imshow(toPil(image_HR), cmap=cm1)
    axs[index, 2].imshow(toPil(image_SR_bicupic), cmap=cm1)
    axs[index, 3].imshow(toPil(image_SR_SRCNN), cmap=cm1)
    axs[index, 4].imshow(toPil(image_SR_SRGAN), cmap=cm1)

    if key == "carbonate" or index == 0:
        axs[0, 0].set_title("LR")
        axs[0, 1].set_title("HR")
        axs[0, 2].set_title("BC")
        axs[0, 3].set_title("SRCNN")
        axs[0, 4].set_title("SRGAN")

    sp = axs2[index, 0].imshow(abs((image_SR_bicupic - image_HR).view(500, 500)) * 255, cmap=cm)
    axs2[index, 0].set_title("BC minus HR")
    axs2[index, 0].set(ylabel=key)
    fig2.colorbar(sp, ax=axs2[index, 0])
    sp = axs2[index, 1].imshow(abs((image_SR_SRCNN - image_HR).view(500, 500)) * 255, cmap=cm)
    axs2[index, 1].set_title("CNN minus HR")
    fig2.colorbar(sp, ax=axs2[index, 1])
    sp = axs2[index, 2].imshow(abs((image_SR_SRGAN - image_HR).view(500, 500)) * 255, cmap=cm)
    axs2[index, 2].set_title("SRGAN minus HR")
    fig2.colorbar(sp, ax=axs2[index, 2])

plt.show()
