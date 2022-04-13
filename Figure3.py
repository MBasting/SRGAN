import json

from Load_dataset import RockDataset
import matplotlib.pyplot as plt
import torchvision.transforms as T
from Bicubic import bicubic_interpolation
from Losses import L2loss, PSNR

to_Tensor = T.ToTensor()

## Evaluate PSNR on validation and test set on trained model
# Retrieve validation and test set (both in valid and test set for each rock type there are 400 images)
sandstone_4_valid = RockDataset("DeepRockSR-2D/sandstone2D/sandstone2D_valid_LR_default_X4",
                                "DeepRockSR-2D/sandstone2D/sandstone2D_valid_HR")
sandstone_4_test = RockDataset("DeepRockSR-2D/sandstone2D/sandstone2D_test_LR_default_X4",
                               "DeepRockSR-2D/sandstone2D/sandstone2D_test_HR")
carbonate_4_valid = RockDataset("DeepRockSR-2D/carbonate2D/carbonate2D_valid_LR_default_X4",
                                "DeepRockSR-2D/carbonate2D/carbonate2D_valid_HR")
carbonate_4_test = RockDataset("DeepRockSR-2D/carbonate2D/carbonate2D_test_LR_default_X4",
                               "DeepRockSR-2D/carbonate2D/carbonate2D_test_HR")
coal_4_valid = RockDataset("DeepRockSR-2D/coal2D/coal2D_valid_LR_default_X4", "DeepRockSR-2D/coal2D/coal2D_valid_HR")
coal_4_test = RockDataset("DeepRockSR-2D/coal2D/coal2D_test_LR_default_X4", "DeepRockSR-2D/coal2D/coal2D_test_HR")

# dit is niet zo'n hele nette manier maar ik weet ff geen andere methode om door alle validate en test sets te kunnen loopen
valid_test_dict = {"sandstone_v": sandstone_4_valid, "sandstone_t": sandstone_4_test, "carbonate_v": carbonate_4_valid,
                   "carbonate_t": carbonate_4_test, "coal_v": coal_4_valid, "coal_t": coal_4_test}

# empty dictionaries which will be filled with psnr values for validation and test images combined
sandstone_psnr = {"bicubic": [], "SRCNN": [], "SRGAN": []}
carbonate_psnr = {"bicubic": [], "SRCNN": [], "SRGAN": []}
coal_psnr = {"bicubic": [], "SRCNN": [], "SRGAN": []}

psnr_dict_SRCNN = {}
psnr_dict_SRGAN = {}
# Opening JSON file
with open('results/psnr_0_1648742813.0668075.json') as json_file:
    psnr_dict_SRCNN = json.load(json_file)

with open('results/psnr_1_1648763675.3855865.json') as json_file:
    psnr_dict_SRGAN = json.load(json_file)

sandstone_psnr["SRCNN"] = psnr_dict_SRCNN["sandstone"]
carbonate_psnr["SRCNN"] = psnr_dict_SRCNN["carbonate"]
coal_psnr["SRCNN"] = psnr_dict_SRCNN["coal"]

sandstone_psnr["SRGAN"] = psnr_dict_SRGAN["sandstone"]
carbonate_psnr["SRGAN"] = psnr_dict_SRGAN["carbonate"]
coal_psnr["SRGAN"] = psnr_dict_SRGAN["coal"]

for key in valid_test_dict:
    dataset = valid_test_dict[key]

    for i, image in enumerate(dataset):
        image_LR = image["LR"]
        image_HR = image["HR"]
        image_SR_bicupic = bicubic_interpolation(image_LR.view(1, 1, image_LR.shape[1], image_LR.shape[2]), 4)

        images_SR = {"bicubic": image_SR_bicupic}

        for image_SR_type in images_SR:
            image_SR = images_SR[image_SR_type]

            L2loss_value = L2loss(image_SR, image_HR)
            PSNR_value = PSNR(L2loss_value)

            if key == "sandstone_v" or key == "sandstone_t":
                sandstone_psnr[image_SR_type].append(PSNR_value)
            if key == "carbonate_v" or key == "carbonate_t":
                carbonate_psnr[image_SR_type].append(PSNR_value)
            if key == "coal_v" or key == "coal_t":
                coal_psnr[image_SR_type].append(PSNR_value)

# create boxplots (Figure 3 in paper)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].boxplot(sandstone_psnr.values(), showfliers=False)
axs[0].set_xticklabels(sandstone_psnr.keys())
axs[0].set(ylabel="PSNR (dB)")
axs[0].set_title("Sandstone Validation + Testing Images")
axs[1].boxplot(carbonate_psnr.values(), showfliers=False)
axs[1].set_xticklabels(carbonate_psnr.keys())
axs[1].set(ylabel="PSNR (dB)")
axs[1].set_title("Carbonate Validation + Testing Images")
axs[2].boxplot(coal_psnr.values(), showfliers=False)
axs[2].set_xticklabels(coal_psnr.keys())
axs[2].set(ylabel="PSNR (dB)")
axs[2].set_title("Coal Validation + Testing Images")

fig.show()
plt.show()
# input('press <ENTER> to continue') # ik weet niet anders krijg ik de plot niet te zien
