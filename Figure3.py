from Load_dataset import RockDataset
import matplotlib.pyplot as plt
import torchvision.transforms as T
from SRCNN_Loss import PSNR, L2loss
from Bicubic import bicubic_interpolation
to_Tensor = T.ToTensor()

## Evaluate PSNR on validation and test set on trained model
# Retrieve validation and test set (both in valid and test set for each rock type there are 400 images)
sandstone_4_valid = RockDataset("DeepRockSR-2D/sandstone2D/sandstone2D_valid_LR_default_X4", "DeepRockSR-2D/sandstone2D/sandstone2D_valid_HR")
sandstone_4_test = RockDataset("DeepRockSR-2D/sandstone2D/sandstone2D_test_LR_default_X4", "DeepRockSR-2D/sandstone2D/sandstone2D_test_HR")
carbonate_4_valid = RockDataset("DeepRockSR-2D/carbonate2D/carbonate2D_valid_LR_default_X4", "DeepRockSR-2D/carbonate2D/carbonate2D_valid_HR")
carbonate_4_test = RockDataset("DeepRockSR-2D/carbonate2D/carbonate2D_test_LR_default_X4", "DeepRockSR-2D/carbonate2D/carbonate2D_test_HR")
coal_4_valid = RockDataset("DeepRockSR-2D/coal2D/coal2D_valid_LR_default_X4", "DeepRockSR-2D/coal2D/coal2D_valid_HR")
coal_4_test = RockDataset("DeepRockSR-2D/coal2D/coal2D_test_LR_default_X4", "DeepRockSR-2D/coal2D/coal2D_test_HR")

# dit is niet zo'n hele nette manier maar ik weet ff geen andere methode om door alle validate en test sets te kunnen loopen
valid_test_dict = {"sandstone_v": sandstone_4_valid, "sandstone_t": sandstone_4_test, "carbonate_v": carbonate_4_valid, "carbonate_t": carbonate_4_test, "coal_v": coal_4_valid, "coal_t": coal_4_test}

# empty dictionaries which will be filled with psnr values for validation and test images combined
sandstone_psnr = {"bicubic": [], "SRCNN": [24, 25, 26], "SRGAN": [24, 25, 26]}
carbonate_psnr = {"bicubic": [], "SRCNN": [24, 25, 26], "SRGAN": [24, 25, 26]}
coal_psnr = {"bicubic": [], "SRCNN": [24, 25, 26], "SRGAN": [24, 25, 26]}

for key in valid_test_dict:
    dataset = valid_test_dict[key]

    for i, image in enumerate(dataset):
        image_LR = image["LR"]
        image_LR = to_Tensor(image_LR)
        image_HR = image["HR"]
        image_HR = to_Tensor(image_HR)
        image_SR_bicupic = bicubic_interpolation(image_LR, 4)
        image_SR_SRCNN = [] # deze moeten door het model gaan om de SR images te verkrijgen 
        image_SR_SRGAN = []

        images_SR = {"bicubic": image_SR_bicupic, "SRCNN": image_SR_SRGAN, "SRGAN": image_SR_SRGAN}
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
fig, axs = plt.subplots(1,3, figsize=(15,5))
axs[0].boxplot(sandstone_psnr.values())
axs[0].set_xticklabels(sandstone_psnr.keys())
axs[0].set(ylabel="PSNR (dB)")
axs[0].set_title("Sandstone Validation + Testing Images")
axs[1].boxplot(carbonate_psnr.values())
axs[1].set_xticklabels(carbonate_psnr.keys())
axs[1].set(ylabel="PSNR (dB)")
axs[1].set_title("Carbonate Validation + Testing Images")
axs[2].boxplot(coal_psnr.values())
axs[2].set_xticklabels(coal_psnr.keys())
axs[2].set(ylabel="PSNR (dB)")
axs[2].set_title("Coal Validation + Testing Images")

fig.show()
plt.show(block=False)
input('press <ENTER> to continue') # ik weet niet anders krijg ik de plot niet te zien 



    