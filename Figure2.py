import matplotlib.pyplot as plt
from Load_dataset import RockDataset
import torchvision.transforms as T
from Bicubic import bicubic_interpolation
to_Tensor = T.ToTensor()
toPil = T.ToPILImage()

# get data, only from validation set 
sandstone_4_valid = RockDataset("DeepRockSR-2D/sandstone2D/sandstone2D_valid_LR_default_X4", "DeepRockSR-2D/sandstone2D/sandstone2D_valid_HR")
carbonate_4_valid = RockDataset("DeepRockSR-2D/carbonate2D/carbonate2D_valid_LR_default_X4", "DeepRockSR-2D/carbonate2D/carbonate2D_valid_HR")
coal_4_valid = RockDataset("DeepRockSR-2D/coal2D/coal2D_valid_LR_default_X4", "DeepRockSR-2D/coal2D/coal2D_valid_HR")

valid_dict = {"sandstone": sandstone_4_valid, "carbonate": carbonate_4_valid, "coal": coal_4_valid}

fig, axs = plt.subplots(3,5, figsize=(5,3))
fig2, axs2 = plt.subplots(3,3, figsize=(5,5))
cm = "Greys"

for key in valid_dict:
    dataset = valid_dict[key]

    for i, image in enumerate(dataset):
        image_LR = image["LR"]
        image_LR = to_Tensor(image_LR)
        image_HR = image["HR"]
        image_HR = to_Tensor(image_HR)
        image_SR_bicupic = bicubic_interpolation(image_LR, 4)
        image_SR_SRCNN = [] # deze moeten door het model gaan om de SR images te verkrijgen 
        image_SR_SRGAN = []

        break # je hebt maar één image nodig uit de dataset, dus je stopt gelijk al eigenlijk 
    

    if key == "carbonate":

        axs[0,0].imshow(toPil(image_LR))
        axs[0,0].set_title("LR")
        axs[0,0].set(ylabel="carbonate")
        axs[0,1].imshow(toPil(image_HR))
        axs[0,1].set_title("HR")
        axs[0,2].imshow(toPil(image_SR_bicupic))
        axs[0,2].set_title("BC")
        axs[0,3].imshow(toPil(image_SR_bicupic))
        axs[0,3].set_title("BC")
        axs[0,4].imshow(toPil(image_SR_bicupic))
        axs[0,4].set_title("BC")


        print((image_SR_bicupic- image_HR).shape)

        sp = axs2[0,0].imshow(toPil(image_SR_bicupic[0] - image_HR[0]), cmap = cm)
        axs2[0,0].set_title("BC minus HR")
        axs2[0,0].set(ylabel="carbonate")
        fig2.colorbar(sp, ax=axs2[0,0])
        sp = axs2[0,1].imshow(toPil((image_SR_bicupic[0] - image_HR[0])), cmap = cm)
        axs2[0,1].set_title("BC minus HR")
        fig2.colorbar(sp, ax=axs2[0,1])
        sp = axs2[0,2].imshow(toPil(image_SR_bicupic[0] - image_HR[0]), cmap = cm)
        axs2[0,2].set_title("BC minus HR")
        fig2.colorbar(sp, ax=axs2[0,2])
    
    if key == "coal":

        axs[1,0].imshow(toPil(image_LR))
        axs[1,0].set(ylabel="coal")
        axs[1,1].imshow(toPil(image_HR))
        axs[1,2].imshow(toPil(image_SR_bicupic))
        axs[1,3].imshow(toPil(image_SR_bicupic))
        axs[1,4].imshow(toPil(image_SR_bicupic))

        sp = axs2[1,0].imshow(toPil(image_SR_bicupic[0] - image_HR[0]), cmap = cm)
        fig2.colorbar(sp, ax=axs2[1,0])
        axs2[1,0].set(ylabel="coal")
        sp = axs2[1,1].imshow(toPil(image_SR_bicupic[0] - image_HR[0]), cmap = cm)
        fig2.colorbar(sp, ax=axs2[1,1], )
        sp = axs2[1,2].imshow(toPil(image_SR_bicupic[0] - image_HR[0]), cmap = cm)
        fig2.colorbar(sp, ax=axs2[1,2])

    if key == "sandstone":

        axs[2,0].imshow(toPil(image_LR))
        axs[2,0].set(ylabel="sandstone")
        axs[2,1].imshow(toPil(image_HR))
        axs[2,2].imshow(toPil(image_SR_bicupic))
        axs[2,3].imshow(toPil(image_SR_bicupic))
        axs[2,4].imshow(toPil(image_SR_bicupic))

        sp = axs2[2,0].imshow(toPil(image_SR_bicupic[0] - image_HR[0]), cmap = cm)
        fig2.colorbar(sp, ax=axs2[2,0])
        axs2[2,0].set(ylabel="sandstone")
        sp = axs2[2,1].imshow(toPil(image_SR_bicupic[0] - image_HR[0]), cmap = cm)
        fig2.colorbar(sp, ax=axs2[2,1])
        sp = axs2[2,2].imshow(toPil(image_SR_bicupic[0] - image_HR[0]), cmap = cm)
        fig2.colorbar(sp, ax=axs2[2,2])

plt.show()

    




