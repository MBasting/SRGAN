import json
import random
import time

import torch
from torch.utils.data import DataLoader
from torchvision.models import vgg19
from tqdm import tqdm

from Discriminator import Discriminator
from Load_dataset import load_dataset
from Losses import DLoss, ADVloss, VGG19_Loss, L2loss, PSNR
from SRCNN import SRCNN


def try_gpu():
    """
    If GPU is available, return torch.device as cuda:0; else return torch.device
    as cpu.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


def load_weights(gen, disc, gen_weight_path, disc_weight_path):
    """
    Loads the saved trained weights in the respective models
    :param gen: Generator (SRCNN)
    :param disc: Discriminator
    :param gen_weight_path: Path to the weights of the Generator
    :param disc_weight_path: Path to the weights of the Discriminator
    :return:
    """
    if gen_weight_path is not None:
        gen.load_state_dict(torch.load(gen_weight_path)) # add: map_location='cpu' if you're working from a macbook
    if disc_weight_path is not None:
        disc.load_state_dict(torch.load(disc_weight_path)) # add: map_location='cpu' if you're working from a macbook


def calculate_psnr(gen, rd_loader_valid_carbo, rd_loader_valid_coal, rd_loader_valid_sand, rock_s_4_test_carbonate,
                   rock_s_4_test_coal, rock_s_4_test_sandstone, phase):
    with torch.no_grad():  # Since we are evaluating no gradients need to calculated -> speedup
        gen.eval()
        disc.eval()
        inner = tqdm(total=3 * len(rd_loader_valid_coal) + 3 * len(rock_s_4_test_coal), desc='Validation', position=1,
                     leave=False)
        loaders = ["carbonate", "coal", "sandstone", "carbonate_test", "coal_test", "sandstone_test"]
        psnr_total = {"carbonate": [], "coal": [], "sandstone": []}
        for index, valid_loader in enumerate(
                [rd_loader_valid_carbo, rd_loader_valid_coal, rd_loader_valid_sand, rock_s_4_test_carbonate,
                 rock_s_4_test_coal, rock_s_4_test_sandstone]):
            psnr_loader = []

            for i_batch, sample_batch in enumerate(valid_loader):
                input_LR = sample_batch["LR"].to(device)
                target_HR = sample_batch["HR"].to(device)

                SR_image = gen(input_LR)
                l2_Loss = L2loss(SR_image, target_HR)
                psnr_single = PSNR(l2_Loss, 2)
                psnr_loader.append(psnr_single)
                inner.update(1)

            loader = loaders[index % 3]
            psnr_total[loader].extend(psnr_loader)
        with open('results/psnr_{}_{}.json'.format(phase, time.time()), 'w') as fp:
            json.dump(psnr_total, fp, indent=4)


# Added label_smoothing option and separate SR and HR batches on suggestion from https://github.com/soumith/ganhacks
# More options were available but these were the simplest ones
# Added since phase 1 training leads to mode collapse Discriminator!
def train(gen, disc, vgg, device, load_from_file=False, weights_path_gen=None, weights_path_disc=None):
    """
    Contains all the code for the training procedure of SRGAN
    :param gen: Model of the Generator (SRCNN)
    :param disc: Model of the Discriminator
    :param vgg: VGG19
    :param device: Run on the CPU or GPU(CUDA)
    :param load_from_file: Load pre-trained weights or not
    :param weights_path_gen: Path to pre-trained generator weights
    :param weights_path_disc: Path to pre-trained discriminator weights
    :return:
    """

    # Specify hyperparameters Generator (SRCNN)
    epochs_gen = 166  # Since now we are not doing 1000 but 600 iterations per epoch
    lr_generator = 1e-4
    alpha = 1e-5
    beta = 5e-3

    L1_loss = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
    criterion_disc = DLoss

    # Specify hyperparameters Discriminator
    lr_disc = 1e-4
    fake_label = 0
    real_label = 1
    epoch_both = 250  # Since we are not doing 1000 but 600 iterations per epoch

    # Load Dataset
    train, valid_carbonate, valid_coal, valid_sand, test_carbonate, test_coal, test_sand = load_dataset()

    # Configure Data Loaders
    mini_batch_size = 16  # Size 192x192
    mini_batch_size_valid = 1
    mini_batch_size_test = 1
    rd_loader_train = DataLoader(train, batch_size=mini_batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                 drop_last=True)  # Number of workers needs some figgeting to increase speedup
    rd_loader_valid_carbo = DataLoader(valid_carbonate, batch_size=mini_batch_size_valid, shuffle=True, num_workers=0)
    rd_loader_valid_coal = DataLoader(valid_coal, batch_size=mini_batch_size_valid, shuffle=True, num_workers=0)
    rd_loader_valid_sand = DataLoader(valid_sand, batch_size=mini_batch_size_valid, shuffle=True, num_workers=0)

    rd_loader_test_carbo = DataLoader(test_carbonate, batch_size=mini_batch_size_test, shuffle=True, num_workers=0)
    rd_loader_test_coal = DataLoader(test_coal, batch_size=mini_batch_size_test, shuffle=True, num_workers=0)
    rd_loader_test_sand = DataLoader(test_sand, batch_size=mini_batch_size_test, shuffle=True, num_workers=0)

    # Keep track of the Generator loss and Discriminator loss during epochs
    loss_generator_train = []
    loss_discriminator_train = []
    psnr_values = []

    if load_from_file:
        load_weights(gen, disc, weights_path_gen, weights_path_disc)

    # Training of SRCNN (Generator)
    for phase, epochs in enumerate([epochs_gen, epoch_both]):
        # Each phase reinitialize optimizer
        optimizer_disc = torch.optim.Adam(disc.parameters(), lr_disc)
        optimizer_gen = torch.optim.Adam(gen.parameters(), lr_generator)

        # Skip phase 1 if the load_from_file flag is activated and weight path of generator is available
        if phase == 0 and load_from_file and weights_path_gen is not None:
            print("SKIP Phase 1")
            continue
        # Skip phase 2 if the load_from_file flag is activated and wieght path of discriminator is available
        if load_from_file and weights_path_disc is not None:
            print("SKIP Phase 2")
            continue

        # TQDM allows us to keep track of the training
        outer = tqdm(range(epochs), position=0, desc='Epoch', leave=True)
        for epoch in outer:

            # Initialize average loss values
            loss_gen_epoch = 0
            loss_disc_epoch = 0
            psnr_epoch = 0

            # Specify Inner progressbar which keeps track of training inside epoch
            inner = tqdm(total=600, desc='Batch', position=1, leave=False)

            # Loop through dataset with DataLoader
            for i_batch, sample_batch in enumerate(rd_loader_train):

                # Specify generator and discriminator mode (other mode would be eval())
                gen.train()
                disc.train()

                # Transfer image to GPU
                input_LR = sample_batch["LR"].to(device)
                target_HR = sample_batch["HR"].to(device)

                # Zero the parameter gradients
                optimizer_gen.zero_grad()
                optimizer_disc.zero_grad()

                # Generate Super Resolution Image
                SR_image = gen(input_LR)

                # Calculate loss
                g_loss = L1_loss(SR_image, target_HR)
                l2_Loss = L2loss(SR_image, target_HR)
                # Calculate Training PSNR
                psnr_single = PSNR(l2_Loss, 2)

                # If we are in the second training phase we also need to train discriminator
                if phase == 1:
                    # Create classification Tensors
                    label_real = torch.full((mini_batch_size, 1), real_label, dtype=torch.float32, device=device)
                    label_fake = torch.full((mini_batch_size, 1), fake_label, dtype=torch.float32, device=device)

                    # Training Super Resolution Images, training of SR and HR separate
                    output_disc_SR = disc(SR_image.detach())  # Output discriminator (prob HR image)
                    # Calculate loss Discriminator
                    loss_disc_SR = criterion_disc(label_fake, output_disc_SR)

                    # Only need the discriminator output of the SR images and p(sr) which is 1 - p(hr)
                    p_sr_fake = torch.ones(mini_batch_size, device=device) - output_disc_SR.detach()

                    # Calculate loss Generator
                    adv_loss = ADVloss(p_sr_fake, device=device)
                    vgg_loss = VGG19_Loss(SR_image, target_HR, vgg)

                    # Backward and optimizer step
                    loss_disc_SR.backward()
                    optimizer_disc.step()

                    # Training on High Resolution Images
                    optimizer_disc.zero_grad()

                    output_disc_HR = disc(target_HR.detach())
                    loss_disc_HR = criterion_disc(label_real, output_disc_HR)

                    # Backward and optimizer step
                    loss_disc_HR.backward()
                    optimizer_disc.step()

                    loss_disc = (loss_disc_HR + loss_disc_SR)
                    # Keep track of the loss value
                    loss_disc_epoch += loss_disc

                    # Update loss calculation
                    g_loss += alpha * vgg_loss + beta * adv_loss

                # Backward step generator
                g_loss.backward()
                optimizer_gen.step()
                # Keep track of average Loss
                loss_gen_epoch += g_loss

                # keep track of average psnr
                psnr_epoch += psnr_single

                # Update progressbar and iteration var
                inner.update(1)
                if phase == 0:
                    inner.set_postfix(loss=g_loss.item(), loss_l2=l2_Loss.item())
                else:
                    inner.set_postfix(loss=g_loss.item(), loss_l2=l2_Loss.item(), adv_loss=adv_loss.item(),
                                      vgg_loss=vgg_loss.item(), dloss=loss_disc.item())

            # Keep track of generator loss and update progressbar
            loss_avg_gen = loss_gen_epoch.item() / len(rd_loader_train)
            loss_generator_train.append(loss_avg_gen)

            # append average psnr loss
            psnr_avg = psnr_epoch / len(rd_loader_train)
            psnr_values.append(psnr_avg)

            if phase == 0:
                outer.set_postfix(loss=loss_avg_gen, psnr=psnr_avg)
            else:
                loss_disc_avg = loss_disc_epoch.item() / len(rd_loader_train)
                loss_discriminator_train.append(loss_disc_avg)
                outer.set_postfix(loss_gen=loss_avg_gen, loss_disc=loss_disc_avg, psnr=psnr_avg)
        # Save generator model weights (for both phases)
        torch.save(gen.state_dict(), 'weights/model_weights_gen_{}_{}.pth'.format(phase + 1, time.time()))
        # Save Discriminator (only for phase 2)
        if phase == 1:
            torch.save(disc.state_dict(), 'weights/model_weights_disc_{}_{}.pth'.format(phase + 1, time.time()))

            # Function to calculate the PSNR values
            calculate_psnr(gen, rd_loader_valid_carbo, rd_loader_valid_coal, rd_loader_valid_sand, rd_loader_test_carbo,
                           rd_loader_test_coal, rd_loader_test_sand, phase)


if __name__ == '__main__':
    m_seed = random.randint(1, 10000)
    random.seed(m_seed)
    torch.manual_seed(m_seed)
    device = try_gpu()
    gen = SRCNN(1)
    disc = Discriminator(1)

    gen.to(device)
    disc.to(device)

    # create the vgg19 network
    vgg_original = vgg19(pretrained=True)  # Load the pretrained network
    vgg_cut = vgg_original.features[:-1]  # Use all Layers before fully connected layer and before max pool layer
    vgg_cut.to(device)

    # Comment out if you want to only train from phase 2
    # train(gen, disc, vgg_cut, device, True, "weights/model_weights_gen_1648742795.0276928.pth", None)

    train(gen, disc, vgg_cut, device)


