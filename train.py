import random

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from Discriminator import Discriminator
from Load_dataset import load_dataset
from SRCNN import SRCNN

m_seed = 1  # or use random.randint(1, 10000) for random reed
random.seed(m_seed)
torch.manual_seed(m_seed)


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
    gen.load_state_dict(torch.load(gen_weight_path))
    disc.load_state_dict(torch.load(disc_weight_path))


def train(gen, disc, vgg, device):
    """
    Contains all the code for the training procedure of SRGAN
    :param gen: Model of the Generator (SRCNN)
    :param disc: Model of the Discriminator
    :param vgg: VGG19
    :param device: Run on the CPU or GPU(CUDA)
    :return:
    """

    # Specify hyperparameters Generator (SRCNN)
    epochs_gen = 100
    nr_of_iterations = 100
    mini_batch_size = 16  # Size 192x192
    lr_generator = 1e-4
    alpha = 1e-5
    beta = 5e-3
    optimizer_gen = torch.optim.Adam(gen.parameters(), lr_generator)

    real_label = 1
    fake_label = 0

    # TODO: Should be replaced by G_Loss
    criterion_gen = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

    # TODO: Should be replaced by D_Loss
    criterion_disc = torch.nn.BCELoss()

    # Specify hyperparameters Discriminator
    lr_disc = 1e-4
    optimizer_disc = torch.optim.Adam(disc.parameters(), lr_disc)
    epoch_both = 150

    # Load Dataset
    train, valid_carbonate, valid_coal, valid_sand, test = load_dataset()

    # Configure Data Loaders
    mini_batch_size_test = 8  # These are higher resolution images and thus might not fit into batches of size 16!
    mini_batch_size_valid = 8
    rd_loader_train = DataLoader(train, batch_size=mini_batch_size, shuffle=True, num_workers=8, pin_memory=True,
                                 drop_last=True)  # Number of workers needs some figgeting to increase speedup
    rd_loader_valid_carbo = DataLoader(valid_carbonate, batch_size=mini_batch_size_valid, shuffle=True, num_workers=0)
    rd_loader_valid_coal = DataLoader(valid_coal, batch_size=mini_batch_size_valid, shuffle=True, num_workers=0)
    rd_loader_valid_sand = DataLoader(valid_sand, batch_size=mini_batch_size_valid, shuffle=True, num_workers=0)
    rd_loader_test = DataLoader(test, batch_size=mini_batch_size_test, shuffle=True, num_workers=0)

    # Keep track of the Generator loss and Discriminator loss during epochs
    loss_generator_train = []
    loss_discriminator_train = []

    # Load label used later for training discriminator
    label_real = torch.full((mini_batch_size, 1), real_label, dtype=torch.float32, device=device)
    label_fake = torch.full((mini_batch_size, 1), fake_label, dtype=torch.float32, device=device)
    label = torch.cat((label_real, label_fake))

    # Training of SRCNN (Generator)
    for phase, epochs in enumerate([epochs_gen, epoch_both]):
        # When done with training phase 1 save the weights of the Generator
        if phase == 1:
            torch.save(gen.state_dict(), 'model_weights_gen.pth')

        outer = tqdm(range(epochs), position=0, desc='Epoch', leave=True)
        for epoch in outer:

            loss_gen_epoch = 0
            loss_disc_epoch = 0

            # Specify Inner progressbar which keeps track of training inside epoch
            inner = tqdm(total=nr_of_iterations, desc='Batch', position=1, leave=False)

            # Necessary because batch_size * len(data_loader) < nr_iterations
            # So we loop over the data another time
            iteration = 0
            stop = False
            while not stop:
                for i_batch, sample_batch in enumerate(rd_loader_train):

                    gen.train()
                    input_LR = sample_batch["LR"].to(device)
                    target_HR = sample_batch["HR"].to(device)

                    # Stop when number of iterations has been reached
                    if iteration >= nr_of_iterations:
                        stop = True
                        break

                    # Zero the parameter gradients
                    optimizer_gen.zero_grad()
                    optimizer_disc.zero_grad()

                    # Generate Super Resolution Image
                    with autocast():
                        SR_image = gen(input_LR)

                    # Calculate loss
                    loss_gen = criterion_gen(SR_image, target_HR)

                    # If we are in the second training phase we also need to train discriminator
                    if phase == 1:
                        disc.train()
                        disc_input = torch.cat((target_HR.detach(), SR_image.detach()))
                        output_disc = disc(disc_input)
                        # Calculate loss
                        loss_disc = criterion_disc(output_disc, label)

                        # Backward and optimizer step
                        loss_disc.backward()
                        optimizer_disc.step()

                        # Keep track of the loss value
                        loss_disc_epoch += loss_disc

                    # Backward step generator
                    loss_gen.backward()
                    optimizer_gen.step()
                    # Keep track of average Loss
                    loss_gen_epoch += loss_gen

                    # Update progressbar and iteration var
                    iteration += 1
                    inner.update(1)
                    inner.set_postfix(loss=loss_gen.item())

            # Keep track of generator loss and update progressbar
            loss_avg_gen = loss_gen_epoch.item() / 1000
            loss_generator_train.append(loss_avg_gen)
            if phase == 0:
                outer.set_postfix(loss=loss_avg_gen)
            else:
                loss_disc_avg = loss_disc_epoch.item()/1000
                loss_discriminator_train.append(loss_disc_avg)
                outer.set_postfix(loss_gen=loss_avg_gen, loss_disc=loss_disc_avg)

            # with torch.no_grad():  # Since we are evaluating no gradients need to calculated -> speedup
            #     gen.eval()
            #     disc.eval()
            #     inner = tqdm(total=3 * len(rd_loader_valid_coal), desc='Validation', position=1, leave=False)
            #     psnr_val = torch.zeros((3, len(rd_loader_valid_coal)))
            #     for index, valid_loader in enumerate(
            #             [rd_loader_valid_carbo, rd_loader_valid_coal, rd_loader_valid_sand]):
            #
            #         for i_batch, sample_batch in enumerate(valid_loader):
            #             input_LR = sample_batch["LR"][0].to(device)
            #             output_HR = gen(input_LR[None,])
            #             toPil = T.ToPILImage()
            #             plt.imshow(toPil(output_HR[0]), cmap="gray")
            #             plt.show()
            #             break

    # with torch.no_grad():  # No gradient calculation needed
    #     inner = tqdm(total=len(rd_loader_test), desc='Testing', position=1, leave=False)
    #     for t_batch in rd_loader_test:
    #         # Any sort of testing should take place here
    #         break

    # Save the Model weights of both Generator and Discriminator
    torch.save(gen.state_dict(), 'model_weights_gen_2.pth')
    torch.save(disc.state_dict(), 'model_weights_gen_2.pth')


if __name__ == '__main__':
    device = try_gpu()
    gen = SRCNN(1)
    disc = Discriminator(1)
    gen.to(device)
    disc.to(device)

    # # VGG uses Coloured images originally so need to duplicate channels or something?
    # vgg_original = vgg19(pretrained=True)
    # vgg_cut = vgg_original.features[:-1] # Use all Layers before fully connected layer and before max pool layer
    # vgg_cut.to(device)

    train(gen, disc, None, device)
