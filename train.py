import torch
from torch.utils.data import DataLoader
from torch import nn
import tifffile as tf
import os
from datasets import PairedSRDataset
from models import *

from loss import Sobel3DLoss

from utils import (
    set_seed,
    save_log,
    save_output_volumes,
    calculate_psnr_ssim_lpips,
    load_encdec_from_autoencoder,
    print_networks,
)

from autoencoder import AutoEncoder3Dto2Dto3D

experiment_name = "my_neurosr"
lr_dir = "data/LR/train"
hr_dir = "data/HR/train"
lr_dir_test = "data/LR/mini_test"
hr_dir_test = "data/HR/mini_test"
checkpoint_dir = f"checkpoints/{experiment_name}"
images_dir = f"images/{experiment_name}"
log_dir = f"logs/{experiment_name}"
encdec_statedict_path = "model_zoo/enc_ec_3d.pth"

# check this!!!
use_ASSL = True
use_sobel = True
init_epoch = 0

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

train_log_path = os.path.join(log_dir, "training_log.txt")
metrics_log_path = os.path.join(log_dir, "metrics_log.txt")

title = f"Training with arguments: {experiment_name}..."
print(title)
save_log(train_log_path, title)

# params for dataloader
hr_shape = (64, 256, 256)
batch_size = 4
n_cpu = 4

# params for generater
img_size = 256
patch_size = 1
in_chans_SR = 64
in_chans_D = 64
embed_dim = 128
depths = (2, 2, 2, 2)
drop_rate = 0.0
resi_connection = "1conv"

# params for discriminator
num_filters_D = 256

# params for optimizers
lr = 1e-4
weight_decay = 0.01

# params for training process
n_epochs = 600
save_interval = 500  # batches
checkpoint_interval = 1  # epochs
warmup_batches = 75000
test_every = 10  # epochs
lambda_sobel = 0.05
lambda_gan = 0.2
lambda_hess = 1
lambda_sparse = 1e-9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# mean and std of the dataset
mean = {"hr": 0.053519323, "lr": 0.04907289}
std = {"hr": 0.04579687, "lr": 0.053449016}

ms_mes = f"mean:{mean}, std:{std}"
save_log(train_log_path, ms_mes)

dataloader = DataLoader(
    PairedSRDataset(lr_dir, hr_dir, mean, std),
    shuffle=True,
    batch_size=batch_size,
    num_workers=n_cpu,
)

test_dataloader = DataLoader(
    PairedSRDataset(lr_dir_test, hr_dir_test, mean, std),
    shuffle=False,
    batch_size=1,
    num_workers=1,
)

generator = NeuroSR(
    img_size=img_size,
    patch_size=patch_size,
    in_chans=in_chans_SR,
    embed_dim=embed_dim,
    depths=depths,
    drop_rate=drop_rate,
    d_state=16,
    mlp_ratio=2.0,
    drop_path_rate=0.1,
    norm_layer=nn.LayerNorm,
    patch_norm=True,
    use_checkpoint=False,
    upscale=4,
    img_range=1.0,
    upsampler="pixelshuffle",
    resi_connection=resi_connection,
    use_ASSL=use_ASSL,
).to(device)

print_networks(generator, verbose=True, log_file_path=train_log_path)

discriminator = UNetDiscriminatorSN(num_in_ch=in_chans_D, num_feat=num_filters_D).to(
    device
)

ae = AutoEncoder3Dto2Dto3D().to(device)

parameters_to_optim = load_encdec_from_autoencoder(
    sr_model=generator,
    autoencoder=ae,
    ckp_path=encdec_statedict_path,
    device=device,
    lock=False,
)

optimizer_G = torch.optim.AdamW(parameters_to_optim, lr=lr, weight_decay=weight_decay)
optimizer_D = torch.optim.AdamW(
    discriminator.parameters(), lr=lr, weight_decay=weight_decay
)

criterion_pixel = nn.L1Loss().to(device)
criterion_GAN = nn.BCEWithLogitsLoss().to(device)
if use_sobel:
    criterion_sobel = Sobel3DLoss().to(device)


# ----------
#  Training
# ----------
if __name__ == "__main__":
    # set_seed(seed=12)
    if init_epoch > 0:
        generator.load_state_dict(
            torch.load(
                checkpoint_dir + f"/generator_{init_epoch}.pth", map_location="cpu"
            ),
            strict=True,
        )
        if os.path.isfile(checkpoint_dir + f"/discriminator_{init_epoch}.pth"):
            discriminator.load_state_dict(
                torch.load(
                    checkpoint_dir + f"/discriminator_{init_epoch}.pth",
                    map_location="cpu",
                ),
                strict=True,
            )

        message = f"Load model state of epoch {init_epoch} form {checkpoint_dir} ."
        print(message)
        save_log(train_log_path, message)

    for epoch in range(init_epoch, n_epochs):
        if epoch > 400:
            # test_every = 1
            checkpoint_interval = 1

        epoch_loss_G = 0.0
        epoch_loss_D = 0.0

        for i, imgs in enumerate(dataloader):
            batches_done = epoch * len(dataloader) + i
            # model inputs
            imgs_lr = imgs["lr"].to(device)
            imgs_hr = imgs["hr"].to(device)

            batch_size = imgs_lr.size(0)

            # Adversarial mground truths
            valid = torch.ones(
                batch_size,
                1,
                imgs_hr.size(2),
                imgs_hr.size(3),
                dtype=torch.float,
                device=device,
                requires_grad=False,
            )

            fake = torch.zeros(
                batch_size,
                1,
                imgs_hr.size(2),
                imgs_hr.size(3),
                dtype=torch.float,
                device=device,
                requires_grad=False,
            )

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            gen_hr, _ = generator(imgs_lr)

            loss_pixel = criterion_pixel(gen_hr, imgs_hr)

            if batches_done < warmup_batches:
                loss_pixel.backward()
                optimizer_G.step()
                epoch_loss_G += loss_pixel

                log_message = "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]" % (
                    epoch + 1,
                    n_epochs,
                    i + 1,
                    len(dataloader),
                    loss_pixel.item(),
                )
                print(log_message)
                save_log(train_log_path, log_message)

                if (batches_done + 1) % save_interval == 0:
                    save_output_volumes(
                        imgs_lr,
                        gen_hr,
                        imgs_hr,
                        batch_size,
                        images_dir,
                        batches_done,
                        mean,
                        std,
                    )
                if (batches_done + 1) % (len(dataloader) * checkpoint_interval) == 0:
                    torch.save(
                        generator.state_dict(),
                        checkpoint_dir + "/generator_%d.pth" % (epoch + 1),
                    )
                    torch.save(
                        discriminator.state_dict(),
                        checkpoint_dir + "/discriminator_%d.pth" % (epoch + 1),
                    )

                if (batches_done + 1) % int(len(dataloader) * test_every) == 0:
                    epoch_loss_G = epoch_loss_G / len(dataloader)
                    epoch_loss_D = epoch_loss_D / len(dataloader)

                    avg_psnr, avg_ssim, avg_lpips = calculate_psnr_ssim_lpips(
                        test_dataloader,
                        generator,
                        device,
                        mean=mean["hr"],
                        std=std["hr"],
                        save_results=False,
                    )

                    log_message = f"Epoch [{epoch + 1}], Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}, Average LPIPS: {avg_lpips:.4f}, Average Loss_G: {epoch_loss_G:.4f}, Average Loss_D: {epoch_loss_D:.4f}"
                    print(log_message)
                    save_log(metrics_log_path, log_message)

                continue
            pred_real = discriminator(imgs_hr).detach()
            pred_fake = discriminator(gen_hr)
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

            if use_sobel:
                loss_sobel = criterion_sobel(gen_hr.unsqueeze(1), imgs_hr.unsqueeze(1))
            else:
                loss_sobel = torch.tensor(0)

            loss_G = (loss_pixel + loss_GAN) / 2 + lambda_sobel * loss_sobel

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())

            loss_real = criterion_GAN(
                pred_real - pred_fake.mean(0, keepdim=True), valid
            )
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            epoch_loss_G += loss_G
            epoch_loss_D += loss_D

            log_message = (
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, pixel: %f]"
                % (
                    epoch + 1,
                    n_epochs,
                    i + 1,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_pixel.item(),
                )
            )
            print(log_message)
            save_log(train_log_path, log_message)

            if (batches_done + 1) % save_interval == 0:
                save_output_volumes(
                    imgs_lr,
                    gen_hr,
                    imgs_hr,
                    batch_size,
                    images_dir,
                    batches_done,
                    mean,
                    std,
                )

            if (batches_done + 1) % (len(dataloader) * checkpoint_interval) == 0:
                torch.save(
                    generator.state_dict(),
                    checkpoint_dir + "/generator_%d.pth" % (epoch + 1),
                )
                torch.save(
                    discriminator.state_dict(),
                    checkpoint_dir + "/discriminator_%d.pth" % (epoch + 1),
                )

            if (batches_done + 1) % int(len(dataloader) * test_every) == 0:
                epoch_loss_G = epoch_loss_G / len(dataloader)
                epoch_loss_D = epoch_loss_D / len(dataloader)

                avg_psnr, avg_ssim, avg_lpips = calculate_psnr_ssim_lpips(
                    test_dataloader,
                    generator,
                    device,
                    mean=mean["hr"],
                    std=std["hr"],
                    save_results=False,
                )

                log_message = f"Epoch [{epoch + 1}], Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}, Average LPIPS: {avg_lpips:.4f}, Average Loss_G: {epoch_loss_G:.4f}, Average Loss_D: {epoch_loss_D:.4f}"
                print(log_message)
                save_log(metrics_log_path, log_message)
