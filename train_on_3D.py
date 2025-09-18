import torch
from torch.utils.data import DataLoader
from torch import nn
import os
from datasets import *
from models import *
from loss import Sobel3DLoss
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_name",
    type=str,
    required=True,
    help="name of your experiment director.",
)
parser.add_argument(
    "--lr_dir",
    type=str,
    default="Neuronal_Image_Dataset/LR_images/axons/train",
)
parser.add_argument(
    "--hr_dir",
    type=str,
    default="Neuronal_Image_Dataset/HR_images/axons/train",
)
parser.add_argument(
    "--lr_dir_test",
    type=str,
    default="Neuronal_Image_Dataset/LR_images/axons/test",
)
parser.add_argument(
    "--hr_dir_test",
    type=str,
    default="Neuronal_Image_Dataset/HR_images/axons/test",
)

parser.add_argument(
    "--pretrain_models",
    type=str,
    default=None,
)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--n_cpu", type=int, default=4)
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--in_chans", type=int, default=64)
parser.add_argument(
    "--embed_dim",
    type=int,
    default=128,
    help="the channel num in deep feature extraction process.",
)
parser.add_argument("--depths", type=tuple[int], default=(2, 2, 2, 2, 2))

parser.add_argument("--init_epoch", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=500)
parser.add_argument(
    "--save_interval", type=int, default=200, help="save output images every # batches."
)
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=1,
    help="save model state_dicts every # epochs.",
)
parser.add_argument(
    "--warmup_batches",
    type=int,
    default=10000,
    help="pretrain with how many batches with L1 pixel loss before training with GAN.",
)
parser.add_argument(
    "--test_every",
    type=float,
    default=10,
    help="test and calculate metrics every # epochs",
)

args = parser.parse_args()
checkpoint_dir = f"checkpoints/{args.exp_name}"
images_dir = f"images/{args.exp_name}"
log_dir = f"logs/{args.exp_name}"

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

train_log_path = os.path.join(log_dir, "training_log.txt")
metrics_log_path = os.path.join(log_dir, "metrics_log.txt")

title = f"Start the training of experiment: {args.exp_name}..."
print(title)
save_log(train_log_path, title)


# params for dataloader
hr_shape = (64, 256, 256)
n_cpu = args.n_cpu

# params for generater
patch_size = 1
use_NSE = True
use_sobel = True
drop_rate = 0.0
resi_connection = "1conv"

# params for discriminator
num_filters_D = 256

# params for optimizers
lr = 1e-4
weight_decay = 0.01

# params for training process
lambda_sobel = 0.05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mean and std of soma dataset
mean = {"hr": 0.09339826, "lr": 0.096223116}
std = {"hr": 0.07190361, "lr": 0.07844172}

# # mean and std for shortest axon terminal
# mean = {"hr": 0.04511516, "lr": 0.037987333}
# std = {"hr": 0.017035864, "lr": 0.019624418}

args.mean = mean
args.std = std

save_log(train_log_path, args.__str__())

dataloader = DataLoader(
    PairedSRDataset(args.lr_dir, args.hr_dir, mean, std),
    shuffle=True,
    batch_size=args.batch_size,
    num_workers=n_cpu,
)

test_dataloader = DataLoader(
    PairedSRDataset(args.lr_dir_test, args.hr_dir_test, mean, std),
    shuffle=False,
    batch_size=2,
    num_workers=2,
)

generator = NeuroSR(
    img_size=args.img_size,
    patch_size=patch_size,
    in_chans=args.in_chans,
    embed_dim=args.embed_dim,
    depths=args.depths,
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
    use_NSE=use_NSE,
).to(device)

print_networks(generator, verbose=True, log_file_path=train_log_path)

discriminator = UNetDiscriminatorSN(num_in_ch=args.in_chans, num_feat=num_filters_D).to(
    device
)

criterion_pixel = nn.L1Loss().to(device)
criterion_GAN = nn.BCEWithLogitsLoss().to(device)
if use_sobel:
    criterion_sobel = Sobel3DLoss().to(device)

optimizer_G = torch.optim.AdamW(
    generator.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer_D = torch.optim.AdamW(
    discriminator.parameters(), lr=lr, weight_decay=weight_decay
)


# ----------
#  Training
# ----------
if __name__ == "__main__":
    if args.init_epoch > 0:
        old_check_dir = checkpoint_dir
        generator.load_state_dict(
            torch.load(old_check_dir + f"/generator_{args.init_epoch}.pth")
        )
        discriminator.load_state_dict(
            torch.load(old_check_dir + f"/discriminator_{args.init_epochh}.pth")
        )

        message = f"Load model state of epoch {args.init_epoch} form {old_check_dir} ."
        print(message)
        save_log(train_log_path, message)

    elif args.pretrain_models is not None:
        generator.load_state_dict(torch.load(args.pretrain_models))
        message = f"Load model generator state from {args.pretrain_models}."
        print(message)
        save_log(train_log_path, message)

    for epoch in range(args.init_epoch, args.n_epochs):
        if epoch > 400:
            args.checkpoint_interval = 1

        if epoch > 450:
            args.test_every = 1
            args.save_interval = 50

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

            gen_hr = generator(imgs_lr)
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)

            if batches_done < args.warmup_batches:
                loss_pixel.backward()
                optimizer_G.step()
                epoch_loss_G += loss_pixel

                log_message = "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]" % (
                    epoch + 1,
                    args.n_epochs,
                    i + 1,
                    len(dataloader),
                    loss_pixel.item(),
                )
                print(log_message)
                save_log(train_log_path, log_message)

                if (batches_done + 1) % args.save_interval == 0:
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
                if (batches_done + 1) % len(dataloader) == 0:
                    torch.save(
                        generator.state_dict(),
                        checkpoint_dir + "/generator_%d.pth" % (epoch + 1),
                    )
                    torch.save(
                        discriminator.state_dict(),
                        checkpoint_dir + "/discriminator_%d.pth" % (epoch + 1),
                    )

                if (batches_done + 1) % int(len(dataloader) * args.test_every) == 0:
                    # if (batches_done + 1) % 10 == 0:
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

                    log_message = f"Epoch [{epoch + 1}], Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}, Average LPIPS: {avg_lpips.item():.4f}, Average Loss_G: {epoch_loss_G:.4f}, Average Loss_D: {epoch_loss_D:.4f}"
                    print(log_message)
                    save_log(metrics_log_path, log_message)

                continue

            pred_real = discriminator(imgs_hr).detach()
            pred_fake = discriminator(gen_hr)
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
            if use_sobel:
                loss_sobel = criterion_sobel(gen_hr.unsqueeze(1), imgs_hr.unsqueeze(1))
            # loss_EMD = criterion_EMD(gen_hr.unsqueeze(1), imgs_hr.unsqueeze(1))

            loss_G = (
                (loss_pixel + loss_GAN) / 2 + lambda_sobel * loss_sobel
                if use_sobel
                else (loss_pixel + loss_GAN) / 2
            )
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
                    args.n_epochs,
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

            if (batches_done + 1) % args.save_interval == 0:
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

            if (batches_done + 1) % len(dataloader) == 0:
                torch.save(
                    generator.state_dict(),
                    checkpoint_dir + "/generator_%d.pth" % (epoch + 1),
                )
                torch.save(
                    discriminator.state_dict(),
                    checkpoint_dir + "/discriminator_%d.pth" % (epoch + 1),
                )

            if (batches_done + 1) % int(len(dataloader) * args.test_every) == 0:
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

                log_message = f"Epoch [{epoch + 1}], Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}, Average LPIPS: {avg_lpips.item():.4f}, Average Loss_G: {epoch_loss_G:.4f}, Average Loss_D: {epoch_loss_D:.4f}"
                print(log_message)
                save_log(metrics_log_path, log_message)
