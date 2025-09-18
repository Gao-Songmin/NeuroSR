import torch
from torch import nn
from torch.utils.data import DataLoader
import tifffile as tf
import os
from degradations import resize3D
from models import *
from datasets import *
from utils import calculate_psnr_ssim_lpips, save_log


experiment_name = ""  # fill in the name of your experiment director
model_num = [500]  # checkpoint numbers

lr_dir_test = "Neuronal_Image_Dataset/LR_images/axons/test"
hr_dir_test = "Neuronal_Image_Dataset/HR_images/axons/test"

log_dir = f"cal_metrics/{experiment_name}"
log_file_path = f"{log_dir}/test_log.txt"

os.makedirs(log_dir, exist_ok=True)

for mn in model_num:
    checkpoint_model = f"checkpoints/{experiment_name}/generator_{mn}.pth"
    output_dir = f"testresults/Synthetic/{experiment_name}/{mn}"

    os.makedirs(output_dir, exist_ok=True)
    # params for generater
    img_size = 256
    patch_size = 1
    in_chans = 64
    embed_dim = 128
    depths = (2, 2, 2, 2, 2)
    drop_rate = 0.0
    resi_connection = "1conv"
    use_NSE = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = {"hr": 0.09339826, "lr": 0.096223116}
    std = {"hr": 0.07190361, "lr": 0.07844172}

    test_dataloader = DataLoader(
        PairedSRDataset(lr_dir_test, hr_dir_test, mean, std),
        shuffle=False,
        batch_size=1,
        num_workers=1,
    )

    generator = NeuroSR(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
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
        use_NSE=use_NSE,
    ).to(device)

    print_networks(generator, verbose=True, log_file_path=log_file_path)

    generator.load_state_dict(torch.load(checkpoint_model))
    generator.eval()

    avg_psnr, avg_ssim, avg_lpips = calculate_psnr_ssim_lpips(
        test_dataloader,
        generator,
        device,
        mean=mean["hr"],
        std=std["hr"],
        save_results=True,
        output_dir=output_dir,
    )
    log_message = f"[Model {mn}], Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}, Average LPIPS: {avg_lpips.item():.4f}"
    print(log_message)
    save_log(log_file_path, log_message)
