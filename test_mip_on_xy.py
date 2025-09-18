from models import NeuroSR
from datasets import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from utils import save_imgs_grid

exp_name = ""  # fill in the name of your experiment director
ckp = 400  # checkpoint number
model_path = f"./checkpoints/MIP/{exp_name}/generator_{ckp}.pth"
test_image_dir = "Neuronal_Image_Dataset/HR_images/axons/test/xy_mip"
output_dir = f"testresults/MIP/Synthetic/{exp_name}/{ckp}"

os.makedirs(output_dir, exist_ok=True)
# params for dataloader
batch_size = 1
scale = 4

# params for generator
img_size = 256
patch_size = 1
in_chans = 1
embed_dim = 96
depths = (4, 4, 4, 4)
drop_rate = 0.0
resi_connection = "1conv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = 0.2
std = 0.2

dataloader = DataLoader(
    ImageDataset_2D(
        test_image_dir,
        hr_shape=(1, 256, 256),
        mean=mean,
        std=std,
        augmentation=False,
        decay_range=(8, 8),
        direction_range=(-1, 1),
        threshold=40,
    ),
    batch_size=batch_size,
    shuffle=False,
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
    upscale=scale,
    img_range=1.0,
    upsampler="pixelshuffle",
    resi_connection=resi_connection,
).to(device)

generator.load_state_dict(torch.load(model_path))
generator.eval()
print(f"load state dict:{model_path}")

for i, imgs in enumerate(dataloader):
    imgs_lr = imgs["lr"].to(device)
    imgs_hr = imgs["hr"].to(device)
    img_name = imgs["fn"][0]

    print(f"img_name:{img_name}")

    sr_imgs = generator(imgs_lr)

    save_imgs_grid(
        imgs_lr,
        sr_imgs,
        imgs_hr,
        images_dir=output_dir,
        img_name=img_name,
        mean=mean,
        std=std,
    )
