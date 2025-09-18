import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from models import *
from datasets import *
from utils import save_imgs_grid


exp_name = ""  # fill in the name of your experiment director
ckp = 400  # checkpoint number

model_path = f"checkpoints/MIP/{exp_name}/generator_{ckp}.pth"
test_image_dir = f"Neuronal_Image_Dataset/HR_images/axons/test/z_mip"
results_dir = f"testresults/MIP/Real/{exp_name}/{ckp}"

os.makedirs(results_dir, exist_ok=True)

# params for dataloader
batch_size = 1

# params for generater
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

z_mip_dataset = LRImageDataset(test_image_dir, mean, std)
print(f"len of z_mip_dataset: {len(z_mip_dataset)}")

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
).to(device)

generator.load_state_dict(torch.load(model_path))
generator.eval()
print(f"load state dict:{model_path}")

dataloader = DataLoader(z_mip_dataset, batch_size=batch_size, shuffle=False)

for img, img_name in dataloader:
    img = img.to(device)
    img_name = img_name[0]
    print(img_name)
    with torch.no_grad():
        img_sr = generator(img)

    gap = (torch.ones((1, 1, img_sr.shape[-2], 5)) * 255).to(device)

    save_imgs_grid(
        img,
        gap,
        img_sr,
        images_dir=results_dir,
        img_name=img_name,
        mean=mean,
        std=std,
    )
