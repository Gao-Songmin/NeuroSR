import torch
from torch import nn
from torch.utils.data import DataLoader
import tifffile as tf
import numpy as np
import os
from utils import resize3D, arr_denormalize
from models import *
from datasets import *


exp_name = "my_neuroSR"
output_dir = f"test_results/{exp_name}/"
checkpoint_model = "model_zoo/neurosr_generator.pth"
# test_image_dir = "data/HR/test"
test_image_dir = "demo"

os.makedirs(output_dir, exist_ok=True)

# params for dataloader
batch_size = 1

# params for generater
img_size = 256
patch_size = 1
in_chans = 64
embed_dim = 128
depths = (2, 2, 2, 2)
use_ASSL = True
drop_rate = 0.0
resi_connection = "1conv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mean and std of the dataset
mean = {"hr": 0.053519323, "lr": 0.04907289}
std = {"hr": 0.04579687, "lr": 0.053449016}


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
    use_ASSL=use_ASSL,
).to(device)

generator.load_state_dict(torch.load(checkpoint_model))
generator.eval()

message = f"Load model state: {checkpoint_model}."
print(message)

real_dataset = TestDatasetForConcat(root=test_image_dir, depth=64, mean=mean, std=std)
print(f"len of lr dataset: {len(real_dataset)}")

dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)

for patches, img_name in dataloader:
    img_name = img_name[0]
    print(img_name)
    patch_sr_list = []
    patch_bi_list = []
    patch_linear_list = []
    for patch_lr in patches:
        patch_lr_linear = resize3D(patch_lr.squeeze(), (64, 256, 256), order=1)
        patch_linear_list.append(patch_lr_linear)
        patch_lr = torch.tensor(patch_lr).to(device)
        with torch.no_grad():
            patch_sr, _ = generator(patch_lr)
        patch_sr = patch_sr.detach().cpu().numpy().squeeze()
        patch_sr_list.append(patch_sr)

    img_linear = np.concatenate(np.array(patch_linear_list), axis=0)
    img_sr = np.concatenate(np.array(patch_sr_list), axis=0)

    img_linear_out = (arr_denormalize(img_linear, mean["hr"], std["hr"]) * 255).astype(
        np.uint8
    )

    img_sr_out = (arr_denormalize(img_sr, mean["hr"], std["hr"]) * 255).astype(np.uint8)

    tf.imwrite(
        output_dir + f"/{img_name}_linear.tiff",
        img_linear_out,
    )
    tf.imwrite(
        output_dir + f"/{img_name}_sr.tiff",
        img_sr_out,
    )
