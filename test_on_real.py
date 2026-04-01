import torch
from torch import nn
from torch.utils.data import DataLoader
import tifffile as tf
import numpy as np
import os
from utils import resize3D
from models import *
from datasets import *


ckp = 450
exp_name = "PretrainedEncDec_3Dto2Dv2_skipf3_CASSv2_repair_MLP_shrink1.0,4.0_decay_6,10_over5,75_0.17pi"
# exp_name = "PretrainedEncDec_3Dto2D_skipf3_CSS_inVSSB_depth8_Decay_6,10_over5,75_0.17pi_Poisson_500_Gaus1.0,1.4_MP"

# supr_aug = "ls0.05_gan_40warm"
output_dir = (
    f"/home/gsm/python/Neuron_SR/MambaIR_NeuroConv/test_results/{exp_name}/{ckp}/Cat/train"
)
checkpoint_model = f"/home/gsm/python/Neuron_SR/MambaIR_NeuroConv/checkpoints/{exp_name}/generator_{ckp}.pth"
test_image_dir = "/home/gsm/data/Neuron_SR/soma_and_terminal_blocks/train"
# test_image_dir = "/home/gsm/data/Neuron_SR/multi_scale_soma_blocks/layer2"
os.makedirs(output_dir, exist_ok=True)

# params for dataloader
batch_size = 1

# params for generater
img_size = 256
patch_size = 1
in_chans = 64
embed_dim = 128
depths = (2, 2, 2, 2)
use_CASS = True
drop_rate = 0.0
resi_connection = "1conv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # mean and std of soma dataset
# mean = {"hr": 0.09339826, "lr": 0.096223116}
# std = {"hr": 0.07190361, "lr": 0.07844172}

# # mean and std of shortest axon terminal
# mean = {"hr": 0.04511516, "lr": 0.04511516}
# std = {"hr": 0.017035864, "lr": 0.017035864}

# mean and std of soma and terminal dataset
mean = {"hr": 0.053519323, "lr": 0.04907289}
std = {"hr": 0.04579687, "lr": 0.053449016}

# # FALSE mean and std of soma and terminal dataset with noise
# mean = {"hr": 0.041036185, "lr": 0.088972494}
# std = {"hr": 0.097351074, "lr": 0.05570848}

# # mean and std of soma and terminal dataset with noise
# mean = {"hr": 0.053519323, "lr": 0.07661523}
# std = {"hr": 0.04579687, "lr": 0.0545185}


generator = MambaIR_3Dto2D(
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
    use_CASS=use_CASS,
).to(device)

generator.load_state_dict(torch.load(checkpoint_model))
generator.eval()

message = f"Load model state: {checkpoint_model}."
print(message)

lr_dataset = TestDatasetForConcat(root=test_image_dir, depth=64, mean=mean, std=std)
print(f"len of lr dataset: {len(lr_dataset)}")

dataloader = DataLoader(lr_dataset, batch_size=batch_size, shuffle=False)

for patches, img_name in dataloader:
    img_name = img_name[0]
    print(img_name)
    patch_sr_list = []
    patch_bi_list = []
    patch_linear_list = []
    for patch_lr in patches:
        patch_lr_linear = resize3D(patch_lr.squeeze(), (64, 256, 256), order=1)
        patch_linear_list.append(patch_lr_linear)
        # patch_lr_bi = resize3D(patch_lr.squeeze(), (64, 256, 256), order=3)
        # patch_bi_list.append(patch_lr_bi)
        patch_lr = torch.tensor(patch_lr).to(device)
        with torch.no_grad():
            patch_sr, _ = generator(patch_lr)
        patch_sr = patch_sr.detach().cpu().numpy().squeeze()
        patch_sr_list.append(patch_sr)

    img_linear = np.concatenate(np.array(patch_linear_list), axis=0)
    # img_bi = np.concatenate(np.array(patch_bi_list), axis=0)
    img_sr = np.concatenate(np.array(patch_sr_list), axis=0)

    tf.imwrite(
        output_dir + f"/{img_name}_linear_cat.tiff",
        denormalize(img_linear, "hr", mean, std),
    )
    tf.imwrite(
        output_dir + f"/{img_name}_sr_cat.tiff", denormalize(img_sr, "hr", mean, std)
    )
    # tf.imwrite(
    #     output_dir + f"/{img_name}_lr_bi_cat.tiff", denormalize(img_bi, "hr", mean, std)
    # )
