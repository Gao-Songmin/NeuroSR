import torch
from torch import nn
from torch.utils.data import DataLoader
import tifffile as tf
import os
from degradations import resize_3d
from models import *
from datasets import *


ckp = 500  # checkpoint number
exp_name = ""  # fill in the name of your experiment director
output_dir = f"testresults/Real/{exp_name}/{ckp}/Concat"
checkpoint_model = f"checkpoints/{exp_name}/generator_{ckp}.pth"
test_image_dir = "Neuronal_Image_Dataset/HR_images/somata"

os.makedirs(output_dir, exist_ok=True)

# params for dataloader
batch_size = 1

# params for generater
img_size = 256
patch_size = 1
in_chans = 64
embed_dim = 128
depths = (2, 2, 2, 2, 2)
use_NSE = True
drop_rate = 0.0
resi_connection = "1conv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mean and std of soma dataset
mean = {"hr": 0.09339826, "lr": 0.096223116}
std = {"hr": 0.07190361, "lr": 0.07844172}

# # mean and std of shortest axon terminal
# mean = {"hr": 0.04511516, "lr": 0.04511516}
# std = {"hr": 0.017035864, "lr": 0.017035864}

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

generator.load_state_dict(torch.load(checkpoint_model))
generator.eval()

lr_dataset = TestDatasetForConcat(test_image_dir, 64, mean, std)
print(len(lr_dataset))

dataloader = DataLoader(lr_dataset, batch_size=batch_size, shuffle=False)

for patches, img_name in dataloader:
    img_name = img_name[0]
    print(img_name)
    patch_sr_list = []
    patch_bi_list = []
    patch_lr_list = []
    for patch_lr in patches:
        patch_lr = resize_3d(patch_lr.squeeze(), (64, 256, 256), order=1)
        patch_lr_list.append(patch_lr)
        patch_lr = torch.tensor(patch_lr).to(device)
        with torch.no_grad():
            patch_sr = generator(patch_lr)
        patch_sr = patch_sr.detach().cpu().numpy().squeeze()
        patch_sr_list.append(patch_sr)

    img_lr = np.concatenate(np.array(patch_lr_list), axis=0)
    img_sr = np.concatenate(np.array(patch_sr_list), axis=0)

    tf.imwrite(
        output_dir + f"/{img_name}_linear_cat.tiff",
        denormalize(img_lr, "hr", mean, std),
    )
    tf.imwrite(
        output_dir + f"/{img_name}_sr_cat.tiff", denormalize(img_sr, "hr", mean, std)
    )
