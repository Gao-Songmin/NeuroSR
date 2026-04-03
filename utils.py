import torch
from torch import Tensor
from typing import Optional
from lpips import LPIPS
from torch import nn
import random
import numpy as np
from numpy.typing import NDArray
import tifffile as tf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import zoom
from wpsnr import *


def resize3D(volume, new_size: tuple[int, int, int], order: int) -> NDArray:
    """
    order : int, optional
        The order of the spline interpolation, default is 3 (bicubic).
        The order has to be in the range 0-5.
    """
    if isinstance(volume, Tensor):
        volume = volume.cpu().numpy()
    z, y, x = new_size
    d, h, w = volume.shape
    zoom_factor = (z / d, y / h, x / w)
    return zoom(volume, zoom_factor, order=order)


def set_seed(seed=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"Random seed set to {seed}")


def load_encdec_from_autoencoder(
    sr_model: nn.Module,
    autoencoder: nn.Module,
    ckp_path: str = "model_zoo/model.pth",
    device: Optional[torch.device] = None,
    lock: bool = False,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder.load_state_dict(torch.load(ckp_path))
    autoencoder.eval()

    sr_model.encoder.load_state_dict(autoencoder.encoder.state_dict(), strict=True)
    sr_model.decoder.load_state_dict(autoencoder.decoder.state_dict(), strict=True)

    if lock:
        for p in sr_model.encoder.parameters():
            p.requires_grad = False

        for p in sr_model.decoder.parameters():
            p.requires_grad = False

        sr_model.encoder.eval()
        sr_model.decoder.eval()
    parameters = [p for p in sr_model.parameters() if p.requires_grad]

    return parameters


def print_networks(
    net: nn.Module, verbose=True, log_file_path: Optional[torch.device] = None
):
    num_params = 0
    message = "---------- Networks initialized -------------"
    for param in net.parameters():
        num_params += param.numel()
    if verbose:
        message += net.__str__() + "\n"
    message += "[Network] Total number of parameters : %.3f M\n" % (num_params / 1e6)
    message += "-----------------------------------------------"
    print(message)
    with open(log_file_path, "a") as log_file:
        log_file.write(message)


def arr_denormalize(array: NDArray, mean, std):
    array = array * std + mean
    array = array.clip(0, 1)
    return array


def denormalize(tensor, mean, std):
    print(f"mean, std:{mean, std}")
    if isinstance(mean, float):
        mean = [mean] * tensor.shape[1]

    if isinstance(std, float):
        std = [std] * tensor.shape[1]

    for c in range(tensor.shape[1]):
        tensor[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensor, 0, 1)


def save_log(log_file_path, message):
    """Helper function to append a message to the log file."""
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")


def save_output_volumes(
    imgs_lr, gen_hr, imgs_hr, batch_size, images_dir, batches_done, mean, std
):
    imgs_lr_np = imgs_lr.cpu().numpy()
    gen_hr_np = gen_hr.detach().cpu().numpy()
    imgs_hr_np = imgs_hr.cpu().numpy()
    for i in range(batch_size):
        img_lr = (
            resize3D(
                arr_denormalize(imgs_lr_np[i], mean["lr"], std["lr"]),
                imgs_hr.shape[1:4],
                order=3,
            ).clip(0, 1)
            * 255
        ).astype(np.uint8)
        img_gen = (arr_denormalize(gen_hr_np[i], mean["hr"], std["hr"]) * 255).astype(
            np.uint8
        )
        img_hr = (arr_denormalize(imgs_hr_np[i], mean["hr"], std["hr"]) * 255).astype(
            np.uint8
        )
        tf.imwrite(images_dir + f"/lr_batch{batches_done + 1}_{i}.tif", img_lr)
        tf.imwrite(
            images_dir + f"/gen_batch{batches_done + 1}_{i}.tif",
            img_gen,
        )
        tf.imwrite(images_dir + f"/hr_batch{batches_done + 1}_{i}.tif", img_hr)


def preprocess_for_lpips(img_np):
    # img_np: numpy array with shape [H, W, 3], value range [0, 255] or [0, 1]
    img_np = img_np / img_np.max()  # convert [0,255] -> [0,1]
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # [H, W]
    img_tensor = img_tensor * 2 - 1  # [0,1] -> [-1,1]
    return img_tensor


def calculate_psnr_ssim_lpips(test_dataloader, model, device, mean, std):
    """Calculate PSNR and SSIM for the test dataset and save slices to a .tiff file."""
    model.eval()
    psnr_sum = 0
    ssim_sum = 0
    lpips_sum = 0
    count = 0

    loss_fn_alex = LPIPS(net="alex")

    with torch.no_grad():
        for batch_idx, imgs in enumerate(test_dataloader):
            inputs, targets = imgs["lr"], imgs["hr"]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            outputs = outputs.detach().cpu().numpy()
            targets = targets.cpu().numpy()
            outputs, targets = (
                arr_denormalize(outputs, mean, std),
                arr_denormalize(targets, mean, std),
            )
            data_range = 1.0

            for i in range(outputs.shape[0]):
                psnr_batch_sum = 0
                ssim_batch_sum = 0
                lpips_batch_sum = 0
                slice_count = outputs.shape[1]

                for j in range(slice_count):
                    output_img = outputs[i, j]
                    target_img = targets[i, j]

                    psnr = peak_signal_noise_ratio(
                        target_img,
                        output_img,
                        data_range=data_range,
                    )

                    ssim = structural_similarity(
                        target_img, output_img, data_range=data_range
                    )

                    lpips = loss_fn_alex(
                        preprocess_for_lpips(target_img),
                        preprocess_for_lpips(output_img),
                    )

                    psnr_batch_sum += psnr
                    ssim_batch_sum += ssim
                    lpips_batch_sum += lpips.item()

                psnr_avg_per_sample = psnr_batch_sum / slice_count
                ssim_avg_per_sample = ssim_batch_sum / slice_count
                lpips_avg_per_sample = lpips_batch_sum / slice_count

                psnr_sum += psnr_avg_per_sample
                ssim_sum += ssim_avg_per_sample
                lpips_sum += lpips_avg_per_sample

                count += 1

                print(
                    f"<{batch_idx * inputs.shape[0] + i + 1}> testing images finished."
                )

    model.train()

    return psnr_sum / count, ssim_sum / count, lpips_sum / count
