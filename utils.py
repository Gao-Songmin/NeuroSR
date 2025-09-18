import torch
from lpips import LPIPS
from torch import nn
import torch.nn.functional as F
import numpy as np
from numpy.typing import NDArray
import os
import tifffile as tf
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.stats import truncnorm
from degradations import resize_3d


def print_networks(net: nn.Module, verbose=True, log_file_path=None):
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


def save_imgs_grid(imgs_lr, gen_hr, imgs_hr, images_dir, img_name, mean, std, gap=None):
    imgs_lr = nn.functional.interpolate(
        imgs_lr,
        scale_factor=(4, 1),
        mode="bicubic",
        align_corners=False,
    )

    if gap is None:
        gap = (
            torch.ones((imgs_hr.shape[0], imgs_hr.shape[1], imgs_hr.shape[2], 5)) * 255
        ).cuda()

    img_grid = denormalize(
        torch.cat((imgs_lr, gap, gen_hr, gap, imgs_hr), -1), mean, std
    )

    save_image(
        img_grid,
        images_dir + f"/{img_name}.tiff",
        nrow=1,
        normalize=False,
    )


def save_imgs_grid_for_SSR(imgs, imgs_ssr, images_dir, img_name, mean, std, gap=None):
    if gap is None:
        gap = (
            torch.ones((imgs_ssr.shape[0], imgs_ssr.shape[1], imgs_ssr.shape[2], 5))
            * 255
        ).cuda()

    img_grid = denormalize(torch.cat((imgs, gap, imgs_ssr), -1), mean, std)

    save_image(
        img_grid,
        images_dir + f"/{img_name}.tiff",
        nrow=1,
        normalize=False,
    )


def save_output_volumes(
    imgs_lr, gen_hr, imgs_hr, batch_size, images_dir, batches_done, mean, std
):
    imgs_lr_np = imgs_lr.cpu().numpy()
    gen_hr_np = gen_hr.detach().cpu().numpy()
    imgs_hr_np = imgs_hr.cpu().numpy()
    for i in range(batch_size):
        img_lr = (
            resize_3d(
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


def truncnorm_sampler(a, b, size):
    norm_a, norm_b = (a - 0.0) / 0.3, (b - 0.0) / 0.3  # 即 (-3.33, 3.33)
    samples = truncnorm.rvs(norm_a, norm_b, loc=0.0, scale=0.3, size=size)
    return samples


def boardered_normalize(img):
    """normalize an image to [-1, 1]"""
    img = img / img.max()
    img = img * 2 - 1
    return img


def preprocess_for_lpips(img_np):
    # img_np: numpy array with shape [H, W, 3], value range [0, 255] or [0, 1]
    img_np = img_np / img_np.max()  # convert [0,255] -> [0,1]
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # [H, W]
    img_tensor = img_tensor * 2 - 1  # [0,1] -> [-1,1]
    return img_tensor


def calculate_psnr_ssim_lpips(
    test_dataloader, model, device, mean, std, save_results=False, output_dir="."
):
    """Calculate PSNR and SSIM for the test dataset and save slices to a .tiff file."""
    model.eval()
    psnr_sum = 0
    ssim_sum = 0
    lpips_sum = 0
    count = 0
    loss_fn_alex = LPIPS(net="alex")

    with torch.no_grad():
        for batch_idx, imgs in enumerate(test_dataloader):
            inputs, targets, f_names = imgs["lr"], imgs["hr"], imgs["fn"]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).detach().cpu().numpy()
            # inputs = inputs.cpu().numpy()
            targets = targets.cpu().numpy()
            outputs, targets = (
                arr_denormalize(outputs, mean, std),
                arr_denormalize(targets, mean, std),
            )

            data_range = 1.0

            for i in range(outputs.shape[0]):  # 遍历batch中的每个样本
                psnr_batch_sum = 0
                ssim_batch_sum = 0
                lpips_batch_sum = 0
                slice_count = outputs.shape[1]  # 获取切片数量

                # 创建一个列表用于存储每个样本的输出和目标切片
                output_slices = []
                target_slices = []

                for j in range(slice_count):  # 遍历每个切片
                    output_img = outputs[i, j]
                    # output_img = inputs[i, j]
                    target_img = targets[i, j]

                    # 计算当前切片的PSNR和SSIM
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
                    lpips_batch_sum += lpips

                    # 将切片添加到列表中
                    output_slices.append(output_img)
                    target_slices.append(target_img)

                # 平均每个样本中所有切片的PSNR和SSIM
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

                if save_results:
                    f_name = f_names[i]
                    input_np = inputs[i].cpu().numpy().squeeze()
                    input_up = resize_3d(input_np, (64, 256, 256), order=1)
                    input_up = (arr_denormalize(input_up, mean, std) * 255).astype(
                        np.uint8
                    )
                    output = (outputs[i].squeeze() * 255).astype(np.uint8)
                    target = (targets[i].squeeze() * 255).astype(np.uint8)

                    tf.imwrite(output_dir + f"/{f_name}_lr_up.tiff", input_up)
                    tf.imwrite(output_dir + f"/{f_name}_sr.tiff", output)
                    tf.imwrite(output_dir + f"/{f_name}_hr.tiff", target)

    model.train()

    return psnr_sum / count, ssim_sum / count, lpips_sum / count
