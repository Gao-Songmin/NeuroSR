import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
from lpips import LPIPS
from torch import nn
import random
import numpy as np
from numpy.typing import NDArray
import pyiqa
from piq import vif_p
from piq.ms_ssim import multi_scale_ssim
import tifffile as tf
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.stats import truncnorm
from scipy.ndimage import zoom
from wpsnr import *

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def robust_norm(
    x: NDArray, p_low: float = 1.0, p_high: float = 99.0, eps: float = 1e-8
) -> NDArray:
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + eps)
    return x


def norm_per_channel(x_chw, p_low=1, p_high=99, gamma=0.8):
    """
    x_chw: (C,H,W)
    return: (C,H,W) in [0,1]
    """
    out = np.empty_like(x_chw, dtype=np.float32)
    for c in range(x_chw.shape[0]):
        xc = robust_norm(x_chw[c], p_low, p_high)
        if gamma is not None:
            xc = np.power(xc, gamma)
        out[c] = xc
    return out


def save_feature_png(x_hw_01, save_path, cmap="magma"):
    """
    x_hw_01: (H,W) float in [0,1]
    """
    plt.imsave(save_path, x_hw_01, cmap=cmap, vmin=0.0, vmax=1.0)


def aggregate_map(x_chw, mode="l2"):
    if mode == "mean":
        m = x_chw.mean(axis=0)
    elif mode == "max":
        m = x_chw.max(axis=0)
    elif mode == "l2":
        m = np.sqrt((x_chw**2).sum(axis=0))
    else:
        raise ValueError(mode)
    return m


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


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


def rope(x, shape, base=10000):
    channel_dims, feature_dim = shape[:-1], shape[-1]
    k_max = feature_dim // (2 * len(channel_dims))

    assert feature_dim % k_max == 0

    # angles
    theta_ks = 1 / (base ** (torch.arange(k_max, device=x.device) / k_max))
    angles = torch.cat(
        [
            t.unsqueeze(-1) * theta_ks
            for t in torch.meshgrid(
                [torch.arange(d, device=x.device) for d in channel_dims], indexing="ij"
            )
        ],
        dim=-1,
    )

    # rotation
    rotations_re = torch.cos(angles).unsqueeze(dim=-1)
    rotations_im = torch.sin(angles).unsqueeze(dim=-1)

    x = x.reshape(*x.shape[:-1], -1, 2)
    x_re = x[..., :1]
    x_im = x[..., 1:]
    pe_x = torch.cat(
        [
            x_re * rotations_re - x_im * rotations_im,
            x_im * rotations_re + x_re * rotations_im,
        ],
        dim=-1,
    )
    return pe_x.flatten(-2)


def decompose_matrix(A, d, niter=2):
    # Perform SVD
    U, S, V = torch.svd_lowrank(A, q=d, niter=niter)

    # Truncate U, S, and V to get dimensions N x d and d x N
    U_d = U[:, :d]
    S_d = S[:d]
    V_d = V[:, :d]

    # Construct the two matrices
    # Matrix 1 (N x d)
    M1 = U_d * torch.sqrt(S_d)
    # Matrix 2 (d x N)
    M2 = V_d * torch.sqrt(S_d)
    return M1, M2


def get_sine_svd(pos_embed, svd_dim=128, niter=2):
    pos_embed = pos_embed.flatten(0, 2)
    pos_sim = pos_embed @ pos_embed.T
    pos_softmax = F.softmax(pos_sim, dim=-1)
    # pos_softmax = pos_softmax * (pos_softmax > 0.1*pos_softmax.mean())
    trancated_q, trancated_k = decompose_matrix(pos_softmax, svd_dim, niter)
    trancated_q, trancated_k = (
        trancated_q.unsqueeze(0),
        trancated_k.unsqueeze(0),
    )  # (1, N, svd_dim), (1, N, svd_dim)
    return [trancated_q, trancated_k]


class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding."""

    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat(
            [
                t.unsqueeze(-1) * theta_ks
                for t in torch.meshgrid(
                    [torch.arange(d) for d in channel_dims], indexing="ij"
                )
            ],
            dim=-1,
        )

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer("rotations", rotations)

    def forward(self, x):
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


def process_for_sr(img: Tensor, mean, std) -> Tensor:
    img = img.type(torch.float32) / 255.0
    img = (img - mean) / std
    return img


def abs_l1(img: Tensor) -> Tensor:
    return img.abs().sum(dim=tuple(range(1, img.ndim))).mean()


def set_seed(seed=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"Random seed set to {seed}")


def extract_sr_state_dict(full_sd: dict) -> dict:
    return {
        k: v
        for k, v in full_sd.items()
        if not k.startswith("encoder.") or k.startswith("decoder.")
    }


def load_sr_sd_only(full_net: nn.Module, sr_sd: dict, strict: bool = False):
    return full_net.load_state_dict(sr_sd, strict=strict)


def save_sr_checkpoint_only(path: str, full_net: nn.Module, optimizer_G):
    sr_sd = extract_sr_state_dict(full_net.state_dict())
    torch.save({"sr_state_dict": sr_sd, "optimizer_G": optimizer_G}, path)


def load_sr_checkpoint_only(path: str, full_net: nn.Module, optimizer_G):
    ckp = torch.load(path)
    sr_only_sd = ckp["sr_state_dict"]
    full_net.load_state_dict(sr_only_sd, strict=False)
    if "optimizer_G" in ckp:
        optimizer_G.load_state_dict(ckp["optimizer_G"])


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


def arr_normalize(array: NDArray):
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        array = (array - array.min()) / (array.max() - array.min())

    array = array.clip(0, 1)
    return array


def normalize(imgs: Tensor, mean, std) -> Tensor:
    imgs = imgs / 255.0
    imgs = (imgs - mean) / std
    return imgs


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


def save_imgs_grid(imgs_lr, gen_hr, imgs_hr, images_dir, img_name, gap=None):
    imgs_lr = nn.functional.interpolate(
        imgs_lr,
        scale_factor=(4, 1),
        mode="bicubic",
        align_corners=False,
    )
    if gap is None:
        gap = (torch.ones((4, 1, imgs_hr.shape[-2], 5)) * 255).cuda()

    img_grid = denormalize(torch.cat((imgs_lr, gap, gen_hr, gap, imgs_hr), -1))

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


def truncnorm_sampler(a, b, size):
    norm_a, norm_b = (a - 0.0) / 0.3, (b - 0.0) / 0.3  # 即 (-3.33, 3.33)
    samples = truncnorm.rvs(norm_a, norm_b, loc=0.0, scale=0.3, size=size)
    return samples


def boardered_normalize(img):
    """normalize an image to [-1, 1]"""
    img = img / img.max()
    img = img * 2 - 1
    return img


def tensor_denormalize(tensor: Tensor, mean, std):
    tensor = tensor * std + mean
    # tensor = tensor / tensor.max()
    tensor = tensor.clamp(0, 1)
    return tensor


def preprocess_for_lpips_tensor(img: Tensor) -> Tensor:
    img = img.type(torch.float32) / img.max()  # convert [0,255] -> [0,1]
    # img = img.type(torch.float32)  # have change to [0, 1] in tensor denormalization()
    img = img.unsqueeze(0).unsqueeze(0)  # [H, W] -> [B,C,H,W]
    img = img * 2 - 1  # [0,1] -> [-1,1]
    return img


def preprocess_for_lpips(img_np):
    # img_np: numpy array with shape [H, W, 3], value range [0, 255] or [0, 1]
    img_np = img_np / img_np.max()  # convert [0,255] -> [0,1]
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # [H, W]
    img_tensor = img_tensor * 2 - 1  # [0,1] -> [-1,1]
    return img_tensor


def nrmse_torch(pred: Tensor, gt: Tensor, normalization="range"):
    pred = pred.float()
    gt = gt.float()
    mse = torch.mean((pred - gt) ** 2)
    rmse = torch.sqrt(mse)
    if normalization == "range":
        denom = gt.max() - gt.min()
    elif normalization == "mean":
        denom = torch.mean(gt)
    elif normalization == "std":
        denom = torch.std(gt)
    elif normalization == "energy":
        denom = torch.sqrt(torch.sum(gt**2) / gt.numel())
    else:
        raise ValueError(
            "normalization must be one of: 'range', 'mean', 'std', 'energy'"
        )
    return rmse / (denom + 1e-8)


def calculate_metrics_on_mip(
    test_dataloader, model, device, mean, std, save_results=False
):
    """Calculate PSNR and SSIM for the test dataset and save slices to a .tiff file."""
    model.eval()
    psnr_sum = 0
    ssim_sum = 0
    lpips_sum = 0
    vif_sum = 0
    niqe_sum = 0
    piqe_sum = 0
    nrqm_sum = 0
    count = 0

    # output_dir = r"/nas/projects/Neuron_SR/MambaIR_NeuroConv/testresults/outputs"
    # GT_dir = r"/nas/projects/Neuron_SR/MambaIR_NeuroConv/testresults/GT"
    # os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
    # os.makedirs(GT_dir, exist_ok=True)  # 确保目录存在
    loss_fn_alex = LPIPS(net="alex").to(device)

    niqe = pyiqa.create_metric(
        "niqe",
        device=device,
        pretrained_model_path="/home/gsm/python/PYIQA/niqe_modelparameters.mat",
    )

    piqe = pyiqa.create_metric("piqe", device=device)

    nrqm = pyiqa.create_metric(
        "nrqm",
        device=device,
        pretrained_model_path="/home/gsm/python/PYIQA/nrqm_model.mat",
    )

    with torch.no_grad():
        for batch_idx, imgs in enumerate(test_dataloader):
            inputs, targets = imgs["lr"], imgs["hr"]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            outputs, targets = (
                (tensor_denormalize(outputs, mean, std) * 255)
                .type(torch.uint8)
                .type(torch.float32)
                / 255.0,
                (tensor_denormalize(targets, mean, std) * 255)
                .type(torch.uint8)
                .type(torch.float32)
                / 255.0,
            )

            data_range = 1.0

            outputs_mip = outputs.max(1, keepdim=True)
            targets_mip = targets.max(1, keepdim=True)
            for i in range(outputs.shape[0]):  # 遍历batch中的每个样本
                psnr_batch_sum = 0
                ssim_batch_sum = 0
                lpips_batch_sum = 0

                slice_count = outputs.shape[1]  # 获取切片数量

                for j in range(slice_count):  # 遍历每个切片
                    output_img = outputs[i, j]
                    target_img = targets[i, j]

                    lpips = loss_fn_alex(
                        preprocess_for_lpips_tensor(target_img),
                        preprocess_for_lpips_tensor(output_img),
                    )

                    target_img = target_img.cpu().numpy().squeeze().squeeze()
                    output_img = output_img.cpu().numpy().squeeze().squeeze()

                    # 计算当前切片的PSNR和SSIM
                    psnr = peak_signal_noise_ratio(
                        target_img,
                        output_img,
                        data_range=data_range,
                    )
                    # print(f"psnr: {psnr}")

                    ssim = structural_similarity(
                        target_img, output_img, data_range=data_range
                    )
                    # print(f"ssim: {ssim}")

                    psnr_batch_sum += psnr
                    ssim_batch_sum += ssim
                    lpips_batch_sum += lpips.item()
                    # # 将切片添加到列表中
                    # output_slices.append(output_img)
                    # target_slices.append(target_img)

                # 平均每个样本中所有切片的PSNR和SSIM
                psnr_avg_per_sample = psnr_batch_sum / slice_count
                ssim_avg_per_sample = ssim_batch_sum / slice_count
                lpips_avg_per_sample = lpips_batch_sum / slice_count

                psnr_sum += psnr_avg_per_sample
                ssim_sum += ssim_avg_per_sample
                lpips_sum += lpips_avg_per_sample

                vif_sum += vif_p(
                    outputs_mip[i],
                    targets_mip[i],
                    data_range=data_range,
                ).item()
                niqe_sum += niqe(outputs_mip[i]).item()
                piqe_sum += piqe(outputs_mip[i]).item()
                nrqm_sum += nrqm(outputs_mip[i]).item()

                count += 1

                print(
                    f"<{batch_idx * inputs.shape[0] + i + 1}> testing images finished."
                )

                # if save_results:
                #     # 将数据缩放到 [0, 255] 并转换为 uint8 类型
                #     outputs = (outputs * 255).astype(np.uint8)
                #     targets = (targets * 255).astype(np.uint8)

                #     # 生成文件名
                #     output_filename = f"output_{batch_idx * len(outputs) + i + 1}.tiff"
                #     output_path = os.path.join(output_dir, output_filename)

                #     GT_filename = f"GT_{batch_idx * len(outputs) + i + 1}.tiff"
                #     GT_path = os.path.join(GT_dir, GT_filename)

                #     # 使用 tifffile 保存三维数组为 .tiff 文件
                #     tf.imwrite(output_path, outputs)
                #     # 使用 tifffile 保存三维数组为 .tiff 文件
                #     tf.imwrite(GT_path, targets)

    model.train()

    return (
        psnr_sum / count,
        ssim_sum / count,
        lpips_sum / count,
        vif_sum / count,
        niqe_sum / count,
        piqe_sum / count,
        nrqm_sum / count,
    )


def calculate_psnr_ssim_lpips_gpu(
    test_dataloader, model, device, mean, std, save_results=False
):
    """Calculate PSNR and SSIM for the test dataset and save slices to a .tiff file."""
    model.eval()
    psnr_sum = 0
    ssim_sum = 0
    lpips_sum = 0
    count = 0

    # output_dir = r"/nas/projects/Neuron_SR/MambaIR_NeuroConv/testresults/outputs"
    # GT_dir = r"/nas/projects/Neuron_SR/MambaIR_NeuroConv/testresults/GT"
    # os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
    # os.makedirs(GT_dir, exist_ok=True)  # 确保目录存在
    loss_fn_alex = LPIPS(net="alex").to(device)

    with torch.no_grad():
        for batch_idx, imgs in enumerate(test_dataloader):
            inputs, targets = imgs["lr"], imgs["hr"]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            outputs, targets = (
                (tensor_denormalize(outputs, mean, std) * 255)
                .type(torch.uint8)
                .type(torch.float32)
                / 255.0,
                (tensor_denormalize(targets, mean, std) * 255)
                .type(torch.uint8)
                .type(torch.float32)
                / 255.0,
            )

            data_range = 1.0

            for i in range(outputs.shape[0]):  # 遍历batch中的每个样本
                psnr_batch_sum = 0
                ssim_batch_sum = 0
                lpips_batch_sum = 0
                slice_count = outputs.shape[1]  # 获取切片数量

                for j in range(slice_count):  # 遍历每个切片
                    output_img = outputs[i, j]
                    target_img = targets[i, j]

                    lpips = loss_fn_alex(
                        preprocess_for_lpips_tensor(target_img),
                        preprocess_for_lpips_tensor(output_img),
                    )

                    target_img = target_img.cpu().numpy().squeeze().squeeze()
                    output_img = output_img.cpu().numpy().squeeze().squeeze()

                    # 计算当前切片的PSNR和SSIM
                    psnr = peak_signal_noise_ratio(
                        target_img,
                        output_img,
                        data_range=data_range,
                    )
                    # print(f"psnr: {psnr}")

                    ssim = structural_similarity(
                        target_img, output_img, data_range=data_range
                    )
                    # print(f"ssim: {ssim}")

                    psnr_batch_sum += psnr
                    ssim_batch_sum += ssim
                    lpips_batch_sum += lpips.item()

                    # # 将切片添加到列表中
                    # output_slices.append(output_img)
                    # target_slices.append(target_img)

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

                # if save_results:
                #     # 将数据缩放到 [0, 255] 并转换为 uint8 类型
                #     outputs = (outputs * 255).astype(np.uint8)
                #     targets = (targets * 255).astype(np.uint8)

                #     # 生成文件名
                #     output_filename = f"output_{batch_idx * len(outputs) + i + 1}.tiff"
                #     output_path = os.path.join(output_dir, output_filename)

                #     GT_filename = f"GT_{batch_idx * len(outputs) + i + 1}.tiff"
                #     GT_path = os.path.join(GT_dir, GT_filename)

                #     # 使用 tifffile 保存三维数组为 .tiff 文件
                #     tf.imwrite(output_path, outputs)
                #     # 使用 tifffile 保存三维数组为 .tiff 文件
                #     tf.imwrite(GT_path, targets)

    model.train()

    return psnr_sum / count, ssim_sum / count, lpips_sum / count


def calculate_psnr_ssim_lpips(
    test_dataloader, model, device, mean, std, save_results=False
):
    """Calculate PSNR and SSIM for the test dataset and save slices to a .tiff file."""
    model.eval()
    psnr_sum = 0
    ssim_sum = 0
    lpips_sum = 0
    count = 0

    # output_dir = r"/nas/projects/Neuron_SR/MambaIR_NeuroConv/testresults/outputs"
    # GT_dir = r"/nas/projects/Neuron_SR/MambaIR_NeuroConv/testresults/GT"
    # # os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
    # # os.makedirs(GT_dir, exist_ok=True)  # 确保目录存在
    loss_fn_alex = LPIPS(net="alex")

    with torch.no_grad():
        for batch_idx, imgs in enumerate(test_dataloader):
            inputs, targets = imgs["lr"], imgs["hr"]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            outputs = outputs.detach().cpu().numpy()
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

                # # 创建一个列表用于存储每个样本的输出和目标切片
                # output_slices = []
                # target_slices = []

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
                    lpips_batch_sum += lpips.item()

                    # # 将切片添加到列表中
                    # output_slices.append(output_img)
                    # target_slices.append(target_img)

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

                # if save_results:
                #     # 将数据缩放到 [0, 255] 并转换为 uint8 类型
                #     outputs = (outputs * 255).astype(np.uint8)
                #     targets = (targets * 255).astype(np.uint8)

                #     # 生成文件名
                #     output_filename = f"output_{batch_idx * len(outputs) + i + 1}.tiff"
                #     output_path = os.path.join(output_dir, output_filename)

                #     GT_filename = f"GT_{batch_idx * len(outputs) + i + 1}.tiff"
                #     GT_path = os.path.join(GT_dir, GT_filename)

                #     # 使用 tifffile 保存三维数组为 .tiff 文件
                #     tf.imwrite(output_path, outputs)
                #     # 使用 tifffile 保存三维数组为 .tiff 文件
                #     tf.imwrite(GT_path, targets)

    model.train()

    return psnr_sum / count, ssim_sum / count, lpips_sum / count


def calculate_ref_metrics(
    test_dataloader, model, device, mean, std, save_results=False
):
    """Calculate PSNR and SSIM for the test dataset and save slices to a .tiff file."""
    model.eval()
    psnr_sum = 0
    ssim_sum = 0
    lpips_sum = 0
    vif_sum = 0
    nrmse_sum = 0
    ms_ssim_sum = 0
    iwpsnr_sum = 0
    gwpsnr_sum = 0
    dists_sum = 0
    fsim_sum = 0

    count = 0

    loss_fn_alex = LPIPS(net="alex").to(device)
    metric_dists = pyiqa.create_metric(
        "dists",
        device=device,
        pretrained_model_path="/home/gsm/python/PYIQA/DISTS_weights.pth",
    )
    metric_fsim = pyiqa.create_metric("fsim", device=device)

    with torch.no_grad():
        for batch_idx, imgs in enumerate(test_dataloader):
            inputs, targets = imgs["lr"], imgs["hr"]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)

            outputs, targets = (
                (tensor_denormalize(outputs, mean, std) * 255)
                .type(torch.uint8)
                .type(torch.float32)
                / 255.0,
                (tensor_denormalize(targets, mean, std) * 255)
                .type(torch.uint8)
                .type(torch.float32)
                / 255.0,
            )

            data_range = 1.0

            for i in range(outputs.shape[0]):  # 遍历batch中的每个样本
                psnr_sample_sum = 0
                ssim_sample_sum = 0
                lpips_sample_sum = 0
                vif_sample_sum = 0
                ms_ssim_sample_sum = 0
                iwpsnr_sample_sum = 0
                gwpsnr_sample_sum = 0
                dists_sample_sum = 0
                fsim_sample_sum = 0

                # print(f"outputs.shape:{outputs.shape}")
                slice_count = outputs.shape[1]  # 获取切片数量

                for j in range(slice_count):  # 遍历每个切片
                    output_img = outputs[i, j]
                    target_img = targets[i, j]

                    out_img_tensor = output_img.unsqueeze(0).unsqueeze(0)
                    target_img_tensor = target_img.unsqueeze(0).unsqueeze(0)

                    vif_score = vif_p(
                        out_img_tensor, target_img_tensor, data_range=data_range
                    )
                    ms_ssim_score = multi_scale_ssim(
                        out_img_tensor, target_img_tensor, data_range=data_range
                    )
                    iwpsnr_score = intensity_wpsnr(
                        out_img_tensor, target_img_tensor, data_range=data_range
                    )
                    gwpsnr_score = gradient_wpsnr(
                        out_img_tensor, target_img_tensor, data_range=data_range
                    )
                    dists_score = metric_dists(out_img_tensor, target_img_tensor)
                    fsim_score = metric_fsim(
                        out_img_tensor.repeat(1, 3, 1, 1),
                        target_img_tensor.repeat(1, 3, 1, 1),
                    )

                    lpips = loss_fn_alex(
                        preprocess_for_lpips_tensor(target_img),
                        preprocess_for_lpips_tensor(output_img),
                    )

                    target_img = target_img.cpu().numpy().squeeze().squeeze()
                    output_img = output_img.cpu().numpy().squeeze().squeeze()

                    # 计算当前切片的PSNR和SSIM
                    psnr = peak_signal_noise_ratio(
                        target_img,
                        output_img,
                        data_range=data_range,
                    )
                    # print(f"psnr: {psnr}")

                    ssim = structural_similarity(
                        target_img, output_img, data_range=data_range
                    )
                    # print(f"ssim: {ssim}")

                    psnr_sample_sum += psnr
                    ssim_sample_sum += ssim
                    lpips_sample_sum += lpips.item()
                    vif_sample_sum += vif_score.item()
                    ms_ssim_sample_sum += ms_ssim_score.item()
                    iwpsnr_sample_sum += iwpsnr_score.item()
                    gwpsnr_sample_sum += gwpsnr_score.item()
                    dists_sample_sum += dists_score.item()
                    fsim_sample_sum += fsim_score.item()

                # 平均每个样本中所有切片的PSNR和SSIM
                psnr_avg_per_sample = psnr_sample_sum / slice_count
                ssim_avg_per_sample = ssim_sample_sum / slice_count
                lpips_avg_per_sample = lpips_sample_sum / slice_count
                vif_avg_per_sample = vif_sample_sum / slice_count
                ms_ssim_avg_per_sample = ms_ssim_sample_sum / slice_count
                iwpsnr_avg_per_sample = iwpsnr_sample_sum / slice_count
                gwpsnr_avg_per_sample = gwpsnr_sample_sum / slice_count
                dists_avg_per_sample = dists_sample_sum / slice_count
                fsim_avg_per_sample = fsim_sample_sum / slice_count

                psnr_sum += psnr_avg_per_sample
                ssim_sum += ssim_avg_per_sample
                lpips_sum += lpips_avg_per_sample
                vif_sum += vif_avg_per_sample
                ms_ssim_sum += ms_ssim_avg_per_sample
                iwpsnr_sum += iwpsnr_avg_per_sample
                gwpsnr_sum += gwpsnr_avg_per_sample
                dists_sum += dists_avg_per_sample
                fsim_sum += fsim_avg_per_sample

                nrmse_sum += nrmse_torch(outputs[i], targets[i])

                count += 1

                print(
                    f"<{batch_idx * inputs.shape[0] + i + 1}> testing images finished."
                )

    # model.train()

    return (
        psnr_sum / count,
        ssim_sum / count,
        lpips_sum / count,
        vif_sum / count,
        nrmse_sum / count,
        ms_ssim_sum / count,
        iwpsnr_sum / count,
        gwpsnr_sum / count,
        dists_sum / count,
        fsim_sum / count,
    )
