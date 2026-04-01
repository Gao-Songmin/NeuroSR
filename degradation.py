import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, convolve
from torch import Tensor
import torch
import glob
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MaskedEdgeCometExpTail3D_GPU:
    def __init__(
        self,
        length=31,
        decay_range=(2, 10),
        direction=(0, 1, 0),
        spread_angle=np.pi / 6,
        threshold=40,
        prob=1,
        device=torch.device("cuda"),
    ):
        self.length = length
        self.decay_range = decay_range
        self.direction = direction
        self.spread_angle = spread_angle
        self.threshold = threshold
        self.prob = prob
        self.device = device

    def __call__(self, volume: NDArray) -> NDArray:
        if np.random.rand() > self.prob:
            return volume

        volume = volume.astype(np.float32)

        vol_contrast = NonlinearScatter(gamma=2, sigma=(1, 1, 1)).__call__(volume)

        vol_contrast = (
            torch.tensor(vol_contrast).unsqueeze(0).unsqueeze(0).to(self.device)
        )
        volume = torch.tensor(volume).unsqueeze(0).unsqueeze(0).to(self.device)
        print(f"vol_contrast.shape: {vol_contrast.shape}")

        decay = np.random.uniform(*self.decay_range)
        direction = self.direction
        kernel_3D = self.generate_comet_kernel_3d(
            length=self.length, decay=decay, direction=direction
        )

        mask = (vol_contrast > self.threshold).type(torch.float32)

        vol_blurred = F.conv3d(
            volume, weight=kernel_3D, stride=1, padding=(self.length - 1) // 2
        )
        mask_blurred = F.conv3d(
            mask, weight=kernel_3D, stride=1, padding=(self.length - 1) // 2
        )

        mask_final = mask_blurred - mask

        result = volume.clone()
        result[mask_final > 0] = vol_blurred[mask_final > 0]
        result = result.detach().cpu().numpy().squeeze().squeeze()
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def generate_comet_kernel_3d(self, length, direction, decay) -> NDArray:
        dx, dy, dz = direction
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        dx, dy, dz = dx / norm, dy / norm, dz / norm

        # coords = np.arange(length)
        kernel = np.zeros((length, length, length), dtype=np.float32)
        center = length // 2

        for i in range(length):
            for j in range(length):
                for k in range(length):
                    vec = np.array([i - center, j - center, k - center])
                    dist = np.linalg.norm(vec)
                    if dist == 0:
                        continue
                    vec_norm = vec / dist
                    angle = np.arccos(np.clip(np.dot(vec_norm, [dx, dy, dz]), -1, 1))
                    if angle < self.spread_angle:
                        weight = np.exp(-dist / decay)
                        kernel[i, j, k] = weight

        kernel /= kernel.sum()
        kernel_tensor = (
            torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).to(self.device)
        )
        return kernel_tensor


class MaskedEdgeCometExpTail_3D:
    def __init__(
        self,
        length=31,
        decay_range=(2, 10),
        direction=(0, 1, 0),
        spread_angle=np.pi / 6,
        threshold=40,
        prob=1,
    ):
        self.length = length
        self.decay_range = decay_range
        self.direction = direction
        self.spread_angle = spread_angle
        self.threshold = threshold
        self.prob = prob

    def __call__(self, volume: NDArray) -> NDArray:
        if isinstance(volume, torch.Tensor):
            volume = volume.detach().cpu().numpy()

        if volume.ndim != 3:
            raise ValueError("The dimension number of the volume is expected to be 3.")

        if np.random.rand() > self.prob:
            return volume

        decay = np.random.uniform(*self.decay_range)
        direction = self.direction
        kernel_3D = self.generate_comet_kernel_3d(
            length=self.length, decay=decay, direction=direction
        )

        vol_contrast = NonlinearScatter(gamma=2, sigma=(1, 1, 1)).__call__(volume)
        vol_np = volume.astype(np.float32)

        mask = (vol_contrast > self.threshold).astype(np.float32)

        vol_blurred = convolve(vol_np, weights=kernel_3D, mode="nearest")
        mask_blurred = convolve(mask, weights=kernel_3D, mode="nearest")

        mask_final = mask_blurred - mask

        result = vol_np.copy()
        result[mask_final > 0] = vol_blurred[mask_final > 0]

        # result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def generate_comet_kernel_3d(self, length, direction, decay) -> NDArray:
        dx, dy, dz = direction
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        dx, dy, dz = dx / norm, dy / norm, dz / norm

        # coords = np.arange(length)
        kernel = np.zeros((length, length, length), dtype=np.float32)
        center = length // 2

        for i in range(length):
            for j in range(length):
                for k in range(length):
                    vec = np.array([i - center, j - center, k - center])
                    dist = np.linalg.norm(vec)
                    if dist == 0:
                        continue
                    vec_norm = vec / dist
                    angle = np.arccos(np.clip(np.dot(vec_norm, [dx, dy, dz]), -1, 1))
                    if angle < self.spread_angle:
                        weight = np.exp(-dist / decay)
                        kernel[i, j, k] = weight

        kernel /= kernel.sum()
        return kernel


class NonlinearScatter:
    def __init__(self, gamma=1.5, sigma=(1, 1), prob=1):
        self.gamma = gamma
        self.sigma = sigma
        self.prob = prob

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        img_np = np.asarray(img)
        img_np = img_np / 255.0
        # gamma = np.random.uniform(*self.gamma_range)
        img_np = np.power(img_np, self.gamma)
        img_np = gaussian_filter(img_np, self.sigma)
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        if len(self.sigma) == 2:
            return Image.fromarray(img_np)
        else:
            return img_np


def mean_downsampleH_3d(volume, stride: int = 4):
    if isinstance(volume, Tensor):
        volume = volume.cpu().numpy()
    volume = volume.astype(np.float32)
    d, h, w = volume.shape[:3]
    c = 1 if volume.ndim == 3 else volume.shape[3]

    new_h = h // stride * stride
    volume = volume[:, :new_h, ...]

    if c == 1:
        volume = volume.reshape(d, new_h // stride, stride, w)
        down_sampled = volume.mean(2)
    else:
        volume = volume.reshape(d, new_h // stride, stride, w, c)
        down_sampled = volume.mean(2)

    return down_sampled


def max_downsampleH_3d(volume, stride: int = 4):
    if isinstance(volume, Tensor):
        volume = volume.cpu().numpy()

    d, h, w = volume.shape[:3]
    c = 1 if volume.ndim == 3 else volume.shape[3]

    new_h = h // stride * stride
    volume = volume[:, :new_h, ...]

    if c == 1:
        volume = volume.reshape(d, new_h // stride, stride, w)
        down_sampled = volume.max(2)
    else:
        volume = volume.reshape(d, new_h // stride, stride, w, c)
        down_sampled = volume.max(2)

    return down_sampled


class SimpleDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        f = self.files[index]
        f_name = f.split("/")[-1]
        img = tf.imread(f)
        return img, f_name

    def __len__(self):
        return len(self.files)


# if __name__ == "__main__":
#     import glob
#     import tifffile as tf
#     import os
#     from concurrent.futures import ProcessPoolExecutor, as_completed
#     import random
#     import time
#     import multiprocessing

#     def lr_transforms(volume: NDArray):
#         threshold = random.randint(5, 75)
#         angle = np.random.uniform(low=np.pi / 6, high=np.pi / 6)
#         volume = MaskedEdgeCometExpTail_3D(
#             length=31, decay_range=(6, 10), threshold=threshold, spread_angle=angle
#         ).__call__(volume)
#         volume = max_downsampleH_3d(volume, stride=4)
#         volume = gaussian_filter(volume, sigma=(0.2, 0.4, 0.2))
#         return volume

#     def process_one_image(
#         vol: NDArray,
#         f_name: str,
#         save_dir: str,
#     ):
#         vol = vol.squeeze()
#         vol_lr = lr_transforms(vol)
#         vol_lr = np.clip(vol_lr, 0, 255).astype(np.uint8)
#         tf.imwrite(os.path.join(save_dir, f_name), vol_lr)
#         return f_name

#     hr_dir = "/home/gsm/data/Neuron_SR/Human_soma_blocks/train"
#     lr_dir = "/home/gsm/data/Neuron_SR/Human_soma_blocks_comet_tailed/decay_6,10_over5,75_0.17pi/train"

#     os.makedirs(lr_dir, exist_ok=True)

#     dataset = SimpleDataset(hr_dir)
#     dataloader = DataLoader(
#         dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
#     )

# tasks = []
# for volumes, f_names in dataloader:
#     for vol, f_name in zip(volumes, f_names):
#         tasks.append((vol, f_name, lr_dir))

# with ProcessPoolExecutor(max_workers=40) as executor:
#     futures = [executor.submit(process_one_image, *args) for args in tasks]
#     for future in as_completed(futures):
#         fname = future.result()
#         print(f"[Done] {fname}")

# i = 0
# for imgs, f_names in dataloader:
#     for img, f_name in zip(imgs, f_names):
#         f_name = process_one_image(img, f_name, lr_dir)
#         print(f"[Done] {f_name}")

#     i += 1
#     if i > 10:
#         break


if __name__ == "__main__":
    import os, random
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import numpy as np
    import tifffile as tf
    from scipy.ndimage import gaussian_filter
    import torch
    from torch.utils.data import DataLoader
    # 你的自定义：MaskedEdgeCometExpTail_3D, max_downsampleH_3d, SimpleDataset

    def lr_transforms(volume: np.ndarray):
        threshold = random.randint(5,75)
        angle = np.random.uniform(low=np.pi / 6, high=np.pi / 6)  # 你原来的设定
        volume = MaskedEdgeCometExpTail_3D(
            length=31, decay_range=(6, 10), threshold=threshold, spread_angle=angle
        )(volume)
        volume = max_downsampleH_3d(volume, stride=4)
        volume = gaussian_filter(volume, sigma=(0.2, 0.8, 0.2))
        return volume

    def process_one_image_thread(vol, f_name: str, save_dir: str):
        if isinstance(vol, torch.Tensor):
            vol = vol.detach().cpu().numpy()
        vol = np.asarray(vol).squeeze()

        vol_lr = lr_transforms(vol)
        vol_lr = np.clip(vol_lr, 0, 255).astype(np.uint8)

        out_path = os.path.join(save_dir, f_name)
        # 用 TiffWriter 上下文，保证句柄立即关闭
        with tf.TiffWriter(out_path, bigtiff=True) as tw:
            tw.write(vol_lr)
        return f_name

    # hr_dir = "/home/gsm/data/Neuron_SR/Human_soma_blocks/train"
    # lr_dir = "/home/gsm/data/Neuron_SR/Human_soma_blocks_comet_tailed/decay_6,10_over5,75_0.17pi/train"
    hr_dir = "/home/gsm/data/Neuron_SR/soma_and_terminal_blocks/test"
    lr_dir = "/home/gsm/data/Neuron_SR/soma_and_terminal_blocks_comet_tailed/decay_6,10_over5,75_0.17pi_MP_Gaus0.8,0.2/test"
    os.makedirs(lr_dir, exist_ok=True)

    dataset = SimpleDataset(hr_dir)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )

    # 线程数别太大；I/O 为主建议 4~8，看磁盘/CPU 决定
    max_workers = 8
    # 控制在飞任务数，避免占用过多内存/FD
    max_inflight = 64

    futures = set()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for volumes, f_names in dataloader:
            for vol, f_name in zip(volumes, f_names):
                fut = ex.submit(process_one_image_thread, vol, f_name, lr_dir)
                futures.add(fut)
                # 进行节流：在飞任务超过阈值时，等一批完成
                if len(futures) >= max_inflight:
                    done = {f for f in futures if f.done()}
                    for f in done:
                        print("[Done]", f.result())
                    futures -= done

        # 收尾：把剩下的都取完
        for f in as_completed(list(futures)):
            print("[Done]", f.result())
