import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, convolve
from torch import Tensor
import torch
from PIL import Image


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


if __name__ == "__main__":
    import os, random
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import tifffile as tf
    from torch.utils.data import DataLoader
    from datasets import SimpleDataset

    def lr_transforms(volume: np.ndarray):
        threshold = random.randint(5, 75)
        angle = np.random.uniform(low=np.pi / 6, high=np.pi / 6)
        volume = MaskedEdgeCometExpTail_3D(
            length=31, decay_range=(6, 10), threshold=threshold, spread_angle=angle
        )(volume)
        volume = max_downsampleH_3d(volume, stride=4)
        volume = gaussian_filter(volume, sigma=(0.2, 0.4, 0.2))
        return volume

    def process_one_image_thread(vol, f_name: str, save_dir: str):
        if isinstance(vol, torch.Tensor):
            vol = vol.detach().cpu().numpy()
        vol = np.asarray(vol).squeeze()

        vol_lr = lr_transforms(vol)
        vol_lr = np.clip(vol_lr, 0, 255).astype(np.uint8)

        out_path = os.path.join(save_dir, f_name)

        with tf.TiffWriter(out_path, bigtiff=True) as tw:
            tw.write(vol_lr)
        return f_name

    hr_dir = "data/HR/test"
    lr_dir = "data/LR/test"
    os.makedirs(lr_dir, exist_ok=True)

    dataset = SimpleDataset(hr_dir)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )

    max_workers = 8
    max_inflight = 64

    futures = set()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for volumes, f_names in dataloader:
            for vol, f_name in zip(volumes, f_names):
                fut = ex.submit(process_one_image_thread, vol, f_name, lr_dir)
                futures.add(fut)
                if len(futures) >= max_inflight:
                    done = {f for f in futures if f.done()}
                    for f in done:
                        print("[Done]", f.result())
                    futures -= done

        for f in as_completed(list(futures)):
            print("[Done]", f.result())
