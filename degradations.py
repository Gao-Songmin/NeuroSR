import numpy as np
from numpy.typing import NDArray
from PIL import Image
import tifffile as tf
from scipy.ndimage import gaussian_filter, zoom, convolve
import os
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import glob


##### Degradations for 2D images. #####


class DirectionExpTail2D:
    def __init__(
        self,
        pad_axis=0,
        length=31,
        decay_range=(4, 4),
        direction_range=(-1, 1),
        prob=1,
        threshold=120,
    ):
        self.pad_axis = pad_axis
        self.length = length
        self.decay_range = decay_range
        self.direction_range = direction_range
        self.prob = prob
        self.threshold = threshold  # 高亮判定阈值（0–255）

    def __call__(self, img: Image.Image):
        if np.random.rand() > self.prob:
            return img

        decay = np.random.uniform(*self.decay_range)
        direction = (np.random.uniform(*self.direction_range), -1)

        kernel_2D = self.generate_direction_kernel(
            self.length,
            decay,
            direction=direction,
        )

        img_contrast = np.asarray(NonlinearScatter().__call__(img))

        img_np = np.asarray(img).astype(np.float32)

        mask = (img_contrast > self.threshold).astype(np.float32)
        blurred = convolve(img_np, weights=kernel_2D, mode="nearest")
        mask_blurred = convolve(mask, weights=kernel_2D, mode="nearest")

        mask_final = mask_blurred - mask

        result = img_np.copy()
        result[mask_final > 0] = blurred[mask_final > 0]

        result = np.clip(result, 0, 255).astype(np.uint8)

        return Image.fromarray(result)

    def generate_direction_kernel(self, length, decay, direction):
        dx, dy = direction
        norm = np.sqrt(dx**2 + dy**2)
        dx /= norm
        dy /= norm

        coords = np.linspace(0, length - 1, length)
        kernel = np.exp(-coords / decay)
        kernel /= kernel.sum()

        kernel_2d = np.zeros((length, length))
        for i in range(length):
            x = int(round(i * dx))
            y = int(round(i * dy))
            cx = length // 2 + x
            cy = length // 2 + y
            if 0 <= cy < length and 0 <= cx < length:
                kernel_2d[cy, cx] = kernel[i]

        kernel_2d /= kernel_2d.sum()
        return kernel_2d


class NonlinearScatter:
    def __init__(self, gamma=1.5, sigma=(1, 1), prob=1):
        self.gamma = gamma
        self.sigma = sigma
        self.prob = prob

    def __call__(self, img: Image.Image):
        if np.random.rand() > self.prob:
            return img
        img_np = np.asarray(img)
        img_np = img_np / 255.0
        img_np = np.power(img_np, self.gamma)
        img_np = gaussian_filter(img_np, self.sigma)
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)


class GaussianFilter:
    def __init__(self, sigma: tuple[float, float]):
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image:
        img_np = np.asarray(img)
        blurred_np = gaussian_filter(img_np, sigma=self.sigma)
        return Image.fromarray(blurred_np)


class DownSample2D:
    def __init__(self, strides: tuple[int, int]):
        self.strides = strides

    def __call__(self, img: Image.Image) -> Image:
        img = np.asarray(img)
        img_lr = img[:: self.strides[0], :: self.strides[1], ...]  # (H, W, C)
        return Image.fromarray(img_lr)


class MeanDownSampleH:
    def __init__(self, stride: int = 4):
        self.stride = stride

    def __call__(self, img: Image.Image) -> Image:
        img_np = np.asarray(img).astype(np.float32)
        h, w = img_np.shape[:2]
        c = 1 if img_np.ndim == 2 else img_np.shape[2]

        new_h = h // self.stride * self.stride
        img_np = img_np[:new_h]
        if c == 1:
            img_np = img_np.reshape(new_h // self.stride, self.stride, w)
            down_sampled = img_np.mean(1)
        else:
            img_np = img_np.reshape(new_h // self.stride, self.stride, w, c)
            down_sampled = img_np.mean(1)

        down_sampled = np.uint8(np.clip(down_sampled, 0, 255))

        return Image.fromarray(down_sampled)


class MaxDownSampleH:
    def __init__(self, stride: int = 4):
        self.stride = stride

    def __call__(self, img: Image.Image) -> Image:
        img_np = np.asarray(img).astype(np.float32)
        h, w = img_np.shape[:2]
        c = 1 if img_np.ndim == 2 else img_np.shape[2]

        new_h = h // self.stride * self.stride
        img_np = img_np[:new_h]
        if c == 1:
            img_np = img_np.reshape(new_h // self.stride, self.stride, w)
            down_sampled = img_np.max(1)
        else:
            img_np = img_np.reshape(new_h // self.stride, self.stride, w, c)
            down_sampled = img_np.max(1)

        down_sampled = np.uint8(np.clip(down_sampled, 0, 255))

        return Image.fromarray(down_sampled)


##### Degradations for 3D images. #####


class CometExpTail3D:
    """The exponential trailing blur for HR_image degradation along x/y axis."""

    def __init__(
        self,
        length=31,
        decay_range=(8, 8),
        direction=(0, 1, 0),
        spread_angle=np.pi / 4,
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

        return result

    def generate_comet_kernel_3d(self, length, direction, decay) -> NDArray:
        dx, dy, dz = direction
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        dx, dy, dz = dx / norm, dy / norm, dz / norm

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


def resize_3d(volume, new_size: tuple[int, int, int], order: int) -> NDArray:
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


class SimpleDataset(Dataset):
    """For volumetric LR-HR data pair generation"""

    def __init__(self, root):
        super().__init__()
        files = sorted(glob.glob(root + "/*.*"))
        self.files = files

    def __getitem__(self, index):
        f = self.files[index % len(self.files)]
        f_name = f.split("/")[-1]
        img = tf.imread(f)
        return img, f_name

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    """Generate volumetric LR-HR data pairs"""
    import glob
    import tifffile as tf
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    def lr_transforms(volume: NDArray):
        volume = CometExpTail3D(
            length=31, decay_range=(8, 8), threshold=40, spread_angle=np.pi / 4
        ).__call__(volume)
        volume = max_downsampleH_3d(volume, stride=4)
        volume = gaussian_filter(volume, sigma=(0.2, 0.4, 0.2))
        return volume

    def process_one_image(
        vol: NDArray,
        f_name: str,
        save_dir: str,
    ):
        vol = vol.squeeze()
        vol_lr = lr_transforms(vol)
        vol_lr = np.clip(vol_lr, 0, 255).astype(np.uint8)
        tf.imwrite(os.path.join(save_dir, f_name), vol_lr)
        return f_name

    hr_dir = r"Neuronal_Image_Dataset\HR_images\axons\train"
    lr_dir = r"Neuronal_Image_Dataset\LR_images\axons\train"
    os.makedirs(lr_dir, exist_ok=True)

    dataset = SimpleDataset(hr_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    tasks = []

    for volumes, f_names in dataloader:
        for vol, f_name in zip(volumes, f_names):
            tasks.append((vol, f_name, lr_dir))

    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_one_image, *args) for args in tasks]
        for future in as_completed(futures):
            fname = future.result()
            print(f"[Done] {fname}")
