import numpy as np
from numpy.typing import NDArray
import glob
import random
import os
import torch
from torch.utils.data import Dataset
from torch import Tensor
from torchvision import transforms
from PIL import Image
import tifffile as tf
from degradations import *


def denormalize(array: NDArray, mode: str, mean, std):
    array = array * std[mode] + mean[mode]
    array = (array * (2**8 - 1)).clip(0, 255).astype(np.uint8)
    return array


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


class ImageDataset_3D(Dataset):
    """For 3D LR-HR data pairs generation"""

    def __init__(
        self, root, hr_shape, mean, std, decay_range=(8, 8), sigmas=(0.2, 0.4, 0.2)
    ):
        super().__init__()
        self.hr_shape = hr_shape
        self.files = sorted(glob.glob(root + "/*.*"))
        self.mean = mean
        self.std = std
        self.decay_range = decay_range
        self.sigmas = sigmas

    def lr_transforms(self, volume: NDArray) -> Tensor:
        volume = volume.squeeze()
        volume = CometExpTail3D(decay_range=self.decay_range)(volume)
        volume = max_downsampleH_3d(volume, stride=4)
        volume = gaussian_filter(volume, sigma=self.sigmas)
        volume = volume.astype(np.float32) / 255.0
        volume = (volume - self.mean["lr"]) / self.std["lr"]
        volume = torch.tensor(volume)
        return volume

    def hr_transforms(self, volume: NDArray) -> Tensor:
        volume = volume.squeeze()
        volume = resize_3d(volume, new_size=self.hr_shape, order=3)
        volume = volume.astype(np.float32) / 255.0
        volume = (volume - self.mean["hr"]) / self.std["hr"]
        volume = torch.tensor(volume)
        return volume

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img_name = os.path.basename(img_path).split(".")[0]
        volume = tf.imread(img_path)
        imgs_lr = self.lr_transforms(volume)
        imgs_hr = self.hr_transforms(volume)
        return {"lr": imgs_lr, "hr": imgs_hr, "fn": img_name}

    def __len__(self):
        return len(self.files)


class PairedSRDataset(Dataset):
    """For training on 3D images"""

    def __init__(self, lr_root, hr_root, mean, std):
        super().__init__()
        lr_files = sorted(glob.glob(lr_root + "/*.tif*"))
        hr_files = sorted(glob.glob(hr_root + "/*.tif*"))
        print(f"lr, hr dataset length: {len(lr_files), len(hr_files)}")
        assert len(lr_files) == len(hr_files), (
            "[Error] lr_files and hr_files are not paired."
        )
        self.paired_files = list(zip(lr_files, hr_files))
        self.mean = mean
        self.std = std

    def __getitem__(self, index):
        f_paths = self.paired_files[index % len(self.paired_files)]
        f_name = f_paths[0].split("/")[-1].split(".")[0]
        img_lr = tf.imread(f_paths[0])
        img_hr = tf.imread(f_paths[1]).squeeze()

        # normalize
        img_lr = img_lr.astype(np.float32) / 255.0
        img_hr = img_hr.astype(np.float32) / 255.0

        # standardize
        img_lr = (img_lr - self.mean["lr"]) / self.std["lr"]
        img_hr = (img_hr - self.mean["hr"]) / self.std["hr"]

        return {"lr": img_lr, "hr": img_hr, "fn": f_name}

    def __len__(self):
        return len(self.paired_files)


class TestDataset_realSR(Dataset):
    """For testing on real SR task of 3D images"""

    def __init__(self, root, range, mean, std):
        super().__init__()
        self.files = glob.glob(root + "/*.tif*")
        self.mean = mean
        self.std = std
        self.clip_range = range

    def clip(self, volume: NDArray, axis="y", a=96, b=160):
        volume = volume.squeeze()
        assert volume.ndim == 3, "The dimension of volume input need to be 3."
        if axis == "y":
            volume = volume[:, a:b, :]
        elif axis == "x":
            volume = volume[:, :, a:b]
            volume = volume.transpose(0, 2, 1)
        else:
            raise ValueError("axis need to be 'y' or 'x'.")

        volume = volume.transpose(1, 0, 2)
        return volume[:, ::-1, :]

    def __getitem__(self, index):
        file = self.files[index]
        img = tf.imread(file)
        img_name = file.split("/")[-1].split(".")[0]
        img = self.clip(img, "y", *self.clip_range)
        # img = np.power(img, 1.2)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean["lr"]) / self.std["lr"]

        return torch.tensor(img), img_name

    def __len__(self):
        return len(self.files)


class TestDatasetForConcat(Dataset):
    def __init__(self, root, depth, mean, std):
        self.files = glob.glob(root + "/*.tif*")
        self.depth = int(depth)
        self.mean = mean
        self.std = std

    def clip(self, volume: NDArray, a=96, b=160):
        volume = volume.squeeze()
        assert volume.ndim == 3, "The dimension of volume input need to be 3."
        volume = volume[:, a:b, :]
        volume = volume.transpose(1, 0, 2)
        return volume[:, ::-1, :]

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        img_name = os.path.basename(path).split(".")[0]
        img = tf.imread(path)
        assert img.shape[1] % self.depth == 0, (
            "Augment depth need to be divided by dim-2 of img."
        )
        iter_num = int(img.shape[1] / self.depth)
        patches = []
        for i in range(iter_num):
            patch = self.clip(img, self.depth * i, self.depth * (i + 1))
            patch = patch.astype(np.float32) / 255.0
            patch = (patch - self.mean["lr"]) / self.std["lr"]
            patches.append(patch)

        return patches, img_name

    def __len__(self):
        return len(self.files)


class ImageDataset_2D(Dataset):
    """For training on 2D images"""

    def __init__(
        self,
        root,
        hr_shape,
        mean,
        std,
        augmentation=True,
        dir_range=(-1, 1),
        decay_range=(2, 10),
        sigmas=(0.4, 0.2),
        threshold=40,
    ):
        super().__init__()
        _, hr_height, hr_width = hr_shape

        self.augmentation = augmentation

        self.lr_transform = transforms.Compose(
            [
                DirectionExpTail2D(
                    decay_range=decay_range,
                    direction_range=dir_range,
                    threshold=threshold,
                    prob=1,
                ),
                MaxDownSampleH(stride=4),
                GaussianFilter(sigmas),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.augment_transform = transforms.Compose(
            [
                RandomRotation([0, 90, 180, 270]),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.tif*"))
        print(f"############length of dataset:{len(self.files)}##############")

    def __getitem__(self, index):
        file_path = self.files[index % len(self.files)]
        img_name = file_path.split("/")[-1].split(".")[0]
        img = Image.open(file_path)

        # data augmentation
        if self.augmentation:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            img = self.augment_transform(img)

        # LR-HR data pair
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr, "fn": img_name}

    def __len__(self):
        return len(self.files)


class LRImageDataset(Dataset):
    """For testing on real SR task of 2D images"""

    def __init__(self, root, mean, std):
        super().__init__()
        self.files = sorted(glob.glob(root + "/*.*"))
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    def __getitem__(self, index):
        img_lr = Image.open(self.files[index])
        img_lr = self.transforms(img_lr)
        img_name = self.files[index].split("/")[-1].split(".")[0]
        return img_lr, img_name

    def __len__(self):
        return len(self.files)
