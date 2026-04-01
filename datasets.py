import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset
from torch import Tensor
import glob
import tifffile as tf
from natsort import natsorted
from degradation import *
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, root, hr_shape, mean, std):
        super().__init__()
        self.hr_shape = hr_shape
        self.files = sorted(glob.glob(root + "/*.*"))
        self.mean = mean
        self.std = std

    def lr_transforms(self, volume: NDArray) -> Tensor:
        volume = volume.squeeze()
        volume = MaskedEdgeCometExpTail_3D(decay_range=(2, 10))(volume)
        volume = max_downsampleH_3d(volume, stride=4)
        volume = gaussian_filter(volume, sigma=(0.2, 0.4, 0.2))
        volume = volume.astype(np.float32) / 255.0
        # volume /= volume.max()
        # volume = (volume - volume.min()) / (volume.max() - volume.min())
        volume = (volume - self.mean["lr"]) / self.std["lr"]
        volume = torch.tensor(volume)
        return volume

    def hr_transforms(self, volume: NDArray) -> Tensor:
        volume = volume.squeeze()
        volume = resize3D(volume, new_size=self.hr_shape, order=3)
        volume = volume.astype(np.float32) / 255.0
        # volume /= volume.max()
        # volume = (volume - volume.min()) / (volume.max() - volume.min())
        volume = (volume - self.mean["hr"]) / self.std["hr"]
        volume = torch.tensor(volume)
        return volume

    def __getitem__(self, index):
        volume = tf.imread(self.files[index % len(self.files)])
        imgs_lr = self.lr_transforms(volume)
        imgs_hr = self.hr_transforms(volume)
        return {"lr": imgs_lr, "hr": imgs_hr}

    def __len__(self):
        return len(self.files)


class ImageDataset_2D(Dataset):
    def __init__(self, root, hr_shape, augmentation=True):
        super().__init__()
        _, hr_height, hr_width = hr_shape

        self.augmentation = augmentation

        self.lr_transform = transforms.Compose(
            [
                # KernelDownSample(kernel_dir=kernel_dir, scale_factor=(0.25, 1)),
                OneSidedMaskDirectionExpBlur(
                    decay_range=(2, 10), direction_range=(-1, 1), threshold=40, prob=1
                ),
                MaxDownSampleH(stride=4),
                isoGaussianFilter((0.4, 0.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean["lr"], std["lr"]),
            ]
        )

        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean["hr"], std["hr"]),
            ]
        )

        self.augment_transform = transforms.Compose(
            [
                RandomRotation([0, 90, 180, 270]),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        file_path = self.files[index % len(self.files)]
        img_name = file_path.split("\\")[-1]
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


class PairedSRDataset(Dataset):
    def __init__(self, lr_root, hr_root, mean, std):
        super().__init__()
        lr_files = natsorted(glob.glob(lr_root + "/*.tif*"))
        hr_files = natsorted(glob.glob(hr_root + "/*.tif*"))
        assert len(lr_files) == len(hr_files), (
            f"[Error] lr_files({len(lr_files)}) and hr_files({len(hr_files)}) are not paired."
        )
        self.paired_files = list(zip(lr_files, hr_files))
        self.mean = mean
        self.std = std

    def __getitem__(self, index):
        f_paths = self.paired_files[index % len(self.paired_files)]
        f_name = f_paths[0].split("/")[-1].split(".")[0]
        img_lr = tf.imread(f_paths[0]).squeeze()
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


class TestDataset(Dataset):
    def __init__(self, root, clip_range, mean, std):
        super().__init__()
        self.files = glob.glob(root + "/*.tif*")
        self.clip_range = clip_range
        self.mean = mean
        self.std = std

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
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean["lr"]) / self.std["lr"]

        return torch.tensor(img), img_name

    def __len__(self):
        return len(self.files)


class TestDatasetForConcat(Dataset):
    def __init__(self, root, depth, mean, std):
        super().__init__()
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


class TestDataset9Cube(Dataset):
    def __init__(self, root, cube_size, depth, mean, std):
        super().__init__()
        self.files = glob.glob(root + "/*.tif*")
        self.cube_size = cube_size
        self.depth = depth
        self.mean = mean
        self.std = std

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        img = tf.imread(path)
        step_x, step_y, step_z = self.cube_size

    def __len__(self):
        return len(self.files)


class SimpleDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        f = self.files[index % len(self.files)]
        f_name = f.split("/")[-1].split(".")[0]
        img = tf.imread(f)

        if img.ndim == 3:
            img = img[..., np.newaxis]

        img = img.transpose(3, 0, 1, 2)
        return img, f_name

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    import tifffile as tf
    from torch.utils.data import DataLoader

    dataset_dir = r"E:\data\Peng1741_soma_blocks\train"
    hr_shape = (64, 256, 256)
    dataset = ImageDataset_3D(dataset_dir, hr_shape)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch_i, imgs in enumerate(dataloader):
        imgs_lr = imgs["lr"].to(device)
        imgs_hr = imgs["hr"].to(device)

        imgs_lr_np = imgs_lr.cpu().numpy()
        imgs_hr_np = imgs_hr.cpu().numpy()

        batch_size = imgs_lr_np.shape[0]

        save_dir = r"./trial/imgs"

        for i in range(batch_size):
            img_lr_np = denormalize(imgs_lr_np[i], mode="lr")
            img_hr_np = denormalize(imgs_hr_np[i], mode="hr")
            # img_lr_np = imgs_lr_np[i]
            # img_hr_np = imgs_hr_np[i]

            print(
                "lr, max, min, avg, std:",
                img_lr_np.max(),
                img_lr_np.min(),
                img_lr_np.mean(),
                img_lr_np.std(),
            )
            print(
                "hr, max, min, avg, std:",
                img_lr_np.max(),
                img_hr_np.min(),
                img_hr_np.mean(),
                img_hr_np.std(),
            )

            tf.imwrite(
                save_dir + f"/lr_{batch_i}_{i}.tif",
                resize3D(img_lr_np, (64, 256, 256), order=3),
            )
            tf.imwrite(save_dir + f"/hr_{batch_i}_{i}.tif", img_hr_np)

        if batch_i >= 10:
            break
