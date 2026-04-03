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
    

class SimpleDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.files = sorted(glob.glob(root + "/*.tif*"))

    def __getitem__(self, index):
        f = self.files[index % len(self.files)]
        f_name = f.split("/")[-1]
        img = tf.imread(f)
        return img, f_name

    def __len__(self):
        return len(self.files)
