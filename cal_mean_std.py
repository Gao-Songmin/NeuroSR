import numpy as np
import glob
import tifffile as tf
from degradations import *
from scipy.ndimage import gaussian_filter


def calculate_hr_mean_std(root: str):
    files = sorted(glob.glob(root + "/*.*"))
    mean = []
    std = []
    for f in files:
        img = tf.imread(f)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        mean.append(img.mean())
        std.append(img.std())
    print("hr_mean=", np.mean(mean), "hr_std=", np.mean(std))


def calculate_lr_mean_std(root: str, stride=4, sigma=(0.2, 0.4, 0.2)):
    files = sorted(glob.glob(root + "/*.*"))
    mean = []
    std = []
    for f in files:
        img = tf.imread(f)
        img = max_downsampleH_3d(img, stride=stride)
        img = gaussian_filter(img, sigma=sigma)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        mean.append(img.mean())
        std.append(img.std())
    print("lr_mean=", np.mean(mean), "lr_std=", np.mean(std))


if __name__ == "__main__":
    dataset_dir = r"Neuronal_Image_Dataset\HR_images\axons\train"
    calculate_hr_mean_std(dataset_dir)
    # calculate_lr_mean_std(dataset_dir)
