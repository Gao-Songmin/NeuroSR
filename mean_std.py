import numpy as np
import glob
import tifffile as tf


def calculate_mean_std(root: str):
    files = sorted(glob.glob(root + "/*.*"))
    print(len(files))
    mean = []
    std = []
    for f in files:
        print(f)
        img = tf.imread(f)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        mean.append(img.mean())
        std.append(img.std())
    print("mean=", np.mean(mean), "std=", np.mean(std))


if __name__ == "__main__":
    dataset_dir = r"data/LR/train"
    print(f"dir: {dataset_dir}")
    calculate_mean_std(dataset_dir)
