from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from skimage.measure import label
from tifffile import imread
from torch.utils.data import Dataset

from skimage.segmentation import relabel_sequential


class Nuclei3DDataset(Dataset):
    def __init__(self, raw_dir: Path, seg_dir: Path, augment: bool = False):
        self.augment = augment
        raw_files = sorted(raw_dir.glob("*.tif"))
        self.images, self.labels = [], []
        for raw_file in raw_files:
            seg_file = seg_dir / raw_file.name
            self.images.append(raw_file)
            self.labels.append(seg_file)

    def __len__(self):
        if self.augment:
            return len(self.images) * 4
        else:
            return len(self.images)

    def augment_input(self, img):
        img = img + np.random.normal(0, 0.05)
        img = img + np.random.normal(0.0, 0.01, img.shape).astype(np.float32)
        return img.astype(np.float32)

    def crop(self, img, seg):
        def crop_slice(shape, crop_size):
            z_start = np.random.randint(0, shape[0] - crop_size[0])
            y_start = np.random.randint(0, shape[1] - crop_size[1])
            x_start = np.random.randint(0, shape[2] - crop_size[2])
            return (
                slice(z_start, z_start + crop_size[0]),
                slice(y_start, y_start + crop_size[1]),
                slice(x_start, x_start + crop_size[2]),
            )

        crop_size = (20, 160, 160)
        zs, ys, xs = crop_slice(img.shape[-3:], crop_size)
        while (seg[:, zs, ys, xs] > 0).sum() < 0.2 * (seg > 0).sum():
            zs, ys, xs = crop_slice(img.shape[-3:], crop_size)

        return img[:, zs, ys, xs], seg[:, zs, ys, xs]

    def __getitem__(self, item):
        idx = item % len(self.images)
        img = imread(self.images[idx]).astype(np.float32)
        if img.max() > 1:
            img = np.clip(img / np.quantile(img, 0.999), 0, 1)
        seg = imread(self.labels[idx]).astype(np.uint16)
        img = img[np.newaxis]
        seg = label(seg)[np.newaxis]

        if self.augment:
            aug_index = np.random.randint(0, 8)
            img, seg = self.crop(img, seg)
            seg = relabel_sequential(seg)[0]
            img = self.augment_input(img)
            if aug_index == 0:
                return img, seg
            elif aug_index == 1:
                return np.flip(img, axis=-1).copy(), np.flip(seg, axis=-1).copy()
            elif aug_index == 2:
                return np.rot90(img, 1, axes=(2, 3)).copy(), np.rot90(
                    seg, 1, axes=(2, 3)
                ).copy()
            elif aug_index == 3:
                return np.rot90(img, 2, axes=(2, 3)).copy(), np.rot90(
                    seg, 2, axes=(2, 3)
                ).copy()
            elif aug_index == 4:
                return np.rot90(img, 3, axes=(2, 3)).copy(), np.rot90(
                    seg, 3, axes=(2, 3)
                ).copy()
            elif aug_index == 5:
                return np.flip(np.rot90(img, 1, axes=(2, 3)), axis=-1).copy(), np.flip(
                    np.rot90(seg, 1, axes=(2, 3)), axis=-1
                ).copy()
            elif aug_index == 6:
                return np.flip(np.rot90(img, 2, axes=(2, 3)), axis=-1).copy(), np.flip(
                    np.rot90(seg, 2, axes=(2, 3)), axis=-1
                ).copy()
            elif aug_index == 7:
                return np.flip(np.rot90(img, 3, axes=(2, 3)), axis=-1).copy(), np.flip(
                    np.rot90(seg, 3, axes=(2, 3)), axis=-1
                ).copy()
        else:
            return img, seg


class Nuclei3D(LightningDataModule):
    def __init__(self, dataset_path: Path):
        super().__init__()
        self.dataset_path = dataset_path

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            Nuclei3DDataset(
                raw_dir=self.dataset_path / "images_train",
                seg_dir=self.dataset_path / "masks_train",
                augment=True,
            ),
            batch_size=8,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            Nuclei3DDataset(
                raw_dir=self.dataset_path / "images_val",
                seg_dir=self.dataset_path / "masks_val",
                augment=False,
            ),
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
