from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from skimage.measure import label
from tifffile import imread
from torch.utils.data import Dataset


class Organoid3DDataset(Dataset):
    def __init__(self, raw_dir: Path, seg_dir: Path, augment: bool = False):
        self.augment = augment
        raw_files = sorted(raw_dir.glob("*.tif"))
        self.images, self.labels = [], []
        for raw_file in raw_files:
            seg_file = seg_dir / raw_file.name
            self.images.append(raw_file)
            self.labels.append(seg_file)

    @staticmethod
    def _normalize_quantiles(data):
        mi, ma = np.quantile(data, [0.003, 0.999])
        return ((data - mi) / (ma - mi)).astype(np.float32)

    def __len__(self):
        if self.augment:
            return len(self.images) * 8
        else:
            return len(self.images)

    def __getitem__(self, item):
        idx = item % len(self.images)
        img = imread(self.images[idx]).astype(np.uint16)[:, ::2, ::2]
        seg = imread(self.labels[idx]).astype(np.uint16)[:, ::2, ::2]
        img = self._normalize_quantiles(img)
        img, seg = self.pad(img, seg)
        img = img[np.newaxis]
        seg = label(seg)[np.newaxis]

        if self.augment:
            aug_index = np.random.randint(0, 8)
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

    def pad(self, img, seg):
        shape = (256, 384, 384)

        padding = [(s - c) // 2 for s, c in zip(shape, img.shape)]
        padding = [(p, shape[i] - img.shape[i] - p) for i, p in enumerate(padding)]
        img = np.pad(img, padding, mode="constant")
        seg = np.pad(seg, padding, mode="constant")
        return img, seg


class Organoid3D(LightningDataModule):
    def __init__(self, dataset_path: Path):
        super().__init__()
        self.dataset_path = dataset_path

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            Organoid3DDataset(
                raw_dir=self.dataset_path / "raw_nhsCF568-train",
                seg_dir=self.dataset_path / "annot_corrected_epi-train",
                augment=True,
            ),
            batch_size=1,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            Organoid3DDataset(
                raw_dir=self.dataset_path / "raw_nhsCF568-val",
                seg_dir=self.dataset_path / "annot_corrected_epi-val",
                augment=False,
            ),
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
