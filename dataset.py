"""
dataset.py — Dual-stream JunctionDataset (unchanged) +
             build_dataloaders now accepts use_sampler=False

The only change from the previous version is the use_sampler parameter
in build_dataloaders(). Set use_sampler=False (the new default) to fix
the T-junction over-prediction caused by sampler + focal loss
double-compensating for class imbalance.
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


LABEL_MAP   = {"No junction": 0, "T-junction": 1, "X-junction": 2}
CLASS_NAMES = ["No junction", "T-junction", "X-junction"]

_RGB_MEAN  = [0.485, 0.456, 0.406]
_RGB_STD   = [0.229, 0.224, 0.225]
_MASK_MEAN = [0.5, 0.5, 0.5]
_MASK_STD  = [0.5, 0.5, 0.5]


class JointTransform:
    def __init__(self, img_size: int = 384, split: str = "train"):
        self.img_size = img_size
        self.split    = split
        self.rgb_color = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.05),
        ])
        self.rgb_to_tensor  = T.Compose([T.ToTensor(), T.Normalize(_RGB_MEAN,  _RGB_STD)])
        self.mask_to_tensor = T.Compose([T.ToTensor(), T.Normalize(_MASK_MEAN, _MASK_STD)])
        self.rgb_erasing    = T.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.5, 2.0))

    def __call__(self, rgb: Image.Image, mask: Image.Image):
        rgb  = TF.resize(rgb,  [self.img_size, self.img_size])
        mask = TF.resize(mask, [self.img_size, self.img_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.split == "train":
            if random.random() > 0.5:
                rgb  = TF.hflip(rgb)
                mask = TF.hflip(mask)

            angle, translate, scale, shear = T.RandomAffine.get_params(
                degrees=(-5, 5),
                translate=(0.04, 0.04),
                scale_ranges=(0.95, 1.05),
                shears=None,
                img_size=[self.img_size, self.img_size],
            )
            rgb  = TF.affine(rgb,  angle, translate, scale, shear)
            mask = TF.affine(mask, angle, translate, scale, shear,
                             interpolation=TF.InterpolationMode.NEAREST)

            rgb = self.rgb_color(rgb)

        rgb_t  = self.rgb_to_tensor(rgb)
        mask_t = self.mask_to_tensor(mask)

        if self.split == "train":
            rgb_t = self.rgb_erasing(rgb_t)

        return rgb_t, mask_t


class JunctionDataset(Dataset):
    def __init__(
        self,
        excel_path : str,
        rgb_dir    : str,
        mask_dir   : str,
        split      : str = "train",
        img_size   : int = 384,
    ):
        self.df        = pd.read_excel(excel_path)
        self.rgb_dir   = rgb_dir
        self.mask_dir  = mask_dir
        self.transform = JointTransform(img_size=img_size, split=split)
        self._labels   = [
            LABEL_MAP[str(self.df.iloc[i, 1]).strip()]
            for i in range(len(self.df))
        ]

    def __len__(self):
        return len(self.df)

    def _load_png(self, directory, raw_filename):
        base = os.path.splitext(str(raw_filename))[0]
        path = os.path.join(directory, base + ".png")
        try:
            return Image.open(path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"\nFile not found: {path}")

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        filename = str(row.iloc[0])
        label    = LABEL_MAP[str(row.iloc[1]).strip()]
        rgb      = self._load_png(self.rgb_dir,  filename)
        mask     = self._load_png(self.mask_dir, filename)
        rgb_t, mask_t = self.transform(rgb, mask)
        return rgb_t, mask_t, label

    def class_counts(self):
        from collections import Counter
        c = Counter(self._labels)
        return [c.get(i, 0) for i in range(len(LABEL_MAP))]

    def sample_weights(self):
        counts  = self.class_counts()
        inv     = [1.0 / max(c, 1) for c in counts]
        weights = [inv[lbl] for lbl in self._labels]
        return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    train_excel    : str,
    train_rgb_dir  : str,
    train_mask_dir : str,
    val_excel      : str,
    val_rgb_dir    : str,
    val_mask_dir   : str,
    batch_size     : int  = 4,
    img_size       : int  = 384,
    num_workers    : int  = 4,
    use_sampler    : bool = False,   # ← NEW: False = rely on focal loss alpha only
):
    """
    use_sampler=True   → WeightedRandomSampler balances batches + Focal Loss
                         (the previous behaviour — caused T-junction over-prediction)
    use_sampler=False  → Standard shuffle + Focal Loss alpha only (recommended)
    """
    train_ds = JunctionDataset(train_excel, train_rgb_dir, train_mask_dir,
                               split="train", img_size=img_size)
    val_ds   = JunctionDataset(val_excel,   val_rgb_dir,   val_mask_dir,
                               split="val",   img_size=img_size)

    if use_sampler:
        sampler      = WeightedRandomSampler(train_ds.sample_weights(),
                                             len(train_ds), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            persistent_workers=(num_workers > 0),
        )
        print("[Dataset] Using WeightedRandomSampler (+ focal loss)")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            persistent_workers=(num_workers > 0),
        )
        print("[Dataset] Shuffle only — focal loss alpha handles imbalance")

    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    print(f"[Dataset] Train : {len(train_ds):,} | Val : {len(val_ds):,}")
    print(f"[Dataset] Train class counts (No/T/X): {train_ds.class_counts()}")
    print(f"[Dataset]   Val class counts (No/T/X): {val_ds.class_counts()}")
    return train_loader, val_loader
