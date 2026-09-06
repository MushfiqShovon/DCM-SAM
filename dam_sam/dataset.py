import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import transforms as T


class JointAMDataset(Dataset):
    """Additive-manufacturing XCT slices with two independent binary defect masks per
    image: pore (class_2) and inclusion (class_3). The `image` column of the pore and
    inclusion CSVs must line up row-for-row (same image, different label column), which is
    how the dataset's own `*_class2.csv` / `*_class3.csv` pairs are generated.

    Every `__getitem__` call returns one image plus both of its masks, so a single joint
    training script can run the pore and inclusion heads on the same input batch without
    reading the image twice.
    """

    def __init__(
        self,
        dataset_dir: str,
        pore_csv: str,
        incl_csv: str,
        image_size: int = T.SAM_IMAGE_SIZE,
        augment: bool = False,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        self.augment = augment

        pore_df = pd.read_csv(self.dataset_dir / f"{pore_csv}.csv")
        incl_df = pd.read_csv(self.dataset_dir / f"{incl_csv}.csv")
        assert len(pore_df) == len(incl_df), (
            f"pore/incl row count mismatch: {pore_csv} has {len(pore_df)}, {incl_csv} has {len(incl_df)}"
        )
        assert (pore_df["image"] == incl_df["image"]).all(), (
            f"{pore_csv} and {incl_csv} must reference the same images in the same order"
        )

        self.images = [str(self.dataset_dir / p) for p in pore_df["image"]]
        self.pore_labels = [str(self.dataset_dir / p) for p in pore_df["label"]]
        self.incl_labels = [str(self.dataset_dir / p) for p in incl_df["label"]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = T.load_image(self.images[idx], self.image_size)
        pore_mask = T.load_mask(self.pore_labels[idx], self.image_size)
        incl_mask = T.load_mask(self.incl_labels[idx], self.image_size)

        if self.augment:
            image, pore_mask, incl_mask = T.joint_augment(image, pore_mask, incl_mask)

        return {
            "image": image,
            "pore_mask": pore_mask,
            "incl_mask": incl_mask,
            "name": os.path.basename(self.images[idx]),
        }


class SingleClassAMDataset(Dataset):
    """Additive-manufacturing XCT slices with a single binary defect mask, for evaluating
    (or training) one head in isolation without loading the other class's labels."""

    def __init__(self, dataset_dir: str, csv_name: str, image_size: int = T.SAM_IMAGE_SIZE):
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        df = pd.read_csv(self.dataset_dir / f"{csv_name}.csv")
        self.images = [str(self.dataset_dir / p) for p in df["image"]]
        self.labels = [str(self.dataset_dir / p) for p in df["label"]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image": T.load_image(self.images[idx], self.image_size),
            "mask": T.load_mask(self.labels[idx], self.image_size),
            "path": self.images[idx],
            "name": os.path.basename(self.images[idx]),
        }


def build_oversample_weights(dataset_dir: str, csv_name: str, oversample_factor: float = 5.0) -> torch.Tensor:
    """Per-image sample weight for WeightedRandomSampler: `oversample_factor` for images
    whose mask has at least one positive pixel, 1.0 for empty ones. Used for the inclusion
    class, which is far sparser than pore in the training set, so a plain shuffled loader
    would show the model very few positive inclusion examples per epoch."""
    dataset_dir = Path(dataset_dir)
    df = pd.read_csv(dataset_dir / f"{csv_name}.csv")
    weights = []
    for label_path in df["label"]:
        arr = np.array(Image.open(dataset_dir / label_path).convert("L"))
        weights.append(float(oversample_factor) if (arr > 0).any() else 1.0)
    return torch.tensor(weights, dtype=torch.float)
