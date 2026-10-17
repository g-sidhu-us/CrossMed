from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int, train: bool):
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class UltrasoundDataset(Dataset):
    def __init__(self, manifest: str | Path, image_size: int = 224, train: bool = False,
                 data_root: str | Path = "data"):
        self.frame = pd.read_csv(manifest)
        required = {"image_path", "label", "dataset", "sample_id"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"Manifest {manifest} is missing columns: {sorted(missing)}")
        self.data_root = Path(data_root).resolve()
        self.transform = build_transform(image_size, train)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = Path(row.image_path)
        if not image_path.is_absolute():
            image_path = self.data_root / image_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return {
            "image": tensor,
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "sample_id": str(row.sample_id),
            "dataset": str(row.dataset),
        }


def make_loader(manifest, image_size, batch_size, num_workers, train=False, data_root="data"):
    dataset = UltrasoundDataset(manifest, image_size=image_size, train=train, data_root=data_root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

