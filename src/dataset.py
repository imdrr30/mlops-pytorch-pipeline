import os
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ])


def get_dataloaders(
    data_dir: str = None,
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    data_dir = data_dir or os.environ.get("DATA_ROOT", "./data")
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=get_transforms(True))
    val_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=get_transforms(False))
    options = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True}
    return DataLoader(train_dataset, shuffle=True, **options), DataLoader(val_dataset, shuffle=False, **options)
