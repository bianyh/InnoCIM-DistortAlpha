from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def build_cifar10_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_cifar10_loaders(
    data_root: str | Path,
    batch_size: int,
    workers: int,
    val_size: int = 5000,
    seed: int = 42,
    train_subset: int | None = None,
    val_subset: int | None = None,
    test_subset: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_root = Path(data_root)
    train_dataset = datasets.CIFAR10(
        root=str(data_root), train=True, download=False, transform=build_cifar10_transforms(train=True)
    )
    val_source = datasets.CIFAR10(
        root=str(data_root), train=True, download=False, transform=build_cifar10_transforms(train=False)
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_root), train=False, download=False, transform=build_cifar10_transforms(train=False)
    )

    train_len = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(range(len(train_dataset)), [train_len, val_size], generator=generator)
    train_dataset = torch.utils.data.Subset(train_dataset, list(train_indices))
    val_dataset = torch.utils.data.Subset(val_source, list(val_indices))

    if train_subset is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(min(train_subset, len(train_dataset)))))
    if val_subset is not None:
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(min(val_subset, len(val_dataset)))))
    if test_subset is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(min(test_subset, len(test_dataset)))))

    loader_args = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_args)
    return train_loader, val_loader, test_loader


def build_cifar_model(model_name: str = "cifar10_resnet20", pretrained: bool = False) -> torch.nn.Module:
    return torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        model_name,
        pretrained=pretrained,
        trust_repo=True,
    )
