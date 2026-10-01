"""
Data loader for CIFAR-10 dataset.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_data_loaders(batch_size=128, num_workers=2, subset_fraction=1.0, resize=None):
    """
    Load CIFAR-10 dataset and create data loaders.

    Args:
        batch_size (int): Number of samples per batch. Default is 128.
        num_workers (int): Number of subprocesses for data loading. Default is 2.
        subset_fraction (float): Fraction of dataset to use (0.0 to 1.0).
                                  Useful for quick testing. Default is 1.0.
        resize (int, optional): If provided, images will be resized to (resize, resize).
                                  Default is None (no resizing, standard 32x32).
    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader).
    """
    # Set data root to Part2/data/dataset/ relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, '..', 'data', 'dataset')
    os.makedirs(data_root, exist_ok=True)

    # Define transforms based on resize parameter
    if resize is not None:
        transform_train = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(resize, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                               std=[0.2023, 0.1994, 0.2010])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                               std=[0.2023, 0.1994, 0.2010])
        ])
    else:
        # Standard CIFAR-10 transforms (32x32)
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                               std=[0.2023, 0.1994, 0.2010])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                               std=[0.2023, 0.1994, 0.2010])
        ])

    # Load training dataset with augmentation
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )

    # Load validation dataset (same as train but without augmentation)
    val_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_test
    )

    # Load test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )

    # If subset_fraction is less than 1.0, use only a portion of data
    if subset_fraction < 1.0:
        total_train = len(train_dataset)
        total_val = len(val_dataset)
        total_test = len(test_dataset)

        train_size = int(total_train * subset_fraction)
        val_size = int(total_val * 0.1 * subset_fraction)   # 10% of train for validation
        test_size = int(total_test * subset_fraction)

        train_indices = np.random.permutation(total_train)[:train_size]
        val_indices = np.random.permutation(total_val)[:val_size]
        test_indices = np.random.permutation(total_test)[:test_size]

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        test_dataset = Subset(test_dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("CIFAR-10 dataset loaded successfully.")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")
    print(f"  - Data stored at: {data_root}")
    if resize:
        print(f"  - Images resized to: {resize}x{resize}")

    return train_loader, val_loader, test_loader