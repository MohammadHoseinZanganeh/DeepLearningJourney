"""
data_loader.py
Data loading for MNIST dataset from CSV files with augmentation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import os


class MNISTCSVDataset(Dataset):
    """
    Custom Dataset class for MNIST loaded from CSV files.
    Inherits from torch.utils.data.Dataset.
    """
    
    def __init__(self, csv_path, transform=None, subset_size=None):
        """
        Args:
            csv_path: Path to the CSV file (mnist_train.csv or mnist_test.csv)
            transform: Image transformations to apply
            subset_size: If set, uses only first N samples for quick testing
        """
        self.transform = transform
    
        # Load CSV file - try with and without header
        data = pd.read_csv(csv_path)
        
        # If first column name is a number (like '5'), there's no proper header
        # Re-read without header
        if data.columns[0] == 'label' or str(data.columns[0]).isdigit():
            if str(data.columns[0]).isdigit():
                data = pd.read_csv(csv_path, header=None)
        
        # First column is label, rest are pixel values
        self.labels = data.iloc[:, 0].values.astype(np.int64)
        self.images = data.iloc[:, 1:].values.astype(np.float32)
        
        # Reshape images to 28x28
        self.images = self.images.reshape(-1, 28, 28)
        
        # Normalize pixel values from [0, 255] to [0, 1]
        self.images = self.images / 255.0
        
        # Apply subset for quick testing
        if subset_size is not None:
            self.images = self.images[:subset_size]
            self.labels = self.labels[:subset_size]
            print(f"[INFO] Using subset: {subset_size} samples")
        
        print(f"[INFO] Loaded {len(self.labels)} samples, image shape: {self.images.shape[1:]}")
    
    def __len__(self):
        """Returns total number of samples in the dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Retrieves a single sample at the given index.
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (image_tensor, label)
        """
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL Image for transforms
        image_pil = transforms.ToPILImage()(image)
        
        # Apply transformations
        if self.transform:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        
        return image_tensor, label


def get_train_transforms():
    """
    Training transforms with data augmentation.
    
    """
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  #displacement of image
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_test_transforms():
    """
    Test transforms. No augmentation applied, only normalization.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def create_dataloaders(data_dir='./data/dataset', batch_size=64, num_workers=2, 
                       subset_size=None, val_split=0.1):
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        data_dir: Path to folder containing mnist_train.csv and mnist_test.csv
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading
        subset_size: If set, limit to first N samples for quick testing
        val_split: Fraction of training data to use for validation (default: 0.1)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    train_csv = os.path.join(data_dir, 'mnist_train.csv')
    test_csv = os.path.join(data_dir, 'mnist_test.csv')
    
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training file not found: {train_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test file not found: {test_csv}")
    
    # Load full training data first (without transforms to get the split right)
    full_train_data = pd.read_csv(train_csv)
    
    if subset_size is not None:
        full_train_data = full_train_data.iloc[:subset_size]
    
    # Shuffle before splitting
    full_train_data = full_train_data.sample(frac=1, random_state=42).reset_index(drop=True) # drop makes new idx orderd after shuffling
    
    # Split into train and validation
    val_size = int(len(full_train_data) * val_split)
    train_data = full_train_data.iloc[val_size:]
    val_data = full_train_data.iloc[:val_size]
    
    # Save temporary CSVs for train and val
    train_split_csv = os.path.join(data_dir, 'mnist_train_split.csv')
    val_split_csv = os.path.join(data_dir, 'mnist_val_split.csv')
    
    train_data.to_csv(train_split_csv, index=False)
    val_data.to_csv(val_split_csv, index=False)
    
    # Create datasets with their respective transforms
    train_dataset = MNISTCSVDataset(train_split_csv, get_train_transforms())
    val_dataset = MNISTCSVDataset(val_split_csv, get_test_transforms())  # No augmentation for val
    test_dataset = MNISTCSVDataset(test_csv, get_test_transforms())
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)
    
    print(f"[INFO] Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"[INFO] Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"[INFO] Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def create_fashion_dataloaders(data_dir='./data/fashion', batch_size=64, num_workers=2,
                                subset_size=None, val_split=0.1):
    """
    Create train, validation, and test DataLoaders for Fashion MNIST.

    Reads the raw .gz files directly from the given folder without requiring
    PyTorch's expected subfolder structure (FashionMNIST/raw/).
    Handles the IDX file format used by MNIST-style datasets.

    Args:
        data_dir: Path to folder containing the four .gz files.
        batch_size: Number of samples per batch.
        num_workers: Number of data-loading subprocesses.
        subset_size: If set, limit to first N samples for quick testing.
        val_split: Fraction of training data used for validation.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    import gzip
    import struct
    import numpy as np

    def read_images(path):
        """Read images from an IDX-format .gz file. Returns numpy array (N, 28, 28)."""
        with gzip.open(path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        return images

    def read_labels(path):
        """Read labels from an IDX-format .gz file. Returns numpy array (N,)."""
        with gzip.open(path, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    # Build paths to the four .gz files.
    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

    # Load raw data into numpy arrays.
    train_images = read_images(train_images_path)
    train_labels = read_labels(train_labels_path)
    test_images = read_images(test_images_path)
    test_labels = read_labels(test_labels_path)

    # Normalize pixel values from [0, 255] to [0, 1].
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Optionally use only a subset for quick experimentation.
    if subset_size is not None:
        train_images = train_images[:subset_size]
        train_labels = train_labels[:subset_size]
        test_images = test_images[:subset_size]
        test_labels = test_labels[:subset_size]

    # Shuffle training data before splitting (fixed seed for reproducibility).
    indices = np.random.RandomState(42).permutation(len(train_labels))
    train_images = train_images[indices]
    train_labels = train_labels[indices]

    # Split training data into train and validation sets.
    train_size = int(len(train_labels) * (1 - val_split))
    val_size = len(train_labels) - train_size

    # Wrap numpy arrays in PyTorch TensorDatasets.
    # unsqueeze(1) adds the channel dimension: (N, 28, 28) -> (N, 1, 28, 28).
    train_data = torch.utils.data.TensorDataset(
        torch.from_numpy(train_images[:train_size]).unsqueeze(1),
        torch.from_numpy(train_labels[:train_size]).long()
    )
    val_data = torch.utils.data.TensorDataset(
        torch.from_numpy(train_images[train_size:]).unsqueeze(1),
        torch.from_numpy(train_labels[train_size:]).long()
    )
    test_data = torch.utils.data.TensorDataset(
        torch.from_numpy(test_images).unsqueeze(1),
        torch.from_numpy(test_labels).long()
    )

    # Build DataLoaders.
    # Note: No augmentation transforms are applied here.
    # For transfer learning, frozen features work with raw pixel values.
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)

    print(f"[INFO] Fashion MNIST - Train: {train_size}, Val: {val_size}, Test: {len(test_data)}")

    return train_loader, val_loader, test_loader    


if __name__ == "__main__":
    print("=" * 50)
    print("Testing DataLoader with CSV files...")
    print("=" * 50)
    
    # data_dir is relative to the data/ folder
    data_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        subset_size=100,
        num_workers=0
    )
    
    # Show a sample batch
    images, labels = next(iter(train_loader))
    print(f"\nSample batch - images: {images.shape}, labels: {labels.shape}")
    print(f"Labels: {labels[:10].tolist()}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
    print("\nDataLoader test completed successfully.")