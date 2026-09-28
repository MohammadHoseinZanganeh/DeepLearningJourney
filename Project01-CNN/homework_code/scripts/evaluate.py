"""
evaluate.py
Model evaluation script.

Loads a saved model checkpoint and evaluates it on the full test set.
Useful for obtaining final accuracy numbers without re-training.
"""

import torch
import os
import sys

# Add the parent directory to the Python path so that local modules can be imported.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_loader import create_dataloaders
from models.model import BaseModel
from utils.metrics import accuracy, calculate_accuracy


def evaluate(model_path, config, device=None):
    """
    Evaluate a saved model checkpoint on the test set.

    Args:
        model_path: Path to the .pth checkpoint file.
        config: Dictionary with configuration keys:
            - data.data_dir: Path to the dataset folder.
            - data.batch_size: Batch size for the DataLoader.
            - data.num_workers: Number of data-loading subprocesses.
            - model.num_classes: Number of output classes.
            - model.dropout_rate: Dropout probability (must match training).
        device: torch.device to use. If None, auto-selects CUDA or CPU.

    Returns:
        float: Test accuracy percentage.
    """
    # Auto-select device if not specified.
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load only the test data loader (drops train and val).
    _, _, test_loader = create_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    # Rebuild the model architecture and load the saved weights.
    model = BaseModel(
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode (disables dropout).

    # Evaluate on the test set.
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # No need to track gradients during evaluation.
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct, total = accuracy(outputs, labels)
            total_correct += correct
            total_samples += total

    # Calculate and display final accuracy.
    test_acc = calculate_accuracy(total_correct, total_samples)
    print(f"Test Accuracy: {test_acc:.2f}%")

    return test_acc