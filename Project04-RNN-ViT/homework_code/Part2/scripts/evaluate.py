"""
Evaluation utilities for the Vision Transformer model.
"""

import torch
from tqdm import tqdm
from utils.metrics import accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on a given dataset.

    Args:
        model (nn.Module): The ViT model.
        dataloader (DataLoader): Data loader for evaluation.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run on.

    Returns:
        tuple: (average_loss, accuracy) for the dataset.
    """
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_batches = len(dataloader)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)

    avg_loss = running_loss / total_batches
    avg_acc = running_acc / total_batches
    return avg_loss, avg_acc