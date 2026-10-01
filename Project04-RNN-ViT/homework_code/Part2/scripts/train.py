"""
Training functions for Vision Transformer.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os

from utils.metrics import accuracy
from scripts.evaluate import evaluate  # import from evaluate.py


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The ViT model.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run on.

    Returns:
        tuple: (average_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_batches = len(dataloader)

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)

    avg_loss = running_loss / total_batches
    avg_acc = running_acc / total_batches
    return avg_loss, avg_acc


def train_model(model, train_loader, val_loader, config, device):
    """
    Full training loop.

    Args:
        model (nn.Module): The ViT model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        config (dict): Training configuration (includes epochs, lr, etc.).
        device (torch.device): Device to run on.

    Returns:
        dict: Training history with loss and accuracy lists.
    """
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0.0001)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    model_save_path = config['training'].get('model_save_path', 'best_model.pth')

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return history