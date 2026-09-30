"""
train_transfer.py
Transfer learning script for Fashion MNIST.

Loads a pretrained MNIST model, freezes the feature extractor,
replaces the classifier, and trains on Fashion MNIST.
"""

import datetime
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_loader import create_fashion_dataloaders
from models.model import BaseModel, TransferModel
from utils.metrics import accuracy, calculate_accuracy
from utils.visualization import plot_training_curves, plot_confusion_matrix

FASHION_CLASSES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the classifier head for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / len(loader), calculate_accuracy(total_correct, total_samples)


def validate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / len(loader), calculate_accuracy(total_correct, total_samples)


def train_transfer(config):
    """Run transfer learning pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Fashion MNIST data.
    train_loader, val_loader, test_loader = create_fashion_dataloaders(
        data_dir=config['fashion_data']['data_dir'],
        batch_size=config['fashion_data']['batch_size'],
        num_workers=config['fashion_data']['num_workers'],
        subset_size=config['fashion_data'].get('subset_size')
    )

    # Load pretrained MNIST model.
    print(f"\n[INFO] Loading pretrained model: {config['transfer']['pretrained_path']}")
    pretrained = BaseModel(
        block_type=config['transfer']['pretrained_block'],
        num_classes=10,
        dropout_rate=0.0
    )
    pretrained.load_state_dict(
        torch.load(config['transfer']['pretrained_path'], map_location=device)
    )

    # Build transfer model with frozen backbone.
    model = TransferModel(
        pretrained_model=pretrained,
        num_classes=config['model']['num_classes'],
        dropout_rate=config['transfer']['dropout_rate']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,} ({(100*trainable_params/total_params):.1f}%)")

    # Training setup.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['transfer']['learning_rate'],
        weight_decay=config['transfer']['weight_decay']
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['transfer']['step_size'],
        gamma=config['transfer']['gamma']
    )

    os.makedirs(config['paths']['save_dir'], exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(config['paths']['save_dir'], f'transfer_fashion_{timestamp}.pth')

    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(config['transfer']['num_epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1:2d}/{config['transfer']['num_epochs']}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")

    # Final test evaluation.
    print("\n" + "=" * 50)
    print("Final Evaluation on Fashion MNIST Test Set")
    print("=" * 50)
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    # Save results.
    os.makedirs('results', exist_ok=True)
    plot_training_curves(
        history['train_loss'], history['val_loss'],
        history['train_acc'], history['val_acc'],
        'Fashion',
        save_path='results/transfer_fashion_curves.png'
    )
    plot_confusion_matrix(
        model, test_loader, device, FASHION_CLASSES,
        save_path='results/transfer_fashion_confusion.png'
    )

    return model, history


if __name__ == "__main__":
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    train_transfer(config)