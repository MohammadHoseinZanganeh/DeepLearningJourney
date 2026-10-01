"""
train.py
Training loop for the model.

This module handles the complete training pipeline:
- Single-epoch training with forward/backward passes.
- Validation on held-out data after each epoch.
- Model checkpointing based on best validation accuracy.
- Final evaluation on the test set.
- Visualization of sample predictions and first-layer filters.

Supports both SGD (with momentum) and Adam optimizers.
Swap the commented lines in train() to switch between them.
"""

import datetime
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
# Add the parent directory to the Python path so that local modules can be imported.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_loader import create_dataloaders
from models.model import BaseModel
from utils.metrics import accuracy, calculate_accuracy


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Perform a single training epoch over the provided data loader.

    Iterates over all batches, computes the loss, backpropagates gradients,
    and updates model weights. Tracks running loss and accuracy.

    Args:
        model: The PyTorch model to train.
        loader: DataLoader yielding (images, labels) batches for training.
        criterion: Loss function (e.g., CrossEntropyLoss).
        optimizer: Optimizer for weight updates (e.g., SGD, Adam).
        device: torch.device on which to run computations.

    Returns:
        tuple: (average_loss, accuracy_percentage) for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        # Standard training step: zero gradients, forward pass, loss, backward pass, update.

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1) # Get the class with highest score.
        total_correct += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / len(loader), calculate_accuracy(total_correct, total_samples)


def validate(model, loader, criterion, device):
    """
    Evaluate the model on a validation or test data loader.

    Runs in no_grad mode for efficiency. No weight updates are performed.
    Computes average loss and accuracy across all batches.

    Args:
        model: The PyTorch model to evaluate.
        loader: DataLoader yielding (images, labels) batches.
        criterion: Loss function.
        device: torch.device on which to run computations.

    Returns:
        tuple: (average_loss, accuracy_percentage).
    """

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


def train(config):
    """
    Run the full training pipeline.

    Loads the dataset, builds the model with the specified block type,
    trains for the configured number of epochs, saves the best checkpoint
    (based on validation accuracy), and performs a final evaluation on the
    held-out test set. Also generates sample prediction images and first-layer
    filter visualizations.

    Args:
        config: Dictionary with the following expected keys:
            - model.block_type: 'A', 'B', or 'C'.
            - model.num_classes: Number of output classes (10 for MNIST).
            - block_{type}.dropout_rate: Dropout probability.
            - block_{type}.learning_rate: Initial learning rate.
            - block_{type}.weight_decay: L2 regularization strength.
            - block_{type}.step_size: Scheduler step size (epochs).
            - block_{type}.gamma: Scheduler multiplicative decay factor.
            - data.data_dir: Path to the dataset folder.
            - data.batch_size: Batch size for DataLoaders.
            - data.num_workers: Number of data-loading subprocesses.
            - data.subset_size: If set, limit the dataset to this many samples.
            - training.num_epochs: Total number of epochs.
            - paths.save_dir: Directory in which to save model checkpoints.

    Returns:
        tuple: (trained_model, history_dict). The returned model holds the
               weights from the last epoch. history_dict contains lists of
               'train_loss', 'val_loss', 'train_acc', 'val_acc' per epoch.
    """
    # Automatically select GPU if CUDA is available, otherwise fall back to CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    block_type = config['model']['block_type']

    # Extract block type and its associated hyperparameters from the config.
    block_config = config[f'block_{block_type}']
    dropout_rate = block_config['dropout_rate']
    learning_rate = block_config['learning_rate']
    weight_decay = block_config['weight_decay']
    step_size = block_config['step_size']
    gamma = block_config['gamma']

    # Prepare data loaders for training, validation, and testing.
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        subset_size=config['data'].get('subset_size')
    )

    # Build the chosen architecture
    model = BaseModel(
        block_type=block_type,
        num_classes=config['model']['num_classes'],
        dropout_rate=dropout_rate
    ).to(device)

    # Loss: Cross Entropy
    criterion = nn.CrossEntropyLoss()
    
    # ---- Optimizer selection ----
    # Uncomment the desired optimizer and ensure the learning rate in config.yaml matches:
    #   - SGD:    lr = 0.1  (with momentum)
    #   - Adam:   lr = 0.001

    # SGD with momentum (recommended for final, well-generalized results).
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay
    )
    
    # #check the learning rate for Adam optimizer  in config (0.001)
    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=learning_rate,
    #     weight_decay=weight_decay
    # )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )

    # Create a unique checkpoint filename using a timestamp to avoid overwriting previous runs.
    os.makedirs(config['paths']['save_dir'], exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"best_model_{block_type}_{timestamp}.pth"
    model_path = os.path.join(config['paths']['save_dir'], model_filename)

    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    # ---- Main training loop ----
    for epoch in range(config['training']['num_epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step() # Apply learning rate decay

        # Record history for later plotting.
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1:2d}/{config['training']['num_epochs']}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint if validation accuracy improves.
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")
    
    # ---- Final evaluation on the held-out test set ----
    print("\n" + "=" * 50)
    print("Final Evaluation on Test Set")
    print("=" * 50)
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    # Show sample predictions
    from utils.visualization import show_classified_examples
    os.makedirs('results', exist_ok=True)
    show_classified_examples(
        model, test_loader, device, block_type,
        num_examples=3,
        save_path=f'results/sample_predictions_{block_type}_{timestamp}.png'
    )

    # Visualize first layer filters
    from utils.visualization import visualize_first_layer_filters
    visualize_first_layer_filters(
        model, test_loader, device, block_type,
        save_path=f'results/filters_layer1_{block_type}_{timestamp}.png'
    )

    return model, history