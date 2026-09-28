"""
visualization.py
Plotting functions for training visualization.

Provides utilities to:
- Plot training/validation loss and accuracy curves.
- Display sample predictions with true vs predicted labels.
- Visualize first convolutional layer filter responses on a test image.
"""

import os
import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, block_type, save_path=None):
    """
    Plot training and validation loss and accuracy curves side by side.

    Args:
        train_losses: List of average training losses per epoch.
        val_losses: List of average validation losses per epoch.
        train_accs: List of training accuracies (percentages) per epoch.
        val_accs: List of validation accuracies (percentages) per epoch.
        block_type: Architecture identifier ('A', 'B', or 'C') used in the plot title.
        save_path: If provided, saves the figure to this path instead of displaying it.
    """
    epochs = range(1, len(train_losses) + 1)

    # Human-readable architecture names for plot titles.
    architecture_names = {
        'A': 'Residual Network',
        'B': 'Inception Network',
        'C': 'ResNeXt Network'
    }
    arch_name = architecture_names.get(block_type, f'Block {block_type}')

    # Create side-by-side subplots for loss and accuracy.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f'Architecture: Block {block_type} - {arch_name}',
        fontsize=14,
        fontweight='bold'
    )

    # ---- Loss subplot ----
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True)

    # ---- Accuracy subplot ----
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] Training curves saved to: {save_path}")
    else:
        plt.show()


def show_sample_predictions(model, test_loader, device, num_samples=8):
    """
    Display a grid of sample predictions from the test set.

    Takes the first batch from the test loader and shows up to num_samples
    images with their true and predicted labels.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for the test set.
        device: torch.device to run inference on.
        num_samples: Number of samples to display (default: 8, arranged as 2x4).
    """
    model.eval()
    # Grab the first batch from the test loader.
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples].to(device), labels[:num_samples]

    with torch.no_grad():
        outputs = model(images)
        _, preds = outputs.max(1)

    # Bring tensors back to CPU for plotting.
    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    axes = axes.flatten()

    for i in range(num_samples):
        # Denormalize the image (undo the Normalize transform with MNIST stats).
        img = images[i].squeeze()
        img = img * 0.3081 + 0.1307
        img = torch.clamp(img, 0, 1)

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {labels[i]}, Pred: {preds[i]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def show_classified_examples(model, test_loader, device, block_type, num_examples=3, save_path=None):
    """
    Display randomly selected test images with true and predicted labels.

    Unlike show_sample_predictions, this function randomly samples images
    from the entire test set (not just the first batch), so each call
    produces different examples.

    Correct predictions are shown with green titles; incorrect ones with red.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for the test set.
        device: torch.device to run inference on.
        block_type: Architecture identifier ('A', 'B', or 'C') for the plot title.
        num_examples: Number of random samples to show (default: 3).
        save_path: If provided, saves the figure to this path.
    """
    model.eval()

    # Collect all test images and labels from every batch.
    all_images = []
    all_labels = []
    for imgs, lbls in test_loader:
        all_images.append(imgs)
        all_labels.append(lbls)

    # Concatenate into single tensors of shape (N, C, H, W) and (N,).
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Randomly select num_examples indices without replacement.
    indices = torch.randperm(len(all_images))[:num_examples]
    images = all_images[indices].to(device)
    labels = all_labels[indices].to(device)

    # Run inference on the selected images.
    with torch.no_grad():
        outputs = model(images)
        _, preds = outputs.max(1)

    # Move to CPU for plotting.
    images, labels, preds = images.cpu(), labels.cpu(), preds.cpu()

    # Create a horizontal strip of subplots.
    fig, axes = plt.subplots(1, num_examples, figsize=(num_examples * 3, 3))
    if num_examples == 1:
        axes = [axes]  # Make iterable when there is only one subplot.

    for i in range(num_examples):
        # Denormalize the image.
        img = images[i].squeeze()
        img = img * 0.3081 + 0.1307
        img = torch.clamp(img, 0, 1)

        # Green title if correct, red if wrong.
        color = 'green' if preds[i] == labels[i] else 'red'
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(
            f'True: {labels[i].item()}\nPred: {preds[i].item()}',
            color=color
        )
        axes[i].axis('off')

    # Add a super title with the architecture name.
    architecture_names = {
        'A': 'Residual Network',
        'B': 'Inception Network',
        'C': 'ResNeXt Network'
    }
    arch_name = architecture_names.get(block_type, f'Block {block_type}')
    fig.suptitle(
        f'Architecture: Block {block_type} - {arch_name}',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Sample predictions saved to: {save_path}")
    plt.close()


def visualize_first_layer_filters(model, test_loader, device, block_type, save_path=None):
    """
    Visualize the outputs of all 32 filters in the first convolutional layer.

    A single test image is passed through the first conv layer (conv1),
    and the resulting 32 feature maps (each 28x28) are plotted in a 4x8 grid.
    Brighter regions indicate where a filter detected its preferred pattern.

    Args:
        model: Trained PyTorch model (must have a 'conv1' attribute).
        test_loader: DataLoader for the test set (used to grab one image).
        device: torch.device to run inference on.
        block_type: Architecture identifier ('A', 'B', or 'C') for the plot title.
        save_path: If provided, saves the figure to this path.
    """
    model.eval()

    # Grab a single test image (first image of the first batch).
    images, _ = next(iter(test_loader))
    image = images[0:1].to(device)  # Keep the batch dimension: shape (1, 1, 28, 28).

    # Register a forward hook to capture the output of the first conv layer.
    activation = {}

    def hook_fn(module, input, output):
        activation['conv1'] = output

    hook = model.conv1.register_forward_hook(hook_fn)

    # Forward pass (activations are stored in the hook).
    with torch.no_grad():
        model(image)

    # Remove the hook after capturing.
    hook.remove()

    # Extract feature maps: shape becomes (32, 28, 28).
    feature_maps = activation['conv1'][0].cpu()

    # Plot all 32 filters in a 4x8 grid.
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    arch_names = {'A': 'Residual', 'B': 'Inception', 'C': 'ResNeXt'}
    fig.suptitle(
        f'Block {block_type} - {arch_names.get(block_type, "")}: First Layer Filters (32 filters)',
        fontsize=14,
        fontweight='bold'
    )

    for i, ax in enumerate(axes.flat):
        if i < 32:
            ax.imshow(feature_maps[i], cmap='viridis')
            ax.set_title(f'Filter {i + 1}', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[INFO] Filter visualization saved to: {save_path}")


def plot_confusion_matrix(model, test_loader, device, class_names, save_path=None):
    """
    Plot confusion matrix for classification evaluation.

    Diagonal entries are correct predictions. Off-diagonal entries show
    which classes are most often confused with each other.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for test data.
        device: torch.device for inference.
        class_names: List of human-readable class names.
        save_path: If provided, saves the figure to this path.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Fashion MNIST (Transfer Learning)')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Confusion matrix saved to: {save_path}")
    else:
        plt.show()