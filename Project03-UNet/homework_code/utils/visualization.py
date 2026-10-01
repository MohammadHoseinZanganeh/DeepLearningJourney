"""
visualization module for plotting training results and comparing predictions.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_training_history(
    history: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
) -> None:
    """
    plot training and validation loss and mIoU curves.

    Args:
        history: list of dicts with epoch, train_loss, val_loss, train_miou, val_miou
        save_path: path to save the figure (if None, just shows)
    """
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_miou = [h["train_miou"] for h in history]
    val_miou = [h["val_miou"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # loss plot
    axes[0].plot(epochs, train_loss, "b-", label="Train Loss")
    axes[0].plot(epochs, val_loss, "r-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # mIoU plot
    axes[1].plot(epochs, train_miou, "b-", label="Train mIoU")
    axes[1].plot(epochs, val_miou, "r-", label="Val mIoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].set_title("Training and Validation mIoU")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to: {save_path}")
    plt.show()


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    denormalize image tensor to numpy array for display.

    Args:
        tensor: normalized image tensor (C, H, W)

    Returns:
        numpy array (H, W, C) with values in [0, 1]
    """
    img = tensor.permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def visualize_predictions(
    images: List[torch.Tensor],
    masks_true: List[torch.Tensor],
    masks_pred: List[torch.Tensor],
    class_names: List[str],
    save_path: Optional[Path] = None,
    num_samples: Optional[int] = None,
) -> None:
    """
    show input image, ground truth, and prediction side by side.

    Args:
        images: list of input images (tensors)
        masks_true: list of ground truth masks
        masks_pred: list of predicted masks
        class_names: list of class names
        save_path: path to save figure (optional)
        num_samples: number of samples to show (default: all)
    """
    if num_samples is not None:
        images = images[:num_samples]
        masks_true = masks_true[:num_samples]
        masks_pred = masks_pred[:num_samples]
    
    num_samples = len(images)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # input image
        img = denormalize_image(images[i])
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Image", fontsize=10)
        axes[i, 0].axis("off")
        
        # ground truth
        axes[i, 1].imshow(masks_true[i], cmap="tab10", vmin=0, vmax=len(class_names)-1)
        axes[i, 1].set_title("Ground Truth", fontsize=10)
        axes[i, 1].axis("off")
        
        # prediction
        axes[i, 2].imshow(masks_pred[i], cmap="tab10", vmin=0, vmax=len(class_names)-1)
        axes[i, 2].set_title("Prediction", fontsize=10)
        axes[i, 2].axis("off")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    plt.show()


def get_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 5,
) -> tuple:
    """
    get predictions for visualization.

    Args:
        model: trained model
        loader: dataloader
        device: torch device
        num_samples: number of samples to return

    Returns:
        tuple of (images, ground_truth_masks, predicted_masks)
    """
    model.eval()
    images_list = []
    masks_true_list = []
    masks_pred_list = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            
            images_list.extend(imgs.cpu())
            masks_true_list.extend(masks.cpu())
            masks_pred_list.extend(preds.cpu())
            
            if len(images_list) >= num_samples:
                break
    
    return images_list[:num_samples], masks_true_list[:num_samples], masks_pred_list[:num_samples]