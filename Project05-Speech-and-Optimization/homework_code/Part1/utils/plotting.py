"""Plotting helper functions for training curves and confusion matrices."""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path, feature_type="mel"):
    """Plot training/validation loss and accuracy curves side by side.

    Args:
        train_losses (list[float]): Training loss for each epoch.
        val_losses (list[float]): Validation loss for each epoch.
        train_accuracies (list[float]): Training accuracy for each epoch.
        val_accuracies (list[float]): Validation accuracy for each epoch.
        save_path (str): File path where the plot image is saved.
        feature_type (str): "mel" or "hubert" for title.
    """
    epoch_numbers = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    feature_name = "Mel-Spectrogram + CNN" if feature_type == "mel" else "HuBERT + MLP"

    # Loss curve
    axes[0].plot(epoch_numbers, train_losses, 'b-', label="Train Loss", linewidth=2)
    axes[0].plot(epoch_numbers, val_losses, 'r-', label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title(f"{feature_name} - Loss Curve", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(epoch_numbers, train_accuracies, 'b-', label="Train Accuracy", linewidth=2)
    axes[1].plot(epoch_numbers, val_accuracies, 'r-', label="Validation Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title(f"{feature_name} - Accuracy Curve", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Find best validation accuracy (LAST occurrence of max value)
    best_val_acc = max(val_accuracies)
    # Find the LAST occurrence of the best value
    reversed_indices = val_accuracies[::-1]
    last_index = len(val_accuracies) - 1 - reversed_indices.index(best_val_acc)
    best_epoch = last_index + 1
    
    # Add annotation
    axes[1].annotate(f'Best: {best_val_acc:.4f} at Epoch {best_epoch}',
                    xy=(best_epoch, best_val_acc),
                    xytext=(best_epoch + 1.5, best_val_acc - 0.08),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                    fontsize=10,
                    color='green',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="green", alpha=0.8))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Training curves saved to '{save_path}'")


def plot_confusion_matrix(confusion_matrix_values, class_names, save_path, feature_type="mel"):
    """Plot and save a confusion matrix heatmap.

    Args:
        confusion_matrix_values (numpy.ndarray): Confusion matrix values.
        class_names (list[str]): Class names in label order.
        save_path (str): File path where the plot image is saved.
        feature_type (str): "mel" or "hubert" for title.
    """
    plt.figure(figsize=(8, 6))
    
    feature_name = "Mel-Spectrogram + CNN" if feature_type == "mel" else "HuBERT + MLP"
    
    sns.heatmap(
        confusion_matrix_values,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(f"{feature_name} - Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to '{save_path}'")