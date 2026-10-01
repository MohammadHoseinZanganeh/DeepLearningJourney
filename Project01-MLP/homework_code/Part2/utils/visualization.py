import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


def plot_training_history(history: Dict[str, List[float]], save_path: str = None) -> None:
    """
    Draw two charts: one for loss and one for accuracy during training
    
    Args:
        history: A dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Where to save the image file (optional)
    """
    # Create a figure with two side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # X-axis: epoch numbers (1, 2, 3, ...)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Left plot: Loss over time
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss During Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Accuracy over time
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy During Training', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to file if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    
    Args:
        y_true: True labels (e.g., [1, 0, 2, 1, ...])
        y_pred: Predicted labels (e.g., [1, 0, 1, 1, ...])
        num_classes: Number of classes (10 for MNIST)
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    # Initialize empty matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Fill the matrix: cm[true_label][predicted_label] += 1
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label][pred_label] += 1
    
    return cm


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         num_classes: int = 10,
                         save_path: str = None) -> np.ndarray:
    """
    Draw a confusion matrix as a heatmap
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes (default 10 for MNIST)
        save_path: Where to save the image (optional)
        
    Returns:
        The confusion matrix as a numpy array
    """
    # Calculate confusion matrix
    cm = calculate_confusion_matrix(y_true, y_pred, num_classes)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Draw heatmap manually
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Count')
    
    # Add numbers inside each cell
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm[i, j]), 
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    # Labels and title
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Set tick labels (0, 1, 2, ..., 9)
    plt.xticks(range(num_classes), range(num_classes))
    plt.yticks(range(num_classes), range(num_classes))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    
    return cm


def plot_misclassified_samples(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                               num_samples: int = 10, save_path: str = None,
                               image_shape: tuple = (28, 28)) -> None:
    """
    Display misclassified samples with true and predicted labels
    
    Args:
        X: Input images (flattened)
        y_true: True labels
        y_pred: Predicted labels
        num_samples: Number of misclassified samples to show
        save_path: Path to save the figure
        image_shape: Shape to reshape images for display
    """
    # Find misclassified indices
    misclassified_mask = y_true != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    if len(misclassified_indices) == 0:
        print("no misclassified samples found! Perfect classification!")
        return
    
    # Randomly select samples
    num_samples = min(num_samples, len(misclassified_indices))
    selected_indices = np.random.choice(misclassified_indices, num_samples, replace=False)
    
    # Create figure
    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i, idx in enumerate(selected_indices):
        ax = axes[i]
        
        # Reshape and display image
        img = X[idx].reshape(image_shape)
        ax.imshow(img, cmap='gray')
        
        # Add title with true and predicted labels
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        ax.set_title(f'LLabels: True: {true_label} , Pred: {pred_label}', fontweight='bold', fontsize=10)
        ax.axis('off')
        
        # Add red border for misclassified samples
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Misclassified Samples ({num_samples} examples)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Misclassified samples saved to {save_path}")
    
    plt.show()


def plot_per_class_accuracy(per_class_acc: np.ndarray, save_path: str = None) -> None:
    """
    Plot bar chart of per-class accuracy
    
    Args:
        per_class_acc: Array of accuracies for each class
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    classes = range(len(per_class_acc))
    colors = ['#2ecc71' if acc == max(per_class_acc) else 
             '#e74c3c' if acc == min(per_class_acc) else 
             '#3498db' for acc in per_class_acc]
    
    bars = plt.bar(classes, per_class_acc, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Digit Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy on Test Set', fontsize=14, fontweight='bold')
    plt.xticks(classes)
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch( label=f'Best (Class {np.argmax(per_class_acc)})'),
        Patch( label=f'Worst (Class {np.argmin(per_class_acc)})'),
        Patch( label='Others')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

