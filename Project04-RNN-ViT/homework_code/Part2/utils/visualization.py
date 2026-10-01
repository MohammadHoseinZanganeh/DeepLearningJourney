"""
Visualization utilities for plotting training curves.
"""

import matplotlib.pyplot as plt


def plot_training_history(history, save_path="training_curves.png"):
    """
    Plot training and validation loss and accuracy.

    Args:
        history (dict): Dictionary containing lists of train_loss, val_loss,
                        train_acc, val_acc.
        save_path (str): Path to save the plot image.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs, history['val_loss'], label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(epochs, history['val_acc'], label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Training curves saved to {save_path}")