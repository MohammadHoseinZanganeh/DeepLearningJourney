import matplotlib.pyplot as plt

def plot_results(history, save_path="training_results.png"):
    """
    Plot training and validation loss and accuracy curves
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot image
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o', color='blue')
    axes[0].plot(history['val_loss'], label='Validation Loss', marker='s', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o', color='blue')
    axes[1].plot(history['val_acc'], label='Validation Accuracy', marker='s', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  
    plt.close()  
    
    print(f" Plot saved to: {save_path}")

def plot_comparison(results_dict, save_path="model_comparison.png"):
    """
    Plot comparison between different models
    
    Args:
        results_dict: Dictionary with model names as keys and histories as values
        save_path: Path to save the plot image
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['blue', 'green', 'red']
    
    # 1. Loss comparison
    for i, (model_name, history) in enumerate(results_dict.items()):
        axes[0].plot(history['val_loss'], label=model_name, marker='o', color=colors[i % len(colors)])
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Validation Loss Comparison', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Accuracy comparison
    for i, (model_name, history) in enumerate(results_dict.items()):
        axes[1].plot(history['val_acc'], label=model_name, marker='s', color=colors[i % len(colors)])
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Validation Accuracy Comparison', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Comparison plot saved to: {save_path}")