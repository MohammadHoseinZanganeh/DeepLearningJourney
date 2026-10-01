"""
Main script for training and evaluating Vision Transformer on CIFAR-10.

This script loads the configuration, creates the data loaders,
instantiates the ViT model, runs the training loop, evaluates on the test set,
and saves the results (plots, test report, and model weights).
All outputs are saved with unique names based on the model hyperparameters.
"""

import os
import sys
import yaml
import torch

# Add the project root to Python path so that modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import get_data_loaders
from models.model import VisionTransformer
from scripts.train import train_model
from scripts.evaluate import evaluate
from utils.visualization import plot_training_history


def load_config(config_path="config/config.yaml"):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file (default: config/config.yaml).

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Main execution function.
    """

    # Load configuration from the default file
    config = load_config()
    print("=" * 50)
    print("CONFIGURATION LOADED")
    print("=" * 50)
    print(config)

    # Set the device (GPU if available and enabled, else CPU)
    use_cuda = config['device'].get('use_cuda', True)
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    print(f"\nUsing device: {device}")

    # Create data loaders
    # resize is optional; if present in config, images will be resized.
    # subset_fraction can be used for quick testing.
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=config['training']['batch_size'],
        num_workers=config['data'].get('num_workers', 2),
        subset_fraction=config['data'].get('subset_fraction', 1.0),
        resize=config['data'].get('resize', None)   # None means no resizing
    )

    # Create the Vision Transformer model with parameters from config
    model_config = config['model']
    model = VisionTransformer(
        image_size=model_config['image_size'],
        patch_size=model_config['patch_size'],
        in_channels=model_config['in_channels'],
        num_classes=model_config['num_classes'],
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        mlp_ratio=model_config['mlp_ratio'],
        dropout=model_config.get('dropout', 0.0),
        attention_dropout=model_config.get('attention_dropout', 0.0)
    )
    model = model.to(device)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train the model
    # The training loop runs for the specified number of epochs.
    # Best model is tracked but not saved inside the training function;
    # we will save the final model later with a unique name.
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Prepare unique output names based on configuration
    # This ensures that results from different experiments do not overwrite.
    embed = config['model']['embed_dim']
    layers = config['model']['num_layers']
    heads = config['model']['num_heads']
    patch = config['model']['patch_size']

    # Create the results directory if it does not exist
    os.makedirs("results", exist_ok=True)

    # Unique plot filename
    plot_name = f"training_curves_embed{embed}_layers{layers}_heads{heads}_patch{patch}.png"
    plot_path = os.path.join("results", plot_name)

    # Unique results text file name
    result_name = f"test_results_embed{embed}_layers{layers}_heads{heads}_patch{patch}.txt"
    result_path = os.path.join("results", result_name)

    # Save the training curves (loss and accuracy plots)
    plot_training_history(history, save_path=plot_path)

    # Evaluate the model on the test set
    print("\n" + "=" * 50)
    print("EVALUATING ON TEST SET")
    print("=" * 50)
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # Save test results to a unique text file
    with open(result_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("TEST RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write("\n")
        f.write("=" * 50 + "\n")
        f.write("TRAINING HISTORY (LAST EPOCH)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Final Train Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}\n")
        f.write(f"Final Validation Loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}\n")
        f.write("\n")
        f.write("=" * 50 + "\n")
        f.write("CONFIGURATION USED\n")
        f.write("=" * 50 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("=" * 50 + "\n")
        f.write("BEST VALIDATION ACCURACY (from training)\n")
        f.write("=" * 50 + "\n")
        # The best_val_acc is not directly stored in history; we can compute it.
        best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")

    print(f"Results saved to {result_path}")

    # Save the final model with a unique name that includes hyperparameters
    # and the test accuracy, so that it is easy to identify later.
    os.makedirs("models/saved_models", exist_ok=True)
    model_name = (
        f"vit_embed{embed}_"
        f"layers{layers}_"
        f"heads{heads}_"
        f"patch{patch}_"
        f"acc{test_acc:.3f}.pth"
    )
    model_save_path = os.path.join("models/saved_models", model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()