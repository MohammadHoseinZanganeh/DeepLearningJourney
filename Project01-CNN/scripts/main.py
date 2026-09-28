"""
main.py
Main entry point to run the entire pipeline.
"""

import os
import sys
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.train import train
from utils.visualization import plot_training_curves


def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print(f"Part 2: Block {config['model']['block_type']} - Network")
    print("=" * 50)

    # Train (includes validation and final test evaluation)
    print("\nStarting training...")
    model, history = train(config)

    # Plot training and validation curves
    os.makedirs('./results', exist_ok=True)
    plot_training_curves(
        history['train_loss'], history['val_loss'],
        history['train_acc'], history['val_acc'],
        config['model']['block_type'],
        save_path=f'./results/training_curves_block{config["model"]["block_type"]}.png'
    )


if __name__ == "__main__":
    main()