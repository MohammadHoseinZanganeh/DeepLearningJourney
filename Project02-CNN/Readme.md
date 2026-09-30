

# Handwritten Digit Classification with Modern CNN Architectures

Implementation and comparison of three modern CNN building blocks (Residual, Inception, and ResNeXt) for classifying handwritten digits (MNIST dataset), plus a transfer learning experiment on Fashion MNIST.

## Project Structure

```homework_code/
├── config/
│   └── config.yaml              # Hyperparameters and training settings
├── data/
│   ├── data_loader.py            # Custom Dataset + Fashion MNIST loader
│   ├── dataset/                  # MNIST CSV files
│   └── fashion/                  # Fashion MNIST .gz files
├── models/
│   ├── model.py                  # Block A, B, C + TransferModel
│   └── saved_models/             # Trained model checkpoints (.pth)
├── notebooks/
│   └── test_eval.ipynb           # Notebook for quick model evaluation
├── scripts/
│   ├── main.py                   # Train CNN on MNIST
│   ├── train.py                  # Training loop
│   ├── train_transfer.py         # Transfer learning on Fashion MNIST
│   └── evaluate.py               # Standalone evaluation script
├── utils/
│   ├── metrics.py                # Accuracy calculation
│   └── visualization.py          # Plots, confusion matrix, filters
├── results/                      # Output images
├── requirements.txt
└── README.md
```

## Architectures

| Block | Type | Key Idea |
|-------|------|----------|
| A | Residual | Skip connections (ResNet-style) |
| B | Inception | Parallel heterogeneous branches |
| C | ResNeXt | Parallel homogeneous branches + residual |

All architectures share the same base structure (Table 1) and differ only in steps 3, 4, and 6.

## Transfer Learning (Fashion MNIST)

A pretrained MNIST model is reused for Fashion MNIST classification:
- Conv layers (99.9% of parameters) are **frozen** as feature extractors
- Only the final Linear layer (0.1%) is retrained on clothing images
- Fashion MNIST data is read directly from `.gz` files using a custom loader

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train on MNIST:
```bash
cd scripts
python main.py
```

3. Transfer learning on Fashion MNIST:
```bash
python train_transfer.py
```

## Results Summary

**MNIST (60,000 samples, 10,000 test):**

| Architecture | Optimizer | Test Accuracy |
|-------------|-----------|---------------|
| Block A | SGD | 99.66% |
| Block A | Adam | 99.53% |
| Block B | Adam | 99.39% |
| Block C | Adam | 99.15% |

**Transfer Learning — Fashion MNIST:**

| Pretrained Model | Frozen Params | Test Accuracy |
|-----------------|---------------|---------------|
| Block A (MNIST) | 99.9% | 76.50% |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy, pandas, matplotlib
- pyyaml, seaborn, scikit-learn

## Requirements
Install all dependencies with:

```bash
pip install -r requirements.txt
```

```bash
torch>=2.11.0
torchvision>=0.26.0
numpy>=2.4.0
pandas>=3.0.0
matplotlib>=3.10.0
pyyaml>=6.0.0
pillow>=12.1.0
scikit-learn>=1.8.0
```

