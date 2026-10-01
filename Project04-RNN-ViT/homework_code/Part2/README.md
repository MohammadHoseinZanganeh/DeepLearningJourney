# Vision Transformer (ViT) for CIFAR-10 Classification

A deep learning project implementing a **Vision Transformer (ViT)** from scratch for image classification on the CIFAR-10 dataset, including hyperparameter experiments, attention map visualization, and comparison with a pretrained ViT-B/16 model.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Installation](#installation)
- [Usage](#usage)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Output Files](#output-files)
- [Dependencies](#dependencies)

---

## Overview

This project implements the **Vision Transformer (ViT)** architecture, originally proposed in *"An Image is Worth 16x16 Words"* (Dosovitskiy et al., 2021), from scratch using PyTorch. The model is trained and evaluated on the CIFAR-10 dataset.

Key features of the project:
- **From-scratch ViT implementation**: Patch Embedding, Multi-Head Attention, Transformer Blocks, and Classification Head.
- **Hyperparameter experimentation**: Configurable `embed_dim`, `num_layers`, `num_heads`, `patch_size`, and `image_size`.
- **Attention Map Visualization**: Custom hooks to extract and visualize attention maps from both the scratch-trained model and a pretrained ViT-B/16.
- **Modular pipeline**: Separate modules for data loading, model definition, training, evaluation, and visualization.

---

## Dataset

**CIFAR-10** — A dataset of 60,000 32×32 color images in 10 classes, with 6,000 images per class.

| Split | Examples | Classes |
|-------|----------|---------|
| Train | 50,000 | 10 |
| Test | 10,000 | 10 |

**Classes:**
```
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

**Data Augmentation:**
- Random Horizontal Flip
- Random Crop with padding
- Normalization (CIFAR-10 mean/std)

The dataset is automatically downloaded and saved to `Part2/data/dataset/` on first run. A `subset_fraction` parameter allows using only a portion of the dataset for quick testing.

---

## Project Structure

```
Part2/
├── config/
│   └── config.yaml                     # Configuration file
├── data/
│   ├── dataset/                        # Saved dataset (auto-downloaded)
│   └── data_loader.py                  # Dataset & DataLoader utilities
├── models/
│   ├── saved_models/                   # Saved model weights
│   └── model.py                        # ViT model definition
├── results/
│   ├── attention/                      # Attention maps (scratch model)
│   ├── attention_pretrained/           # Attention maps (pretrained ViT)
│   ├── BaseModel/                      # Base model results
│   └── OtherModels/                    # Experiment results
├── scripts/
│   ├── evaluate.py                     # Evaluation functions
│   ├── main.py                         # Main entry point
│   ├── train.py                        # Training loop
│   ├── visualize_attention_hook.py     # Attention maps (scratch model)
│   └── vit_pretrained_complete.py      # Attention maps (pretrained model)
├── utils/
│   ├── metrics.py                      # Accuracy metric
│   └── visualization.py                # Training curve plots
└── README.md
```

---

## Model Architecture

The Vision Transformer (ViT) is composed of the following components:

### 1. Patch Embedding (`PatchEmbed`)
- Splits the input image into non-overlapping patches using a 2D convolution.
- Flattens the patches and linearly projects them to `embed_dim`.
- **Shape:** `(B, C, H, W)` → `(B, num_patches, embed_dim)`

### 2. Multi-Head Self-Attention (`MultiHeadAttention`)
- Linear projections for Query (Q), Key (K), and Value (V).
- Scaled dot-product attention with `num_heads` parallel heads.
- Dropout on attention weights and output projection.

### 3. Transformer Encoder Block (`TransformerBlock`)
- **Pre-LayerNorm** architecture:
  - `x = x + Attention(LayerNorm(x))`
  - `x = x + MLP(LayerNorm(x))`
- MLP consists of two linear layers with a GELU activation and dropout.

### 4. Vision Transformer (`VisionTransformer`)
- Prepends a learnable **CLS token** to the patch sequence.
- Adds learnable **positional embeddings**.
- Passes the sequence through `num_layers` Transformer blocks.
- Applies a final LayerNorm and extracts the CLS token representation.
- Classification head: `Linear(embed_dim, num_classes)`.

**Forward Pass:**
```
Image (B, 3, H, W)
  ↓
PatchEmbed → (B, num_patches, embed_dim)
  ↓
Prepend CLS token → (B, num_patches + 1, embed_dim)
  ↓
Add Position Embeddings
  ↓
Transformer Blocks × num_layers
  ↓
LayerNorm
  ↓
Extract CLS token → (B, embed_dim)
  ↓
Classification Head → (B, num_classes)
```

---

## Configuration

All hyperparameters are managed in `config/config.yaml`:

```yaml
model:
  image_size: 32
  patch_size: 4
  in_channels: 3
  embed_dim: 128
  num_heads: 8
  num_layers: 4
  mlp_ratio: 1.0
  num_classes: 10
  dropout: 0.1
  attention_dropout: 0.1

training:
  batch_size: 128
  num_epochs: 2          # Increase for full training (e.g., 30)
  learning_rate: 0.001
  weight_decay: 0.0001
  model_save_path: "models/saved_models/best_model.pth"

data:
  dataset_name: "cifar10"
  data_path: "./data"
  num_workers: 2
  subset_fraction: 0.1   # 1.0 for full dataset
  # resize: 64           # Uncomment to upscale images

device:
  use_cuda: true
```

---

## Installation

### Requirements

```bash
pip install torch torchvision numpy matplotlib tqdm pyyaml
```

### Python Version
- Python 3.8+ (tested on 3.14)

---

## Usage

### 1. Train and Evaluate the Model

Run the main script:

```bash
python scripts/main.py
```

This will:
1. Load configuration from `config.yaml`
2. Download CIFAR-10 (if not cached)
3. Create DataLoaders (with optional subset fraction)
4. Initialize the ViT model based on config
5. Train for the specified number of epochs
6. Evaluate on the test set
7. Save training curves, test results, and model weights with unique names.

### 2. Visualize Attention Maps (Scratch Model)

```bash
python scripts/visualize_attention_hook.py
```

- Loads the trained model weights (`best_model.pth`).
- Extracts attention maps for layers 1, 2, and 4.
- Visualizes the CLS token's attention to patches, showing the first head and the mean of all heads.
- Saves images to `results/attention/`.

### 3. Visualize Attention Maps (Pretrained ViT-B/16)

```bash
python scripts/vit_pretrained_complete.py
```

- Loads the pretrained `vit_b_16` model from `torchvision`.
- Extracts attention maps for layers 4, 8, and 12.
- Visualizes the CLS token's attention, comparing the first head and the mean of all heads.
- Saves images to `results/attention_pretrained/`.

### 4. Switch Hyperparameters

Edit `config/config.yaml` to experiment with different architectures:
```yaml
model:
  embed_dim: 256       # Try 64, 128, 256
  num_layers: 8        # Try 2, 4, 8
  num_heads: 8
  patch_size: 2        # Try 2, 4, 8
```

---

## Training & Evaluation

### Training Loop (`scripts/train.py`)

- **Optimizer:** Adam with weight decay
- **Loss:** CrossEntropyLoss
- **Metrics:** Loss and accuracy per epoch
- **History tracking:** Train/val loss and accuracy

### Evaluation (`scripts/evaluate.py`)

- Evaluates the model on the validation/test set.
- Computes average loss and accuracy.

---

## Results

The following qualitative observations were made from the hyperparameter experiments (30 epochs, full dataset):

- **Base Model:** The base configuration (`embed_dim=128`, `num_layers=4`, `num_heads=8`, `patch_size=4`, `image_size=32`) provides the best balance between model capacity and generalization.
- **Patch Size:** Increasing the patch size to 8 significantly degrades performance, as it reduces the number of patches and the model's ability to capture fine-grained details. Decreasing the patch size to 2 increases the number of patches and computational cost, but does not yield a significant improvement in accuracy.
- **Model Capacity:** Increasing the embedding dimension to 256 or the number of layers to 8 leads to overfitting, resulting in lower test accuracy compared to the base model. Reducing the embedding dimension to 64 also slightly reduces performance.
- **Input Resolution:** Resizing images to 64×64 while keeping `patch_size=4` yields performance comparable to the 32×32 base model, despite the larger number of patches.
- **Attention Visualization (Scratch Model):** Attention maps from layers 1, 2, and 4 show progressive focusing on semantically meaningful patches.
- **Attention Visualization (Pretrained ViT-B/16):** Attention maps from layers 4, 8, and 12 show highly focused and interpretable patterns, demonstrating the benefits of large-scale pretraining.

---

## Output Files

After running the scripts, the following files are generated:

### Results (`results/`)

| Directory | File | Description |
|-----------|------|-------------|
| `BaseModel/` | `training_curves_*.png` | Loss and accuracy curves |
| `BaseModel/` | `test_results_*.txt` | Test loss, accuracy, and config |
| `OtherModels/` | `test_results_*.txt` | Test results for other configurations |
| `attention/` | `img*_attention.png` | Attention maps (scratch model) |
| `attention_pretrained/` | `img*_attention_pretrained.png` | Attention maps (pretrained ViT) |


## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `torchvision` | Dataset loading and pretrained models |
| `numpy` | Numerical operations |
| `matplotlib` | Visualization |
| `tqdm` | Progress bars |
| `pyyaml` | Configuration parsing |

---

## Key Implementation Details

### Patch Embedding
- Uses `nn.Conv2d` with `kernel_size=patch_size` and `stride=patch_size` to extract and project patches efficiently.

### Multi-Head Attention
- Q, K, V are computed in a single linear projection for efficiency.
- Scaled dot-product attention: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`

### Transformer Block
- Pre-LayerNorm is used for better training stability.
- Residual connections around both attention and MLP sub-layers.

### Attention Visualization
- Custom forward hooks are used to capture attention weights.
- The CLS token's attention to patch tokens is reshaped into a 2D grid and visualized as a heatmap.

---

## Authors

This project was developed as part of a Deep Learning course assignment (Part 2).

---

## License

This project is for educational purposes. The CIFAR-10 dataset and pretrained ViT models are subject to their own licensing terms