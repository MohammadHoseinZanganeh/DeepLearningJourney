# 🧠 U-Net for Football Semantic Segmentation

This repository contains a complete implementation of a **U-Net** model for semantic segmentation of football (soccer) match images, developed as part of a deep learning homework assignment. The project supports **two datasets** with different labeling structures, **4 model configurations** (bilinear/transposed upsample × skip connections on/off), and optional **Batch Normalization**.

---

## 📁 Project Structure

```
homework_code/                     # Root directory of the project
│
├── config/                        # Configuration files
│   └── config.yaml               # Main settings (paths, model params, training params)
│
├── data/                          # Data loading module
│   ├── dataset1/                 # First dataset (images & masks in separate folders)
│   ├── dataset2/                 # Second dataset (images & masks in same folder)
│   └── data_loader.py            # Dataset class, DataLoader builder, color mapping
│
├── models/                        # Model architecture
│   └── model.py                  # U-Net implementation (DoubleConv, SimpleUNet)
│
├── notebooks/                     # Jupyter notebooks for exploration
│   └── exploratory.ipynb         # Class distribution analysis and data visualization
│
├── results/                       # Training outputs (generated)
│   └── {run_name}/               # Each experiment gets its own folder
│       ├── best_model.pth        # Best model checkpoint
│       ├── history.json          # Per-epoch loss and mIoU
│       ├── config_info.txt       # Configuration summary
│       └── training_curves.png   # Loss and mIoU plots
│
├── scripts/                       # Executable scripts
│   ├── main.py                   # Train a single configuration (CLI)
│   ├── train.py                  # Training loop, validation, and core logic
│   ├── evaluate.py               # Evaluate trained models and generate visualizations
│   └── run_all_experiments.py    # Run all 4 experiments sequentially
│
├── utils/                         # Utility modules
│   ├── losses.py                 # Loss functions (CrossEntropy, Weighted CE, Focal)
│   ├── metrics.py                # mIoU calculator (MeanIoU class)
│   └── visualization.py          # Plotting training curves and predictions
│
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

---

## 🏗️ Model Architecture

The U-Net is implemented with:

- **3 encoder levels**: `3 → 64 → 128 → 256` channels
- **Bottleneck**: `256 → 512`
- **3 decoder levels**: `512 → 256 → 128 → 64`  
- **Output layer**: `1×1 convolution` to `num_classes` logits

Each block (`DoubleConv`) uses:
- Two `3×3` convolutions with padding=1 (spatial dimensions are preserved)
- Optional **Batch Normalization** (configurable)
- **ReLU** activation

### Upsampling Methods (configurable)

| Method | Implementation | Description |
|--------|---------------|-------------|
| **Bilinear** | `F.interpolate(..., mode='bilinear')` + `Conv2d(3×3)` | Size is doubled via interpolation, then channels are halved with a 3×3 convolution. |
| **Transposed** | `ConvTranspose2d(kernel_size=2, stride=2)` | Simultaneously upsamples and reduces channel count. |

### Skip Connections (configurable)

- When **enabled**: Encoder feature maps are concatenated with decoder features.
- When **disabled**: A **zero tensor** of the same shape is used instead, effectively removing the skip connection.

---

## 📊 Datasets & Preprocessing

Two datasets with different structures are unified:

| Dataset | Structure |
|---------|-----------|
| **Dataset 1** | Images and masks in separate folders, same filename |
| **Dataset 2** | Images and masks in the same folder, masks end with `___fuse.png` |

### Color-to-Class Mapping

A **look-up table (LUT)** of size `256^3 = 16,777,216` is built for fast RGB → class conversion:

```python
key = r * 65536 + g * 256 + b
class_mask = lut[keys]   # Fully vectorized, O(1) per pixel
```

This is significantly faster than looping over pixels.

### Data Split

Data is split using `train_test_split` with:
- `stratify=labels` to preserve the proportion of each dataset in both train and validation sets.
- `test_size=0.1` (configurable).
- `random_state=42` for reproducibility.

### Augmentation (applied only during training)

- **Horizontal flip** (50% probability) – applied to both image and mask.
- **Brightness adjustment** (50% probability, factor 0.8–1.2) – image only.
- **Contrast adjustment** (50% probability, factor 0.8–1.2) – image only.

Masks are resized using `Image.NEAREST` to preserve class labels.

---

## 🧪 Experiments

Four configurations are supported and can be trained using `run_all_experiments.py`:

| # | Upsample | Skip Connections |
|---|----------|------------------|
| 1 | Bilinear | ON  |
| 2 | Bilinear | OFF |
| 3 | Transposed | ON  |
| 4 | Transposed | OFF |

Each experiment saves:
- `best_model.pth` – best model checkpoint
- `history.json` – per-epoch loss and mIoU
- `config_info.txt` – configuration summary
- `training_curves.png` – loss & mIoU plots

---

## ⚙️ Loss Functions

The following losses are supported (configurable via `config.yaml`):

- **CrossEntropyLoss** – standard (default)
- **Weighted CrossEntropy** – with class weights
- **Focal Loss** – for class imbalance (gamma=2.0)

Class **9 (Staff)** is ignored (`ignore_index=9`) in both loss and mIoU calculation since it has no samples.

---

## 📈 Evaluation Metric

**mIoU (mean Intersection over Union)** is calculated using a confusion matrix:

```python
IoU = TP / (TP + FP + FN)
mIoU = mean(IoU over valid classes)
```

---

## 🚀 Usage

### Train a Single Configuration

```bash
python scripts/main.py --upsample bilinear --skip true
python scripts/main.py --upsample transposed --skip false --bn true
```

Options:
- `--upsample {bilinear,transposed}`
- `--skip {true,false}`
- `--bn {true,false}` (optional, default: false)
- `--config CONFIG` (optional, default: `config/config.yaml`)

### Run All Experiments

```bash
python scripts/run_all_experiments.py
```

### Evaluate a Model

```bash
python scripts/evaluate.py --model bilinear_skipOn
```

### Visualize Predictions

Evaluation script automatically generates prediction visualizations for the best model.

---

## 📝 Key Implementation Notes

- LUT-based RGB-to-class conversion provides **~100× speedup** over per-pixel dictionary lookups.
- `Image.NEAREST` is used for mask resizing to **preserve class labels**.
- Skip connections can be **disabled by zeroing out** the skip tensor, allowing fair comparison.
- Batch Normalization is **optional** and can be toggled via config.
- The model is **flexible**: supports 4 configurations, BN on/off, and 3 loss types.

---

## 📂 Results

All training outputs are saved under `results/{run_name}/` where `run_name` follows:

```
{upsample}_skip{On/Off}_bn{On/Off}
```

Example: `bilinear_skipOn_bnOff`

---

## 📚 Dependencies

See `requirements.txt`. Key packages:
- PyTorch
- torchvision
- NumPy
- Pillow
- scikit-learn
- Matplotlib
- PyYAML