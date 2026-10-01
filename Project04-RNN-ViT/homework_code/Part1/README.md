# Named Entity Recognition (NER) with Recurrent Neural Networks

A deep learning project for Named Entity Recognition (NER) on the CoNLL-2003 dataset, implementing and comparing three bidirectional recurrent neural network architectures: **BiRNN**, **BiLSTM**, and **BiGRU**.

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
- [Output Files](#output-files)
- [Dependencies](#dependencies)

---

## Overview

This project tackles the **Named Entity Recognition (NER)** task, which involves identifying and classifying named entities in text into predefined categories such as:

- **PER** — Person
- **ORG** — Organization
- **LOC** — Location
- **MISC** — Miscellaneous

The project implements a modular pipeline including:
- Data loading and preprocessing
- Vocabulary building with frequency cutoff
- Three RNN-based encoder architectures (BiRNN, BiLSTM, BiGRU)
- Training and validation loops
- Comprehensive evaluation with classification reports
- Visualization of training curves and results

---

## Dataset

**CoNLL-2003** — The shared task dataset for language-independent named entity recognition.

| Split | Examples | Size |
|-------|----------|------|
| Train | 14,041 | 6.9 MB |
| Validation | 3,250 | 1.7 MB |
| Test | 3,453 | 1.6 MB |

**Features:**
- `id`: Sentence identifier
- `tokens`: List of word tokens
- `pos_tags`: Part-of-speech tags (47 classes)
- `chunk_tags`: Syntactic chunk tags (23 classes)
- `ner_tags`: NER labels (9 classes)

**NER Labels (IOB2 scheme):**
```
O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
```

The dataset is automatically downloaded and saved to `data/dataset/conll2003` on first run.

---

## Project Structure

```
Part1/
├── config/
│   └── config.yaml                 # Configuration file
├── data/
│   └── dataset/
│       └── conll2003/              # Saved dataset (auto-downloaded)
├── models/
│   ├── model.py                    # Encoder model definition
│   └── saved_models/               # Saved model weights
├── notebooks/
│   └── exploratory.ipynb           # Dataset exploration notebook
├── results/
│   ├── BiGRU/                      # Results for BiGRU model
│   ├── BiLSTM/                     # Results for BiLSTM model
│   └── BiRNN/                      # Results for BiRNN model
├── scripts/
│   ├── train.py                    # Training loop
│   ├── evaluate.py                 # Evaluation functions
│   └── main.py                     # Main entry point
├── utils/
│   ├── metrics.py                  # Classification report generation
│   ├── table_visualization.py      # Report table as image/text
│   └── visualization.py            # Training curve plots
├── data_loader.py                  # Dataset & DataLoader utilities
└── README.md
```

---

## Model Architecture

The model is a simple **Encoder** consisting of three layers:

```
Input Tokens
     ↓
┌─────────────────┐
│   Embedding     │  (vocab_size → embedding_size)
└─────────────────┘
     ↓
┌─────────────────┐
│  Bidirectional  │  (RNN / LSTM / GRU)
│      RNN        │  (embedding_size → hidden_size × 2)
└─────────────────┘
     ↓
┌─────────────────┐
│  Linear (FC)    │  (hidden_size × 2 → num_labels)
└─────────────────┘
     ↓
   Logits [batch, seq_len, num_labels]
```

**Supported RNN types:**
- `birnn` — Bidirectional vanilla RNN
- `bilstm` — Bidirectional LSTM
- `bigru` — Bidirectional GRU

**Key features:**
- Embedding layer with `padding_idx=0`
- Batch-first processing
- Optional dropout (disabled when `num_layers=1`)
- Attention mask support for padding

---

## Configuration

All hyperparameters are managed in `config/config.yaml`:

```yaml
model:
  embedding_size: 64
  hidden_size: 64
  num_layers: 1
  dropout: 0.0
  num_labels: 9
  rnn_type: 'bigru'          # 'birnn' | 'bilstm' | 'bigru'

training:
  batch_size: 32
  num_epochs: 10
  learning_rate: 0.001
  cutoff: 5                  # Minimum token frequency

data:
  dataset_name: "conll2003"
  data_path: "data/dataset/conll2003"

device:
  use_cuda: true             # Use GPU if available
```

---

## Installation

### Requirements

```bash
pip install torch datasets scikit-learn matplotlib tqdm pyyaml numpy
```

### Python Version
- Python 3.8+ (tested on 3.14)

---

## Usage

### 1. Explore the Dataset (Optional)

Open `notebooks/exploratory.ipynb` to inspect:
- Sample sentences and their tags
- Label mappings for NER, POS, and chunk tags
- Dataset statistics

### 2. Train the Model

Run the main script:

```bash
python scripts/main.py
```

This will:
1. Load configuration from `config.yaml`
2. Download the CoNLL-2003 dataset (if not cached)
3. Build the vocabulary with cutoff filtering
4. Create train/validation DataLoaders
5. Initialize the model based on `rnn_type`
6. Train for the specified number of epochs
7. Generate and save evaluation reports
8. Save the trained model weights

### 3. Switch Between Architectures

Simply change `rnn_type` in `config.yaml`:
```yaml
rnn_type: 'bilstm'   # or 'birnn' or 'bigru'
```

### 4. Test the Data Pipeline

```bash
python data_loader.py
```

---

## Training & Evaluation

### Training Loop (`scripts/train.py`)

- **Optimizer:** Adam
- **Loss:** CrossEntropyLoss with `ignore_index=-100` (ignores padding)
- **Metrics:** Loss and token-level accuracy per epoch
- **History tracking:** Train/val loss and accuracy

### Evaluation (`utils/metrics.py`)

- Token-level accuracy (masking out padding)
- Full classification report with:
  - Precision, Recall, F1-score per label
  - Macro and weighted averages
  - Support (number of tokens per class)

---

## Output Files

After running `main.py`, the following files are generated:

### Results (`results/{model_name}/`)

Each model has its own folder named after the model type: `BiGRU`, `BiLSTM`, or `BiRNN`.

| File | Description |
|------|-------------|
| `training_results_{model_name}.png` | Loss and accuracy curves |
| `classification_report_{model_name}.png` | Report table as image |
| `classification_report_{model_name}.txt` | Report as text file |

### Saved Models (`models/saved_models/`)

| File | Description |
|------|-------------|
| `{model_name}_{timestamp}.pth` | Model state dict with metadata |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `datasets` | Hugging Face dataset loading |
| `scikit-learn` | Classification metrics |
| `matplotlib` | Visualization |
| `tqdm` | Progress bars |
| `pyyaml` | Configuration parsing |
| `numpy` | Numerical operations |

---

## Key Implementation Details

### Vocabulary Building

- Special tokens: `<pad>`, `<unk>`, `<s>`, `</s>`
- Tokens with frequency < `cutoff` are mapped to `<unk>`
- Padding index = 0 for embedding layer

### Collate Function

- Dynamic padding to the maximum sequence length in each batch
- Labels padded with `-100` to be ignored by CrossEntropyLoss
- Attention mask: 1 for real tokens, 0 for padding

### Training

- Gradient descent with Adam optimizer
- CrossEntropyLoss with `ignore_index=-100`
- Accuracy computed only on non-padding tokens

---

## Authors

This project was developed as part of a Deep Learning course assignment (Part 1).

---

## License

This project is for educational purposes. The CoNLL-2003 dataset is subject to its own licensing terms.