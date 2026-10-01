# Speech Emotion Recognition (SER) using Mel-Spectrogram and HuBERT Features

This project implements a Speech Emotion Recognition system that compares two different feature extraction approaches:
1. **Traditional hand-crafted features**: Log-mel spectrograms classified by a Convolutional Neural Network (CNN)
2. **Modern self-supervised features**: HuBERT embeddings classified by a Multi-Layer Perceptron (MLP)

The system is trained and evaluated on the **CREMA-D** dataset, which contains emotional speech recordings from actors.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)

---

## Project Overview

Emotion recognition from speech is a challenging task with applications in human-computer interaction, mental health monitoring, and customer service. This project explores two contrasting approaches:

| Approach | Feature Type | Classifier | Strength |
|----------|-------------|------------|----------|
| Traditional | Log-mel spectrogram | CNN | Interpretable, fast extraction |
| Deep Learning | HuBERT embeddings | MLP | Better performance |

By comparing both methods on the same dataset, we can understand the trade-offs between simplicity and performance.

---

## Features

### Dataset
- **CREMA-D** (Crowdsourced Emotional Multimodal Actors Dataset)
- 7,442 audio clips from 91 actors
- 6 emotion classes: Neutral, Happy, Sad, Angry, Fearful, Disgust
- **Filtered** to 4 classes: Neutral, Happy, Angry, Sad
- **Filtered** to speakers 1001-1021 (subset for faster experimentation)

### Audio Processing
- Fixed duration: 3.0 seconds
- Sampling rate: 16 kHz
- Padding/truncation for uniform input size


### Key Directories Explained

| Directory | Purpose |
|-----------|---------|
| config/ | Stores all configuration settings in YAML format |
| data/ | Dataset handling and loading logic |
| models/ | Neural network architectures (CNN and MLP) |
| scripts/ | Executable Python scripts for running the pipeline |
| utils/ | Helper functions for device management, feature extraction, metrics, and plotting |
| checkpoints/ | Automatically created; stores trained model weights |
| results/ | Automatically created; stores plots (training curves, confusion matrices) |
| data/dataset/ | Automatically created; stores raw audio and cached features |
### Feature Extraction
1. **Log-mel Spectrogram**
   - 64 Mel frequency bands
   - 1024 FFT window size
   - 256 hop length
   - Mean-pooled across time → 64-dimensional feature vector

2. **HuBERT Embeddings**
   - `facebook/hubert-base-ls960` model
   - 768-dimensional embeddings
   - Mean-pooled across time → fixed-size feature vector

---

## Architecture

### Mel-Spectrogram Classifier (CNN)
```python
EmotionCNN(
    Conv2d(1 → 128) → ReLU → BatchNorm → MaxPool → Dropout(0.3)
    Conv2d(128 → 256) → ReLU → BatchNorm → MaxPool → Dropout(0.3)
    Conv2d(256 → 512) → ReLU → BatchNorm → MaxPool → Dropout(0.3)
    Linear(flattened → 128) → ReLU → Dropout(0.3)
    Linear(128 → num_classes)
)
```
### ⚠️ Important: Configuration Changes Require Feature Re-Extraction

If you modify the configuration file (`config/config.yaml`), **you must follow these steps**:

1. **Delete the processed data directory**:
   ```bash
   rm -rf data/dataset/processed/
   ```