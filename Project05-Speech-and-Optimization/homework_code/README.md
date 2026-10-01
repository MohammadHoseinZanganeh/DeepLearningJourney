# Homework 5: Speech Emotion Recognition + Optimization Algorithms

This repository contains two independent parts of Homework 5:

- **Part 1**: Speech Emotion Recognition (SER) using Mel-Spectrogram + CNN and HuBERT + MLP on CREMA-D dataset
- **Part 2**: Comparison of optimization algorithms (SGD, Adam, RMSProp) on the Rosenbrock function

---

## Part 1: Speech Emotion Recognition (SER)

### Overview
Compares two feature extraction approaches for emotion recognition from speech:
1. **Traditional**: Log-mel spectrograms + CNN
2. **Deep Learning**: HuBERT embeddings + MLP

### Dataset
- **CREMA-D** (7,442 audio clips, 91 actors)
- Filtered to 4 emotions: Neutral, Happy, Angry, Sad
- Filtered to speakers 1001-1021

### Key Features
- Fixed audio duration: 3.0 seconds, sampling rate: 16 kHz
- Mel-spectrogram: 64 bands, 1024 FFT, 256 hop length
- HuBERT: `facebook/hubert-base-ls960` (768-dim embeddings)

### How to Run
```bash
cd Part1
python scripts/main.py