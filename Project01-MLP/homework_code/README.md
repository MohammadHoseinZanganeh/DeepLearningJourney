# Project01 — Multi-Layer Perceptron (MLP)

A from-scratch implementation of a Multilayer Perceptron (MLP) using NumPy for handwritten digit recognition on the **MNIST** dataset. This project contains two parts.

## Part 1 — Simple 3-Layer Neural Network

A minimal neural network with 2 hidden layers and sigmoid activation, trained with manual backpropagation from scratch.

- Architecture: Input (2) → Hidden (2) → Hidden (3) → Output (1)
- Manual forward and backward pass in pure NumPy

📄 Full details: [Part 1 README](homework_code/Part1/Readme.md)

## Part 2 — MLP for MNIST Classification

A complete MLP implementation for MNIST digit classification, with a modular pipeline for training, evaluation, and visualization.

- Configurable architecture (layers & neurons)
- SGD with Momentum, L2 Regularization, Early Stopping
- Data normalization & batch training
- Visualization: loss/accuracy curves, confusion matrix, misclassified samples
- Evaluation metrics: precision, recall, F1-score

📄 Full details: [Part 2 README](homework_code/Part2/Readme.md)

---
