# MLP Neural Network for MNIST Classification

A complete from-scratch implementation of a Multilayer Perceptron (MLP) neural network using NumPy for handwritten digit recognition on the MNIST dataset.

## 📋 Features

- **Complete Forward & Backward Propagation** implemented from scratch using NumPy
- **Configurable architecture** (number of layers and neurons per layer)
- **SGD with Momentum** optimization
- **L2 Regularization** to prevent overfitting
- **Early Stopping** to save the best model
- **Data Visualization** including:
  - Random sample display per class
  - Class distribution histogram
  - Training loss and accuracy curves
  - Confusion matrix heatmap
  - Misclassified samples visualization
- **Normalization & Standardization** support
- **Batch Training** with configurable batch size
- **Comprehensive evaluation metrics** (precision, recall, F1-score)

## 🏗️ Project Structure
```
homework_code/
├── Part1/ # Initial data exploration
│ ├── images/ # Output images
│ ├── main.py # Part1 main script
│ ├── requirements.txt # Dependencies
│ ├── test.ipynb # Test notebook
│ └── utils.py # Utility functions
│
├── Part2/ # Main implementation
│ ├── config/
│ │ └── config.yaml # Project configuration
│ ├── data/
│ │ └── dataset/ # MNIST dataset
│ │ ├── mnist_train.csv
│ │ └── mnist_test.csv
│ ├── models/
│ │ └── model.py # MLP implementation
│ ├── scripts/
│ │ ├── train.py # Training loop
│ │ ├── evaluate.py # Evaluation functions
│ │ └── init.py
│ ├── utils/
│ │ ├── config_loader.py # YAML config loader
│ │ ├── visualization.py # Plotting functions
│ │ └── metrics.py # Evaluation metrics
│ ├── notebooks/
│ │ └── exploratory.ipynb # EDA notebook
│ ├── ReportImages/ # Report images
  └── results/ # Training outputs
```


## 🚀 Installation

## Installation

```bash
pip install -r requirements.txt

```
## Outputs
Results saved to Part2/results/:

training_history.png - Loss & accuracy curves

confusion_matrix_test.png

misclassified_samples.png

## 🤝 Contributing
Feel free to modify the architecture, add new features, or experiment with different hyperparameters in config.yaml.

cd Part2
python scripts/main.py
