"""A simple Multi-Layer Perceptron (MLP) classifier with 1 hidden layer.

The same architecture is used for HuBERT features.
"""

import torch.nn as nn


class EmotionMLP(nn.Module):
    """Feed-forward network with 1 hidden layer that predicts an emotion class."""

    def __init__(self, input_size, hidden_size, num_classes):
        """Build the MLP layers.

        Args:
            input_size (int): Size of the input feature vector.
            hidden_size (int): Number of units in hidden layer.
            num_classes (int): Number of emotion classes to predict.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),      # Input layer -> Hidden layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),     # Hidden layer -> Output layer
        )

    def forward(self, feature_vector):
        """Run a forward pass.

        Args:
            feature_vector (torch.Tensor): Batch of feature vectors,
                shape (batch_size, input_size).

        Returns:
            torch.Tensor: Raw class scores (logits), shape (batch_size, num_classes).
        """
        return self.network(feature_vector)