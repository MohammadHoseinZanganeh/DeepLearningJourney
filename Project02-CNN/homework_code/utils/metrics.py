"""
metrics.py
Custom evaluation metrics for classification tasks.

Provides helper functions to compute raw correct/total counts and
accuracy percentages. These are used during training, validation,
and final test evaluation.
"""

import torch


def accuracy(outputs, labels):
    """
    Count the number of correct predictions in a batch.

    Args:
        outputs: Raw logits from the model, shape (batch_size, num_classes).
        labels: Ground-truth class indices, shape (batch_size,).

    Returns:
        tuple: (correct_count, total_samples)
            - correct_count: Number of samples where the top prediction matches the label.
            - total_samples: Total number of samples in the batch.
    """
    # Get the predicted class by taking the index with the highest logit value.
    _, preds = outputs.max(1)
    correct = preds.eq(labels).sum().item()
    total = labels.size(0)
    return correct, total


def calculate_accuracy(correct, total):
    """
    Convert correct/total counts into a percentage.

    Args:
        correct: Total number of correctly classified samples.
        total: Total number of samples evaluated.

    Returns:
        float: Accuracy as a percentage (0.0 to 100.0). Returns 0.0 if total is zero.
    """
    return 100.0 * correct / total if total > 0 else 0.0