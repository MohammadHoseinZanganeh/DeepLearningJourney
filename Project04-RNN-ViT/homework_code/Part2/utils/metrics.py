"""
Metrics utilities for model evaluation.
"""

import torch


def accuracy(logits, labels):
    """
    Compute accuracy between logits and labels.

    Args:
        logits (torch.Tensor): Model output of shape (B, C).
        labels (torch.Tensor): Ground truth labels of shape (B,).

    Returns:
        float: Accuracy value between 0 and 1.
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total