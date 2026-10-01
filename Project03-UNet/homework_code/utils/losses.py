"""
loss functions for semantic segmentation.
"""

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance problem.
    
    Formula: FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: focusing parameter (default: 2.0)
        alpha: class weights (default: None)
        ignore_index: index to ignore in loss calculation
    """
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: model predictions (B, C, H, W)
            targets: ground truth masks (B, H, W)
        """
        # compute cross entropy loss (without reduction)
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.alpha, 
            ignore_index=self.ignore_index, 
            reduction='none'
        )
        
        # compute pt (probability of correct class)
        pt = torch.exp(-ce_loss)
        
        # apply focal weight
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


def build_criterion(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    build loss function from config.
    
    supports:
        - "cross_entropy": standard cross entropy loss
        - "weighted_cross_entropy": cross entropy with class weights
        - "focal": focal loss for class imbalance
    """
    loss_type = config["training"].get("loss", "cross_entropy")
    num_classes = config["classes"]["num_classes"]
    
    # Focal Loss for question 1-7
    if loss_type == "focal":
        print("  Using Focal Loss (gamma=2.0)")
        gamma = config["focal_loss"].get("gamma", 2.0)
        alpha = config["focal_loss"].get("alpha", None)
        
        if alpha is not None:
            alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
        
        return FocalLoss(gamma=gamma, alpha=alpha, ignore_index=9)
    
    # Weighted Cross Entropy (with class weights)
    elif "class_weights" in config and config["class_weights"] is not None:
        print("  Using Weighted Cross Entropy Loss")
        weights = torch.tensor(config["class_weights"], dtype=torch.float32).to(device)
        return nn.CrossEntropyLoss(weight=weights, ignore_index=9)
    
    # Standard Cross Entropy (no weights)
    else:
        print("  Using Regular Cross Entropy Loss")
        return nn.CrossEntropyLoss(ignore_index=9)
