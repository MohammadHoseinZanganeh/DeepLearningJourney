"""
metrics module for semantic segmentation evaluation.

contains mIoU calculator and other evaluation metrics.
"""

from typing import Dict, List, Any

import torch


class MeanIoU:
    """
    keeps track of predictions and ground truth to compute mIoU.

    ignores class 9 (staff) because it has no samples.

    Attributes:
        num_classes: total number of classes
        ignore_index: class index to ignore (default 9)
        confusion: confusion matrix tensor
    """

    def __init__(self, num_classes: int, ignore_index: int = 9):
        """
        initialize the MeanIoU calculator.

        Args:
            num_classes: total number of classes
            ignore_index: class index to ignore (staff class)
        """
        self.num_classes: int = num_classes
        self.ignore_index: int = ignore_index
        self.reset()

    def reset(self) -> None:
        """clear the confusion matrix"""
        self.confusion: torch.Tensor = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        add one batch to the confusion matrix.

        Args:
            preds: predicted class indices, shape (B, H, W)
            targets: ground truth class indices, shape (B, H, W)
        """
        preds = preds.cpu().view(-1)
        targets = targets.cpu().view(-1)

        # remove pixels we want to ignore (class 9)
        valid = targets != self.ignore_index
        preds = preds[valid]
        targets = targets[valid]

        # update confusion matrix
        idx = targets * self.num_classes + preds
        self.confusion += torch.bincount(
            idx, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, Any]:
        """
        calculate mIoU from confusion matrix.

        Returns:
            dict with keys:
                "miou": mean IoU across valid classes
                "per_class": list of IoU per class (nan for ignored class)
        """
        per_class_iou: List[float] = []
        for c in range(self.num_classes):
            if c == self.ignore_index:
                per_class_iou.append(float("nan"))
                continue
            tp = self.confusion[c, c].item()
            fn = self.confusion[c, :].sum().item() - tp
            fp = self.confusion[:, c].sum().item() - tp
            denom = tp + fp + fn
            iou = tp / denom if denom > 0 else float("nan")
            per_class_iou.append(iou)

        # average only valid classes (ignore nan)
        valid_ious = [v for v in per_class_iou if v == v]
        miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
        return {"miou": miou, "per_class": per_class_iou}