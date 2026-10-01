"""Small helper functions for computing evaluation metrics."""

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def compute_accuracy(predictions, labels):
    """Compute classification accuracy.

    Args:
        predictions (torch.Tensor): Predicted class indices.
        labels (torch.Tensor): True class indices.

    Returns:
        float: Accuracy between 0 and 1.
    """
    num_correct = (predictions == labels).sum().item()
    num_total = labels.size(0)
    return num_correct / num_total


def compute_confusion_matrix(predictions, labels, num_classes):
    """Compute a confusion matrix.

    Args:
        predictions (torch.Tensor): Predicted class indices.
        labels (torch.Tensor): True class indices.
        num_classes (int): Number of classes.

    Returns:
        numpy.ndarray: Confusion matrix of shape (num_classes, num_classes).
    """
    return confusion_matrix(
        labels.cpu().numpy(),
        predictions.cpu().numpy(),
        labels=list(range(num_classes)),
    )


def compute_class_metrics(predictions, labels, class_names):
    """Compute per-class metrics: precision, recall, f1-score, and support.

    Args:
        predictions (torch.Tensor): Predicted class indices.
        labels (torch.Tensor): True class indices.
        class_names (list[str]): List of class names.

    Returns:
        dict: Dictionary containing per-class metrics and overall accuracy.
    """
    y_true = labels.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    
    # Compute classification report
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )
    
    # Extract per-class metrics
    per_class_metrics = {}
    for class_name in class_names:
        if class_name in report:
            per_class_metrics[class_name] = {
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1-score": report[class_name]["f1-score"],
                "support": report[class_name]["support"],
            }
    
    # Overall accuracy
    overall_accuracy = report.get("accuracy", 0.0)
    
    return {
        "per_class": per_class_metrics,
        "overall_accuracy": overall_accuracy,
        "full_report": report,
    }


def print_class_metrics(metrics, class_names):
    """Print per-class metrics in a readable format.

    Args:
        metrics (dict): Output from compute_class_metrics.
        class_names (list[str]): List of class names.
    """
    print("\n" + "=" * 70)
    print("PER-CLASS PERFORMANCE METRICS")
    print("=" * 70)
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    for class_name in class_names:
        if class_name in metrics["per_class"]:
            m = metrics["per_class"][class_name]
            print(f"{class_name:<12} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1-score']:<12.4f} {m['support']:<10}")
    
    print("-" * 70)
    print(f"{'Overall Accuracy':<12} {metrics['overall_accuracy']:.4f}")
    print("=" * 70)