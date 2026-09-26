
import numpy as np


# This class implemented for further analysis metrics
def print_classification_metrics(cm: np.ndarray) -> None:
    """
    Calculate and print precision, recall, and F1-score for each class
    
    Args:
        cm: Confusion matrix (num_classes x num_classes)
    """
    num_classes = len(cm)
    
    print("\n Metrics for further analysis ")
    print("Classification Metrics for Each Class")
    print("-"*70)
    print(f"{'Class':<10} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-"*70)
    
    # Store metrics for averaging
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    
    for i in range(num_classes):
        # True Positives: diagonal element
        tp = cm[i, i]
        
        # False Positives: sum of column i, excluding diagonal
        fp = np.sum(cm[:, i]) - tp
        
        # False Negatives: sum of row i, excluding diagonal
        fn = np.sum(cm[i, :]) - tp
        
        # Calculate metrics
        # Precision: of all predictions for class i, how many were correct?
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall: of all true class i samples, how many did we find?
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1-Score: harmonic mean of precision and recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)
        
        print(f"{i:<10} {precision:<15.4f} {recall:<15.4f} {f1:<15.4f}")
    
    # Print averages
    print("-"*70)
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1_scores)
    print(f"{'Average':<10} {avg_precision:<15.4f} {avg_recall:<15.4f} {avg_f1:<15.4f}")
    print("-."*70 + "\n")

def calculate_per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """
    Calculate accuracy for each class from confusion matrix
    
    Args:
        cm: Confusion matrix (num_classes × num_classes)
        
    Returns:
        Array of per-class accuracies (in percentage)
    """
    per_class_acc = []
    for i in range(len(cm)):
        total = np.sum(cm[i, :])
        correct = cm[i, i]
        acc = (correct / total * 100) if total > 0 else 0.0
        per_class_acc.append(acc)
    return np.array(per_class_acc)


def find_worst_class(cm: np.ndarray, y_true: np.ndarray) -> tuple:
    """
    Find the class with lowest accuracy
    
    Args:
        cm: Confusion matrix
        y_true: True labels (for counting total samples)
        
    Returns:
        Tuple of (worst_class, accuracy, error_count)
    """
    per_class_acc = calculate_per_class_accuracy(cm)
    worst_class = np.argmin(per_class_acc)
    
    total_worst = np.sum(y_true == worst_class)
    correct_worst = cm[worst_class, worst_class]
    error_count = total_worst - correct_worst
    
    return worst_class, per_class_acc[worst_class], error_count


def find_most_confused_pairs(cm: np.ndarray, top_k: int = 5) -> list:
    """
    Find the most common misclassification pairs
    
    Args:
        cm: Confusion matrix
        top_k: Number of top confused pairs to return
        
    Returns:
        List of tuples (true_class, predicted_class, count)
    """
    num_classes = len(cm)
    confusions = []
    
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:  # Only off-diagonal elements
                confusions.append((i, j, cm[i, j]))
    
    # Sort by count (descending)
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    return confusions[:top_k]


def get_classification_report_detailed(cm: np.ndarray) -> dict:
    """
    Generate detailed classification report
    
    Args:
        cm: Confusion matrix
        
    Returns:
        Dictionary with precision, recall, f1 for each class
    """
    num_classes = len(cm)
    report = {}
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        report[i] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': np.sum(cm[i, :])
        }
    
    return report