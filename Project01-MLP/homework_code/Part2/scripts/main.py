
import os
import numpy as np

from utils.config_loader import load_config, get_model_config, get_training_config, get_data_config, get_paths_config
from data.data_loadder import read_csv_dataset, split_validation
from models.model import MLP
from scripts.train import train
from scripts.evaluate import evaluate
from utils.visualization import plot_training_history, plot_confusion_matrix
from utils.metrics import print_classification_metrics

# # Helper function to interpretation for common confusions 
# def get_confusion_interpretation(true_digit: int, pred_digit: int) -> str:
#     """Provide interpretation for common confusions"""
#     interpretations = {
#         (4, 9): "Similar loop shape",
#         (9, 4): "Similar loop shape",
#         (3, 8): "Both have two loops",
#         (8, 3): "Incomplete loops",
#         (7, 1): "Similar vertical line",
#         (1, 7): "Extended top stroke",
#         (5, 3): "Similar curved shape",
#         (3, 5): "Incomplete top curve",
#         (2, 7): "Similar angular shape",
#         (8, 5): "Similar bottom loop",
#         (0, 6): "Open loop at top",
#         (6, 0): "Closed loop",
#     }
#     return interpretations.get((true_digit, pred_digit), "Visual similarity")


def main():
    """Main function to run the complete training pipeline."""
    
    # Load configuration
    config = load_config('config/config.yaml')
    model_config = get_model_config(config)
    training_config = get_training_config(config)
    data_config = get_data_config(config)
    paths_config = get_paths_config(config)
    
    # Create output directories
    os.makedirs(paths_config['results_dir'], exist_ok=True)
    
    # Load data from CSV files
    print("Loading MNIST dataset from CSV files...")
    train_csv = os.path.join(paths_config['data_dir'], 'mnist_train.csv')
    test_csv = os.path.join(paths_config['data_dir'], 'mnist_test.csv')
    
    X_train, Y_train = read_csv_dataset(train_csv)
    X_test, Y_test = read_csv_dataset(test_csv)
    
    # Normalize if specified
    if training_config['normalization']:
        print("Normalizing data...")
        X_train = X_train / X_train.max()
        X_test = X_test / X_test.max()
    if training_config['standardation']:
        print("Standardation data...")
        X_train = (X_train - X_train.mean()) / X_train.std()
        X_test  = (X_test  - X_train.mean()) / X_train.std() #! wrong implementation

    
    # Split into train and validation
    print(f"Splitting data with validation ratio: {data_config['validation_split']}")
    X_train_final, X_val, Y_train_final, Y_val = split_validation(
        X_train, Y_train, val_ratio=data_config['validation_split']
    )
    
    print(f"\nDataset shapes:")
    print(f"  Training set: {X_train_final.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}")
    
    # Create MLP model
    input_size = X_train_final.shape[1]
    output_size = 10
    layer_sizes = [input_size] + model_config['hidden_layers'] + [output_size]
    # Ex out: [784, 128, 64, 10]

    print(f"\nCreating MLP with architecture: {layer_sizes}")
    mlp = MLP(
        layer_sizes=layer_sizes,
        weight_mean=model_config['weight_mean'],
        weight_std=model_config['weight_std'],
        bias_init=model_config['bias_init'],
        momentum=model_config['momentum']

    )
    
    # Train model
    print(f"\nStarting training for {training_config['epochs']} epochs...")
    history = train(
        model=mlp,
        X_train=X_train_final,
        y_train=Y_train_final,
        X_val=X_val,
        y_val=Y_val,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        learning_rate=training_config['learning_rate'],
        momentum=training_config['momentum'],  
        lambda_l2=training_config['lambda_l2'],  # L2 regularization
        early_stopping_patience=training_config['early_stopping_patience']  # Early stopping 
    )
    
    # Plot training history
    plot_path = os.path.join(paths_config['results_dir'], 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    print(f"Training history saved to: {plot_path}")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = evaluate(mlp, X_test, Y_test)
    print(f"Final Test Results: Loss={test_loss:.4f}, Accuracy={test_acc:.2f}%")
    
    # Confusion matrix on validation set
    print("\nGenerating confusion matrix...")
    y_val_pred = mlp.predict(X_val)
    cm_path = os.path.join(paths_config['results_dir'], 'confusion_matrix_val.png')
    cm = plot_confusion_matrix(Y_val, y_val_pred, num_classes=10, save_path=cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Print classification metrics
    print("\nClassification Metrics:")
    print_classification_metrics(cm)

    
    print("\n")
    print("DETAILED TEST SET ANALYSIS!")
    print("="*70)
    
    #? SECTION: DETAILED TEST SET ANALYSIS
    # Get predictions on test data
    test_predictions = mlp.predict(X_test)
    
    # Calculate confusion matrix for test data
    from utils.visualization import calculate_confusion_matrix
    cm_test = calculate_confusion_matrix(Y_test, test_predictions, num_classes=10)
    
    # Plot and save test confusion matrix
    cm_test_path = os.path.join(paths_config['results_dir'], 'confusion_matrix_test.png')
    plot_confusion_matrix(Y_test, test_predictions, num_classes=10, save_path=cm_test_path)
    print(f"\nTest confusion matrix saved to: {cm_test_path}")
    

    # Per-Class Accuracy Analysis
    print("\n" )
    print("Per-Class Accuracy  (Part 2.16)")
    print("-"*70)
    
    print(f"{'Class':<10} {'Accuracy':<15} {'Correct':<15} {'Total':<15}")
    print("-"*55)
    
    per_class_acc = []  # Store accuracy for each class
    for cls in range(10):
        total_cls = np.sum(Y_test == cls)
        correct_cls = cm_test[cls, cls]
        acc_cls = (correct_cls / total_cls * 100) if total_cls > 0 else 0
        per_class_acc.append(acc_cls)
        print(f"{cls:<10} {acc_cls:<15.2f}% {correct_cls:<15} {total_cls:<15}")
    

    # Worst Performing Class Analysis
    print("\n" )
    print("Worst Performing Class  (Part 2.17)")
    print("-"*70)
    
    # Find the class with lowest accuracy
    worst_class = np.argmin(per_class_acc)
    worst_acc = per_class_acc[worst_class]
    total_worst = np.sum(Y_test == worst_class)
    correct_worst = cm_test[worst_class, worst_class]
    error_count = total_worst - correct_worst
    
    print(f"Worst performing class: Digit {worst_class}")
    print(f"   Accuracy: {worst_acc:.2f}%")
    print(f"   Misclassified: {error_count} out of {total_worst} samples")
    print(f"\nPossible reasons for poor performance on digit {worst_class}:")
    print(f"   - Similar shape to other digits (e.g., 8<->3, 9<->4, 7<->1)")
    print(f"   - High variance in writing styles for this digit")
    print(f"   - Ambiguous features in the dataset")
    

    # Part 2.18: Most Confused Class Pairs
    print("\n" )
    print("MOST CConfused Class Pairs (Part 2.18)")
    print("-"*70)
    
    # Find top 5 most confused class pairs
    confusions = []
    for i in range(10):
        for j in range(10):
            if i != j:  # Only off-diagonal elements
                confusions.append((i, j, cm_test[i, j]))
    
    # Sort by count in descending order
    confusions.sort(key=lambda x: x[2], reverse=True)
    top_confusions = confusions[:5]
    
    print("Top 5 most confused class pairs:")
    print(f"{'True -> Pred':<15} {'Count':<10} ")
    print("-"*50)
    for i, (true_cls, pred_cls, count) in enumerate(top_confusions, 1):
        # interpretation = get_confusion_interpretation(true_cls, pred_cls)
        print(f"{i}. {true_cls} -> {pred_cls}->{count:<10}") #If helper function activatr add {interpretation} in the arg
    

    # Part 2.19: Misclassified Samples Visualization

    print("\n" )
    print("Misclassified Samples Visualization (Part 2.19)")
    print("-"*70)
    
    # Find indices of misclassified samples
    misclassified_mask = Y_test != test_predictions
    misclassified_indices = np.where(misclassified_mask)[0]
    
    if len(misclassified_indices) > 0:
        # Select 10 random misclassified samples
        num_show = min(10, len(misclassified_indices))
        selected = np.random.choice(misclassified_indices, num_show, replace=False)
        
        # Plot the samples
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i, idx in enumerate(selected):
            axes[i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
            axes[i].set_title(f'True: {Y_test[idx]} | Pred: {test_predictions[idx]}', fontweight='bold')
            axes[i].axis('off')
            
            # Add red border for misclassified samples
            for spine in axes[i].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
        
        # Hide unused subplots
        for i in range(num_show, 10):
            axes[i].axis('off')
        
        plt.suptitle('Misclassified Samples (True vs Predicted)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        mis_path = os.path.join(paths_config['results_dir'], 'misclassified_samples.png')
        plt.savefig(mis_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Misclassified samples saved to: {mis_path}")
        print(f"  Displayed {num_show} out of {len(misclassified_indices)} misclassified samples")
    

    # Final Summary
    print("\n" )
    print("Final Summary ")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Best Class: Digit {np.argmax(per_class_acc)} ({max(per_class_acc):.2f}%)")
    print(f"Worst Class: Digit {worst_class} ({worst_acc:.2f}%)")
    print(f"Top Confusion: {top_confusions[0][0]} -> {top_confusions[0][1]} ({top_confusions[0][2]} times)")
    
    print(f"\nAll results saved to: {paths_config['results_dir']}")
  
    
    print(f"\n All results saved to: {paths_config['results_dir']}")

if __name__ == "__main__":
    main()

