import numpy as np
from typing import Dict, List, Tuple
from scripts.evaluate import evaluate

def train_one_epoch(model, X_train: np.ndarray, y_train: np.ndarray, 
                   batch_size: int, learning_rate: float, momentum: float, lambda_l2: float = 0.0) -> Tuple[float, float]:
    """
    Train the model for one complete pass through the training data
    
    Args:
        model: Our MLP model
        X_train: Training images (num_samples, 784)
        y_train: Training labels (num_samples,)
        batch_size: How many samples to process at once
        learning_rate: Step size for weight updates
        
    Returns:
        Average loss and accuracy for this epoch
    """
    # Split data into small batches
    batches = model.create_batches(X_train, y_train, batch_size)
    
    # Track total loss and correct predictions
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Process each batch
    for X_batch, y_batch in batches:
        # Do forward pass, calculate loss, backward pass, and update weights
        batch_loss = model.train_step(X_batch, y_batch, learning_rate, momentum, lambda_l2)
        
        # Accumulate loss (weighted by batch size)
        total_loss += batch_loss * len(X_batch)
        
        # Count correct predictions
        predictions = model.predict(X_batch)
        total_correct += np.sum(predictions == y_batch)
        total_samples += len(y_batch)
    
    # Calculate averages
    avg_loss = total_loss / total_samples
    accuracy = (total_correct / total_samples) * 100.0
    
    return avg_loss, accuracy

def train(model, X_train: np.ndarray, y_train: np.ndarray,
         X_val: np.ndarray, y_val: np.ndarray,
         epochs: int, batch_size: int, learning_rate: float, momentum: float, lambda_l2: float = 0.0,
         early_stopping_patience: int = 10) -> Dict[str, List[float]]:

    """
    Main training loop: train for multiple epochs and track progress
    
    Args:
        model: Our MLP model
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        epochs: How many times to go through the entire training set
        batch_size: Size of mini-batches
        learning_rate: Learning rate for gradient descent
        momentum: Momentum coefficient
        lambda_l2: L2 regularization coefficient
        early_stopping_patience: Stop if no improvement for this many epochs
        
    Returns:
        Dictionary with training history (losses and accuracies)
    """
    # Dictionary to store metrics over time
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
        # 🆕 Early Stopping variables
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_weights = None
    best_biases = None

    print("Starting training...")
    if momentum > 0:
        print(f"• Using Momentum SGD (β={momentum})")
    else:
        print("• Using Standard SGD")
    
    if lambda_l2 > 0:
        print(f"• Using L2 Regularization (λ={lambda_l2})")
    
    if early_stopping_patience > 0:
        print(f"• Early Stopping enabled (patience={early_stopping_patience})")
    
    print("="*70)
    
    # Train for specified number of epochs
    for epoch in range(1, epochs + 1):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, X_train, y_train, batch_size, learning_rate, momentum, lambda_l2
        )
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, X_val, y_val)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 🆕 Early Stopping Logic
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            improved = " ✓"
            
            # Save best model weights
            import copy
            best_weights = copy.deepcopy(model.weights)
            best_biases = copy.deepcopy(model.biases)
        else:
            patience_counter += 1
            improved = f" ({patience_counter}/{early_stopping_patience})"

        # Print progress
        print(f"Epoch {epoch:3d}/{epochs} //  "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
        # 🆕 Check early stopping
        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            print("-."*70)
            print(f"Early stopping activate after {epoch} epochs!")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
            break

    print("_"*70)
    # Restore best weights
    if best_weights is not None:
        model.weights = best_weights
        model.biases = best_biases
        print(f"Restored best model from epoch {best_epoch}")
    
    print("Training completed!")
    
    return history
