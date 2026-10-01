import numpy as np
from typing import Tuple


def evaluate(model, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Test the model on a dataset without training
    
    Args:
        model: Our MLP model
        X: Input data (num_samples, 784)
        y: True labels (num_samples,)
        
    Returns:
        Loss and accuracy on this dataset
    """
    # Forward pass to get predictions
    output = model.forward(X)
    
    # Calculate loss
    loss = model.compute_loss(output, y)
    
    # Calculate accuracy
    predictions = model.predict(X)
    correct = np.sum(predictions == y)
    accuracy = (correct / len(y)) * 100.0
    
    return loss, accuracy
