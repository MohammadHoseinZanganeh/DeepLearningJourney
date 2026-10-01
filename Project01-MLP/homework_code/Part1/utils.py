import numpy as np


def sigmoid(z):
    """
    Sigmoid activation function.

    Args:
        z (np.ndarray): Pre-activation input, any shape.

    Returns:
        np.ndarray: Output in range (0, 1), same shape as z.
    """
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    """
    Derivative of the sigmoid function with respect to z.

    Formula: σ'(z) = σ(z) · (1 - σ(z))

    Used in backpropagation to compute the error signal (delta) at each layer:
        δ = (W^T · δ_next) ⊙ σ'(z)

    Args:
        z (np.ndarray): Pre-activation input, any shape.

    Returns:
        np.ndarray: Sigmoid derivative, same shape as z.
    """
    s = sigmoid(z)
    return s * (1 - s)