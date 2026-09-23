
import numpy as np
from typing import List, Tuple

class MLP:
    """Simple neural network with backpropagation"""

    def __init__(self, layer_sizes: List[int], weight_mean: float = 0.0, weight_std: float = 0.1, bias_init: float = 0.0, momentum: float = 0.9):
        """Initialize the network
        
        Args:
            layer_sizes: Number of neurons in each layer. Example: [784, 128, 64, 10]
            weight_mean: Mean for weight initialization
            weight_std: Standard deviation for weight initialization
            bias_init: Initial value for biases
        """
        self.layer_sizes = layer_sizes
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.weight_velocity: List[np.ndarray] = []
        self.bias_velocity: List[np.ndarray] = []
        self.weight_mean = weight_mean
        self.weight_std = weight_std
        self.bias_init = bias_init
        self.momentum = momentum
        self._initialize_weights()

        # Store intermediate values for backward pass
        self.activations: List[np.ndarray] = []  # Output of each layer after activation
        self.z_values: List[np.ndarray] = []     # Output of each layer before activation

        # Store gradients
        self.weight_gradients: List[np.ndarray] = []
        self.bias_gradients: List[np.ndarray] = []

    def _initialize_weights(self) -> None:
        """Initialize weights and biases"""
        for i in range(len(self.layer_sizes) - 1):
            # Weights: matrix (input_size × output_size)
            w = np.random.normal(
                loc=self.weight_mean,
                scale=self.weight_std,
                size=(self.layer_sizes[i], self.layer_sizes[i + 1])
            )
            # Biases: one row (1 × output_size)
            b = np.full(
                shape=(1, self.layer_sizes[i + 1]),
                fill_value=self.bias_init
            )
            self.weights.append(w)
            self.biases.append(b)
            self.weight_velocity.append(np.zeros_like(w))
            self.bias_velocity.append(np.zeros_like(b))

    def relu(self, z: np.ndarray) -> np.ndarray:
        """ReLU activation function: sets negative values to zero"""
        return np.maximum(0, z)

    def relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU: 1 if z > 0, else 0"""
        return (z > 0).astype(float)

    def softmax(self, z: np.ndarray) -> np.ndarray:
        """Convert output to probabilities (sum to 1)"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: from input to output
        
        Args:
            X: Input data (batch_size, input_size)
            
        Returns:
            Predicted probabilities (batch_size, output_size)
        """
        # Clear previous values
        self.activations = []
        self.z_values = []

        current_activation = X  # Start with input

        # Go through each layer
        for i in range(len(self.weights)):
            # Compute: Z = X × W + b
            z = current_activation @ self.weights[i] + self.biases[i]
            self.z_values.append(z)

            # Apply activation function
            if i < len(self.weights) - 1:
                # Hidden layers: ReLU
                current_activation = self.relu(z)
            else:
                # Output layer: Softmax
                current_activation = self.softmax(z)

            # Store for backward pass
            self.activations.append(current_activation)

        return current_activation

    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray, lambda_l2: float = 0.0) -> float:
        """Compute error (Cross-Entropy Loss function implementation)
        
        Args:
            predictions: Network predictions (batch_size, output_size)
            targets: True labels (batch_size,)
            lambda_l2: L2 regularization coefficient (0.0 = no regularization)
            
        Returns:
            Average loss
        """
        batch_size = predictions.shape[0]

        # Convert targets to one-hot encoding
        # Example: [1, 0, 2] → [[0,1,0], [1,0,0], [0,0,1]]
        targets_one_hot = np.zeros_like(predictions)
        targets_one_hot[np.arange(batch_size), targets] = 1

        # Cross-Entropy formula: -sum(y_true × log(y_pred))
        epsilon = 1e-8  # Prevent log(0)
        ce_loss = -np.mean(np.sum(targets_one_hot * np.log(predictions + epsilon), axis=1))
        
        #  L2 Regularization penalty
        if lambda_l2 > 0:
            l2_penalty = 0.0
            for w in self.weights:
                l2_penalty += np.sum(w ** 2)
            l2_penalty = (lambda_l2 / (2 * batch_size)) * l2_penalty
            total_loss = ce_loss + l2_penalty
        else:
            total_loss = ce_loss
        
        return total_loss

    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray, lambda_l2: float = 0.0) -> None:
        """Backward pass: compute gradients
        
        Args:
            X: Input data (batch_size, input_size)
            y: True labels (batch_size,)
            output: Network predictions (batch_size, output_size)
            lambda_l2: L2 regularization coefficient
        """
        batch_size = X.shape[0]

        # Convert y to one-hot encoding
        y_one_hot = np.zeros_like(output)
        y_one_hot[np.arange(batch_size), y] = 1

        # Clear previous gradients
        self.weight_gradients = []
        self.bias_gradients = []

        # Initial gradient: (prediction - truth) / batch_size
        grad = (output - y_one_hot) / batch_size

        # Go from last layer to first
        for i in reversed(range(len(self.weights))):
            # Activation of previous layer
            activation_prev = X if i == 0 else self.activations[i - 1]
            
            # Weight gradient: dL/dW = A_prev^T × grad
            dW = activation_prev.T @ grad
            
            #  Add L2 regularization gradient
            if lambda_l2 > 0:
                dW += (lambda_l2 / batch_size) * self.weights[i]
            
            # Bias gradient: dL/db = sum(grad)
            dB = np.sum(grad, axis=0, keepdims=True)

            # Store gradients
            self.weight_gradients.insert(0, dW)
            self.bias_gradients.insert(0, dB)

            # Propagate gradient to previous layer
            if i > 0:
                # New grad = current grad × W^T × ReLU derivative
                grad = (grad @ self.weights[i].T) * self.relu_derivative(self.z_values[i - 1])
         #      └─────────────────────┘   └──────────────────────────────────────┘
        #         δ × W^T                         ⊙ f'(Z)

    # def update_weights(self, learning_rate: float) -> None:
        # """Update weights and biases
        
        # Args:
        #     learning_rate: Learning rate (typically 0.01 or 0.001)
        # """
        # for i in range(len(self.weights)):
        #     self.weights[i] -= learning_rate * self.weight_gradients[i]
        #     self.biases[i] -= learning_rate * self.bias_gradients[i]
    def update_weights(self, learning_rate: float, momentum: float = 0.9) -> None:
        """
        Update weights using SGD with Momentum
        
        Args:
            learning_rate: Step size (η)
            momentum: Momentum coefficient (μ), typically 0.9
        """
        for i in range(len(self.weights)):
            # update velocity
            self.weight_velocity[i] = momentum * self.weight_velocity[i] + self.weight_gradients[i]
            self.bias_velocity[i] = momentum * self.bias_velocity[i] + self.bias_gradients[i]
            
            # update weights with velocity
            self.weights[i] -= learning_rate * self.weight_velocity[i]
            self.biases[i] -= learning_rate * self.bias_velocity[i]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels
        
        Args:
            X: Input data (n_samples, input_size)
            
        Returns:
            Predicted labels (n_samples,)
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def train_step(self, X: np.ndarray, y: np.ndarray, learning_rate: float, momentum: float = 0.9, lambda_l2: float = 0.0) -> float:
        """One training step
        
        Args:
            X: Input data (batch_size, input_size)
            y: Labels (batch_size,)
            learning_rate: Learning rate
            lambda_l2: float = 0.0
            
        Returns:
            Loss for this batch (without L2 penalty for monitoring)
        """
        # 1. Forward pass
        output = self.forward(X)
        
        # 2. Compute loss
        loss = self.compute_loss(output, y)
        
        # 3. Compute gradients
        self.backward(X, y, output)
        
        # 4. Update weights
        self.update_weights(learning_rate, momentum)
        
        return loss

    def create_batches(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data into small batches
        
        Args:
            X: Input data
            y: Labels
            batch_size: Size of each batch
            
        Returns:
            List of (X_batch, y_batch) tuples
        """
        indices = np.random.permutation(X.shape[0])
        batches = []
        for start in range(0, X.shape[0], batch_size):
            
            batch_idx = indices[start : start + batch_size]
            batches.append((X[batch_idx], y[batch_idx]))
        return batches


