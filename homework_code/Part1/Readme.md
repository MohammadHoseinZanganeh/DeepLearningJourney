# Part 1: Simple 3-Layer Neural Network

A minimal neural network implementation with 2 hidden layers using sigmoid activation, trained with backpropagation from scratch.

## Files

- `main.py` - Training script with manual backpropagation
- `utils.py` - Sigmoid activation and derivative functions
- `requirements.txt` - Dependencies
- `test.ipynb` - Testing notebook
- `images/` - Output visualizations
  - `5RandomSamplePerClass.png`
  - `HistogramDistributionClasses.png`

## Architecture
Input (2) → Hidden Layer 1 (2 neurons) → Hidden Layer 2 (3 neurons) → Output (1)


**Activation:** Sigmoid for hidden layers, Linear for output

## Initial Parameters

| Layer | Weights | Biases |
|-------|---------|--------|
| W1 (2×2) | `[[1,0], [-1,0]]` | `[[0], [0.5]]` |
| W2 (2×3) | `[[0,-2,-1], [1,1,1]]` | `[[0],[0],[0]]` |
| W3 (3×1) | `[[1],[-1],[1]]` | `[[2]]` |

## Forward Pass

```python
z1 = W1^T @ x + b1
a1 = sigmoid(z1)

z2 = W2^T @ a1 + b2
a2 = sigmoid(z2)

z3 = W3^T @ a2 + b3
y_hat = z3
# Output layer gradient
delta3 = -(y - y_hat)

# Layer 3 gradients
dW3 = (delta3 @ a2.T).T
db3 = delta3

# Layer 2 gradients
da2 = W3 * delta3
delta2 = da2 * sigmoid_deriv(z2)
dW2 = (delta2 @ a1.T).T
db2 = delta2

# Layer 1 gradients
da1 = W2 @ delta2
delta1 = da1 * sigmoid_deriv(z1)
dW1 = (delta1 @ x.T).T
db1 = delta1

cd Part1
python main.py
```