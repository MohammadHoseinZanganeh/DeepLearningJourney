import numpy as np
from utils import sigmoid, sigmoid_deriv

x = np.array([[1],[0.5]]) #input data
y = 3.97 #actual label
learning_rate = 0.5

#weights and biases initation
W1 = np.array([[1, 0], [-1, 0]], dtype=float)
b1 = np.array([[0], [0.5]])
print(f" initial W1 =\n{W1}\n")
print(f" initial b1 =\n{b1}\n")

W2 = np.array([[0, -2, -1], [1, 1, 1]], dtype=float)
b2 = np.zeros((3, 1))
print(f" initial W2 =\n{W2}\n")
print(f" initial b2 =\n{b2}\n")

W3 = np.array([[1], [-1], [1]], dtype=float)
b3 = np.array([[2.0]])
print(f" initial W3 =\n{W3}\n")
print(f" initial b3 =\n{b3}\n")

#forward propagation
z1 = W1.T @ x + b1    # first hiden layer
a1 = sigmoid(z1)

z2 = W2.T @ a1 + b2   #second hiden layar 
a2 = sigmoid(z2)

z3 = W3.T @ a2 + b3 #output layar
y_hat = z3

#back propagation
#level one: Gradient to loss function
"""
L = 0.5*(y - y_hat)^2  =>  dL/dy_hat = -(y - y_hat)
"""
delta3 = -(y - y_hat)

#level two: Gradient to W3 and b3
"""
z3 = W3.T @ a2 + b3
dL/d(W3.T) = delta3 @ a2.T   shape: (1,1)@(1,3) = (1,3)
dL/dW3 = (dL/d(W3.T)).T      shape: (3,1)
"""
dW3 = (delta3 @ a2.T).T                        # (3,1)
db3 = delta3             

#level three: Gradient to sigmoid then to W2 and b2
"""
dL/da2 = W3.T @ delta3 notice that (W3.T @ a2).T = a2.T @ W3 ...
... (A@B).T = B.T@A.T  =>  dL/da2 = W3 @ delta3
dL/dz2 = dL/da2 * sigmoid'(z2)
z2 = W2.T @ a1 + b2
dL/dW2 = (delta2 @ a1).T
delta2 = dL/da2 * sigmoid'(z2)
"""
da2 = W3 * delta3
delta2 = da2 * sigmoid_deriv(z2)  
dW2 = (delta2 @ a1.T).T
db2 = delta2

#level four: Gradient to sigmoid then to W1 and b1
da1 = W2 @ delta2
delta1 = da1 * sigmoid_deriv(z1)

dW1 = (delta1 @ x.T).T
db1 = delta1

#updating weights
params = {
    'W1': (W1, dW1), 'b1': (b1, db1),
    'W2': (W2, dW2), 'b2': (b2, db2),
    'W3': (W3, dW3), 'b3': (b3, db3),
}

for name, (param, grad) in params.items():
    param -= learning_rate * grad
    print(f"{name} after update =\n{param}\n")


