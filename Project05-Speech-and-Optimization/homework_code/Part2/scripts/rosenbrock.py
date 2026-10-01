"""Rosenbrock benchmark function and its gradients.

This file implements Part 1 of the assignment: the 2D Rosenbrock
function f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2, and its partial
derivatives with respect to x and y, computed by hand.
"""


def rosenbrock(x, y):
    """Compute the value of the 2D Rosenbrock function at a point.

    Args:
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.

    Returns:
        float: The value of f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2.
    """
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def grad_x(x, y):
    """Compute the partial derivative of the Rosenbrock function w.r.t. x.

    Derived by hand from f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2:
        df/dx = -2 * (1 - x) - 400 * x * (y - x^2)

    Args:
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.

    Returns:
        float: The value of df/dx at (x, y).
    """
    return -2 * (1 - x) - 400 * x * (y - x ** 2)


def grad_y(x, y):
    """Compute the partial derivative of the Rosenbrock function w.r.t. y.

    Derived by hand from f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2:
        df/dy = 200 * (y - x^2)

    Args:
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.

    Returns:
        float: The value of df/dy at (x, y).
    """
    return 200 * (y - x ** 2)
