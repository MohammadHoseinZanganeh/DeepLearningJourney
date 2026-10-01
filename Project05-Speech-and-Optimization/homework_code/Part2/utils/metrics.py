"""Custom evaluation metrics used to analyze optimizer performance."""

import math


def euclidean_distance(x, y, target_x=1.0, target_y=1.0):
    """Compute the Euclidean distance from (x, y) to a target point.

    The Rosenbrock function used in this assignment has its global
    minimum at (1, 1), so the default target is that point.

    Args:
        x (float): x-coordinate of the current point.
        y (float): y-coordinate of the current point.
        target_x (float): x-coordinate of the target point. Defaults to 1.0.
        target_y (float): y-coordinate of the target point. Defaults to 1.0.

    Returns:
        float: The Euclidean distance between (x, y) and (target_x, target_y).
    """
    return math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
