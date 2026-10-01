"""Training loop for running one optimizer on the Rosenbrock function.

The loop runs **until convergence** — defined as the L2 norm of the
gradient dropping below a user-specified threshold (default 0.001) —
rather than for a fixed number of epochs.  A ``max_epochs`` safety cap
prevents infinite loops in case the optimizer diverges.
"""

import math
import os
import sys
import random

import torch

# Make sure we can import rosenbrock.py (same folder) and metrics.py
# (in the utils/ folder one level up) no matter where this script is run from.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))

from rosenbrock import rosenbrock, grad_x, grad_y  # noqa: E402
from metrics import euclidean_distance  # noqa: E402


def run_optimization(
    optimizer_class,
    optimizer_kwargs,
    start_x,
    start_y,
    max_epochs=10000,
    grad_threshold=0.001,
    noise_std=0.0,
):
    """Run an optimizer on the Rosenbrock function until convergence.

    The loop continues while the gradient norm is greater than
    ``grad_threshold``, up to ``max_epochs`` iterations.  At every
    step the gradient is computed by hand using :func:`grad_x` and
    :func:`grad_y` (Part 1), optionally corrupted with Gaussian noise
    (Part 2, item 6), and passed to the optimizer via ``params.grad``
    before calling ``optimizer.step()`` (Part 2, item 1).

    Args:
        optimizer_class: A subclass of :class:`torch.optim.Optimizer`
            (e.g. :class:`SGDOptimizer`, :class:`AdamOptimizer`).
        optimizer_kwargs (dict): Keyword arguments to pass to the
            optimizer constructor (e.g. ``lr``, ``beta1``).
        start_x (float): Starting x-coordinate.
        start_y (float): Starting y-coordinate.
        max_epochs (int): Maximum number of optimization steps (safety
            cap to prevent infinite loops).
        grad_threshold (float): The loop stops when the L2 norm of the
            gradient drops below this value.
        noise_std (float): Standard deviation of the Gaussian noise added
            to the gradients at every step. Use 0.0 for no noise.

    Returns:
        dict: History with the lists "x", "y", "loss", "distance", and
        "gradient_norm", containing one value per epoch.
    """
    # Leaf tensor that requires grad so the optimizer can read/write .grad.
    params = torch.tensor([start_x, start_y], dtype=torch.float32, requires_grad=True)
    optimizer = optimizer_class([params], **optimizer_kwargs)

    history = {
        "x": [],
        "y": [],
        "loss": [],
        "distance": [],
        "gradient_norm": [],
    }

    epoch = 0
    gradient_norm = float("inf")

    while gradient_norm > grad_threshold and epoch < max_epochs:
        current_x = params[0].item()
        current_y = params[1].item()

        loss_value = rosenbrock(current_x, current_y)
        distance_value = euclidean_distance(current_x, current_y)

        gradient_x = grad_x(current_x, current_y)
        gradient_y = grad_y(current_x, current_y)

        if noise_std > 0.0:
            gradient_x += random.gauss(0.0, noise_std)
            gradient_y += random.gauss(0.0, noise_std)

        grads = torch.tensor([gradient_x, gradient_y], dtype=torch.float32)
        gradient_norm = torch.norm(grads).item()

        # Record the current state *before* the update.
        history["x"].append(current_x)
        history["y"].append(current_y)
        history["loss"].append(loss_value)
        history["distance"].append(distance_value)
        history["gradient_norm"].append(gradient_norm)

        # If the gradient is already small enough, we have converged.
        if gradient_norm <= grad_threshold:
            break

        # Guard against divergence (NaN / Inf gradients).
        if math.isnan(gradient_norm) or math.isinf(gradient_norm):
            break

        # Manually set the gradient and let the optimizer update params in-place.
        params.grad = grads
        optimizer.step()
        optimizer.zero_grad()
        epoch += 1

    return history
