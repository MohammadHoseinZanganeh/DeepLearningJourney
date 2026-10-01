"""Plotting functions used throughout the assignment.

All functions here save a figure to disk instead of showing it on
screen, so that main.py can be run from the command line and still
produce all the plots needed for the report.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # do not require a display to save figures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


def plot_rosenbrock_surface(save_path, x_range=(-1, 1), y_range=(-1, 1)):
    """Plot the 3D surface of the Rosenbrock function (Part 1, item 3).

    Args:
        save_path (str): File path to save the resulting figure.
        x_range (tuple): Minimum and maximum x values to plot.
        y_range (tuple): Minimum and maximum y values to plot.
    """
    x_values = np.linspace(x_range[0], x_range[1], 100)
    y_values = np.linspace(y_range[0], y_range[1], 100)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    z_grid = (1 - x_grid) ** 2 + 100 * (y_grid - x_grid ** 2) ** 2

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x_grid, y_grid, z_grid, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_title("Rosenbrock function on [-1, 1] x [-1, 1]")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


def plot_value_vs_epoch(histories, labels, title, save_path):
    """Plot the Rosenbrock function value against epoch number.

    Args:
        histories (list): List of history dicts, each with a "loss" list.
        labels (list): Name for each history, used in the legend.
        title (str): Plot title.
        save_path (str): File path to save the resulting figure.
    """
    plt.figure(figsize=(7, 4.5))
    for history, label in zip(histories, labels):
        plt.plot(history["loss"], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Function value f(x, y)")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


def plot_distance_vs_epoch(histories, labels, title, save_path):
    """Plot the Euclidean distance to the optimum against epoch number.

    Args:
        histories (list): List of history dicts, each with a "distance" list.
        labels (list): Name for each history, used in the legend.
        title (str): Plot title.
        save_path (str): File path to save the resulting figure.
    """
    plt.figure(figsize=(7, 4.5))
    for history, label in zip(histories, labels):
        plt.plot(history["distance"], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Distance to optimum (1, 1)")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


def plot_trajectory_3d(histories, labels, title, save_path):
    """Plot optimizer paths on top of the Rosenbrock surface, in 3D.

    Args:
        histories (list): List of history dicts, each with "x", "y", and
            "loss" lists.
        labels (list): Name for each history, used in the legend.
        title (str): Plot title.
        save_path (str): File path to save the resulting figure.
    """
    x_values = np.linspace(-2, 2, 150)
    y_values = np.linspace(-1, 3, 150)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    z_grid = (1 - x_grid) ** 2 + 100 * (y_grid - x_grid ** 2) ** 2

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x_grid, y_grid, z_grid, cmap="viridis", alpha=0.55, linewidth=0)

    for history, label in zip(histories, labels):
        xs = np.array(history["x"])
        ys = np.array(history["y"])
        zs = np.array(history["loss"])
        # Filter out diverging points so one exploding optimizer doesn't
        # squash the rest of the plot.  Only keep points within the
        # plotted surface range.
        mask = (xs >= -2) & (xs <= 2) & (ys >= -1) & (ys <= 3) & np.isfinite(zs)
        if mask.sum() > 0:
            ax.plot(xs[mask], ys[mask], zs[mask], marker="o", markersize=1.5, label=label)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_zlim(bottom=0, top=250)
    ax.set_title(title)
    ax.legend()
    ax.view_init(elev=25, azim=60)  # camera angle: elevation and azimuth in degrees

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()

def plot_trajectory_3d_log(histories, labels, title, save_path):
    """Plot optimizer paths on top of the Rosenbrock surface, in 3D with log scale for Z axis.
    
    Args:
        histories (list): List of history dicts, each with "x", "y", and "loss" lists.
        labels (list): Name for each history, used in the legend.
        title (str): Plot title.
        save_path (str): File path to save the resulting figure.
    """
    # Define the Rosenbrock surface with larger range to capture diverging paths
    x_values = np.linspace(-3, 3, 150)
    y_values = np.linspace(-2, 4, 150)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    z_grid = (1 - x_grid) ** 2 + 100 * (y_grid - x_grid ** 2) ** 2
    
    # Apply log transform to the surface for better visualization
    z_grid_log = np.log10(z_grid + 1e-10)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot the surface with log scale
    surf = ax.plot_surface(x_grid, y_grid, z_grid_log, cmap="viridis", alpha=0.5, linewidth=0)

    for history, label in zip(histories, labels):
        xs = np.array(history["x"])
        ys = np.array(history["y"])
        zs = np.array(history["loss"])
        
        # Apply log transform to trajectory points
        zs_log = np.log10(zs + 1e-10)
        
        # Filter points that are within reasonable range
        mask = (xs >= -3) & (xs <= 3) & (ys >= -2) & (ys <= 4) & np.isfinite(zs_log)
        
        if mask.sum() > 0:
            # Plot the trajectory line
            ax.plot(xs[mask], ys[mask], zs_log[mask], 
                   linewidth=2, label=label, marker="o", markersize=2)
            
            # Mark the starting point with a larger marker
            if mask[0]:
                ax.scatter(xs[0], ys[0], zs_log[0], 
                          color='red', s=100, marker='*', 
                          label=f'{label} start')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("log10(f(x, y))")
    ax.set_title(f"{title} (log scale)")
    ax.legend()
    ax.view_init(elev=25, azim=60)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


