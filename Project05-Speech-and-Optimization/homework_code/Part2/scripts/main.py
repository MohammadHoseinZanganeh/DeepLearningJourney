"""Main script for Homework 5 - Part 2 (Optimization).

Running this script (``python scripts/main.py`` from the homework_code
folder) performs every part of the assignment in order and saves all
plots and a summary of results into the results/ folder:

  Part 1: plot the Rosenbrock surface on [-1, 1] x [-1, 1].
  Part 2 / item 3: baseline run of SGD, Adam, and RMSProp.
  Part 2 / item 4: repeat the baseline run at learning rates 0.01 and 1,
                   and compare all three rates (0.1, 0.01, 1) on one plot.
  Part 2 / item 5: run every optimizer from 3 different starting points.
  Part 2 / item 6: repeat the baseline run with noisy gradients.
"""

import os
import sys
import json

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))

from optimizers import SGDOptimizer, AdamOptimizer, RMSPropOptimizer  # noqa: E402
from train import run_optimization  # noqa: E402
from evaluate import compute_final_metrics  # noqa: E402
from plotting import (  # noqa: E402
    plot_rosenbrock_surface,
    plot_value_vs_epoch,
    plot_distance_vs_epoch,
    plot_trajectory_3d,
    plot_trajectory_3d_log,
)


def load_config(config_path):
    """Load hyperparameters from a YAML config file.

    Args:
        config_path (str): Path to the config.yaml file.

    Returns:
        dict: The parsed hyperparameters.
    """
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def make_optimizer_specs(learning_rate, config):
    """Build fresh optimizer specs (class + kwargs) at a given learning rate.

    A fresh set of specs is needed for every run because Adam and
    RMSProp keep internal state that should not be shared between
    separate experiments.

    Args:
        learning_rate (float): Learning rate to use for all three optimizers.
        config (dict): Hyperparameters loaded from config.yaml.

    Returns:
        dict: Mapping from optimizer name to a (class, kwargs) tuple.
    """
    specs = {
        "SGD": (SGDOptimizer, {"lr": learning_rate}),
        "Adam": (
            AdamOptimizer,
            {
                "lr": learning_rate,
                "beta1": config["adam"]["beta1"],
                "beta2": config["adam"]["beta2"],
                "epsilon": config["adam"]["epsilon"],
            },
        ),
        "RMSProp": (
            RMSPropOptimizer,
            {
                "lr": learning_rate,
                "beta": config["rmsprop"]["beta"],
                "epsilon": config["rmsprop"]["epsilon"],
            },
        ),
    }
    return specs


def run_all_optimizers(learning_rate, start_x, start_y, config, noise_std=0.0):
    """Run SGD, Adam, and RMSProp with the same settings and collect their histories.

    Args:
        learning_rate (float): Learning rate to use for every optimizer.
        start_x (float): Starting x-coordinate.
        start_y (float): Starting y-coordinate.
        config (dict): Hyperparameters loaded from config.yaml.
        noise_std (float): Standard deviation of gradient noise (0.0 for none).

    Returns:
        dict: Mapping from optimizer name to its history dict.
    """
    max_epochs = config["num_epochs"]
    grad_threshold = config["gradient_threshold"]
    specs = make_optimizer_specs(learning_rate, config)
    histories = {}
    for name, (opt_class, opt_kwargs) in specs.items():
        histories[name] = run_optimization(
            opt_class, opt_kwargs, start_x, start_y,
            max_epochs=max_epochs, grad_threshold=grad_threshold,
            noise_std=noise_std,
        )
    return histories


def main():
    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    config = load_config(config_path)
    max_epochs = config["num_epochs"]
    grad_threshold = config["gradient_threshold"]
    start_x = config["start_point"]["x"]
    start_y = config["start_point"]["y"]
    default_lr = config["default_learning_rate"]

    all_results = {}

    # ---------------- Part 1, item 3: plot the Rosenbrock surface ----------------
    print("Part 1: plotting the Rosenbrock surface on [-1, 1] x [-1, 1]")
    plot_rosenbrock_surface(os.path.join(results_dir, "part1_rosenbrock_surface.png"))

    # ---------------- Part 2, item 3: baseline run ----------------
    print(f"Part 2 (item 3): baseline run at learning rate {default_lr}")
    histories = run_all_optimizers(default_lr, start_x, start_y, config)
    labels = list(histories.keys())
    values = list(histories.values())

    plot_value_vs_epoch(
        values, labels, f"Function value vs epoch (lr={default_lr})",
        os.path.join(results_dir, "part3_value_vs_epoch.png"),
    )
    plot_distance_vs_epoch(
        values, labels, f"Distance to optimum vs epoch (lr={default_lr})",
        os.path.join(results_dir, "part3_distance_vs_epoch.png"),
    )
    plot_trajectory_3d(
        values, labels, f"Optimizer paths on the Rosenbrock surface (lr={default_lr})",
        os.path.join(results_dir, "part3_trajectory.png"),
    )

    plot_trajectory_3d_log(
    values, labels, 
    f"Optimizer paths on the Rosenbrock surface (lr={default_lr})",
    os.path.join(results_dir, "part3_trajectory_log.png"),
    )
    all_results["part3"] = {name: compute_final_metrics(h) for name, h in histories.items()}
 
    # ---------------- Part 2, item 4: different learning rates ----------------
    print("Part 2 (item 4): comparing different learning rates")
    all_results["part4"] = {}
    # All learning rates to compare: baseline (0.1) + extra (0.01, 1.0)
    all_lrs = [default_lr] + config["learning_rates"]

    # Run each optimizer at each learning rate and collect histories.
    # part4_histories[lr][optimizer_name] = history
    part4_histories = {}
    for lr in all_lrs:
        part4_histories[lr] = run_all_optimizers(lr, start_x, start_y, config)
        all_results["part4"][str(lr)] = {
            name: compute_final_metrics(h) for name, h in part4_histories[lr].items()
        }

    # Individual plots: one per learning rate, showing all three optimizers.
    for lr in all_lrs:
        hists = part4_histories[lr]
        labels = list(hists.keys())
        vals = list(hists.values())

        plot_value_vs_epoch(
            vals, labels, f"Function value vs epoch (lr={lr})",
            os.path.join(results_dir, f"part4_value_vs_epoch_lr{lr}.png"),
        )
        plot_distance_vs_epoch(
            vals, labels, f"Distance to optimum vs epoch (lr={lr})",
            os.path.join(results_dir, f"part4_distance_vs_epoch_lr{lr}.png"),
        )

    # Combined comparison plot: for each optimizer, show all three learning
    # rates on one graph.
    specs = make_optimizer_specs(default_lr, config)
    for opt_name in specs.keys():
        combined_values = []
        combined_labels = []
        for lr in all_lrs:
            combined_values.append(part4_histories[lr][opt_name])
            combined_labels.append(f"lr={lr}")
        plot_value_vs_epoch(
            combined_values, combined_labels,
            f"Function value vs epoch — {opt_name} (all learning rates)",
            os.path.join(results_dir, f"part4_combined_value_vs_epoch_{opt_name}.png"),
        )
        plot_distance_vs_epoch(
            combined_values, combined_labels,
            f"Distance to optimum vs epoch — {opt_name} (all learning rates)",
            os.path.join(results_dir, f"part4_combined_distance_vs_epoch_{opt_name}.png"),
        )

    # ---------------- Part 2, item 5: different starting points ----------------
    print("Part 2 (item 5): comparing different starting points")
    all_results["part5"] = {}
    for i, point in enumerate(config["extra_start_points"]):
        histories = run_all_optimizers(default_lr, point["x"], point["y"], config)
        labels = list(histories.keys())
        values = list(histories.values())

        plot_value_vs_epoch(
            values, labels, f"Function value vs epoch (start=({point['x']}, {point['y']}))",
            os.path.join(results_dir, f"part5_value_vs_epoch_start{i}.png"),
        )
        key = f"start_{i}_({point['x']},{point['y']})"
        all_results["part5"][key] = {name: compute_final_metrics(h) for name, h in histories.items()}

    # ---------------- Part 2, item 6: noisy gradients ----------------
    print("Part 2 (item 6): baseline run with noisy gradients")
    noise_std = config["noise_std"]
    histories = run_all_optimizers(
        default_lr, start_x, start_y, config, noise_std=noise_std
    )
    labels = list(histories.keys())
    values = list(histories.values())

    plot_value_vs_epoch(
        values, labels, f"Function value vs epoch (lr={default_lr}, noisy gradients)",
        os.path.join(results_dir, "part6_value_vs_epoch_noisy.png"),
    )
    plot_distance_vs_epoch(
        values, labels, f"Distance to optimum vs epoch (lr={default_lr}, noisy gradients)",
        os.path.join(results_dir, "part6_distance_vs_epoch_noisy.png"),
    )
    all_results["part6"] = {name: compute_final_metrics(h) for name, h in histories.items()}

    # ---------------- Save the summary metrics ----------------
    results_path = os.path.join(results_dir, "results_summary.json")
    with open(results_path, "w") as results_file:
        json.dump(all_results, results_file, indent=2, default=str)

    print(f"\nAll plots and results were saved to: {results_dir}")


if __name__ == "__main__":
    main()
