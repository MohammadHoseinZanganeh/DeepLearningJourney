"""Simple evaluation helpers for summarizing optimizer runs."""


def compute_final_metrics(history, tolerance=0.01):
    """Compute simple summary metrics from an optimization history.

    Args:
        history (dict): Dictionary with lists "loss", "distance", and
            "gradient_norm" recorded at every epoch.
        tolerance (float): Distance below which we consider the run to
            have "converged" to the optimum. Defaults to 0.01.

    Returns:
        dict: Final loss, final distance, final gradient norm, the
        number of epochs actually run, and the first epoch at which
        the distance to the optimum dropped below the tolerance
        (None if it never did within the run).
    """
    final_loss = history["loss"][-1]
    final_distance = history["distance"][-1]
    final_gradient_norm = history["gradient_norm"][-1]
    epochs_run = len(history["loss"])

    convergence_epoch = None
    for epoch, distance in enumerate(history["distance"]):
        if distance < tolerance:
            convergence_epoch = epoch
            break

    return {
        "final_loss": final_loss,
        "final_distance": final_distance,
        "final_gradient_norm": final_gradient_norm,
        "epochs_run": epochs_run,
        "convergence_epoch": convergence_epoch,
    }
