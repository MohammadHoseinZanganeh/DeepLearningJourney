"""Train the model on either mel-spectrogram or HuBERT features.

Usage:
    python scripts/train.py            # uses feature_type from config.yaml
    python scripts/train.py mel        # force log-mel features
    python scripts/train.py hubert     # force HuBERT features
"""

import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.mlp import EmotionMLP
from models.cnn import EmotionCNN
from utils.metrics import compute_accuracy, compute_confusion_matrix, compute_class_metrics, print_class_metrics
from utils.plotting import plot_confusion_matrix, plot_training_curves
from utils.device import get_device_from_config


def load_config(config_path="config/config.yaml"):
    """Load the YAML configuration file."""
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def load_features(feature_type, processed_data_dir):
    """Load the cached feature file for the given feature type.

    Args:
        feature_type (str): "mel" or "hubert".
        processed_data_dir (str): Folder holding the cached feature files.

    Returns:
        dict: Dictionary with "features" and "labels" tensors.
    """
    file_name = "mel_features.pt" if feature_type == "mel" else "hubert_features.pt"
    file_path = os.path.join(processed_data_dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find '{file_path}'. Run scripts/extract_features.py first."
        )
    return torch.load(file_path)


def split_dataset(features, labels, val_split, test_split, random_seed):
    """Split features/labels into train, validation, and test sets.

    Args:
        features (torch.Tensor): All feature vectors.
        labels (torch.Tensor): All integer labels.
        val_split (float): Fraction of data used for validation.
        test_split (float): Fraction of data used for testing.
        random_seed (int): Seed so the split is reproducible.

    Returns:
        tuple: (train_set, val_set, test_set) as torch Subsets.
    """
    full_dataset = TensorDataset(features, labels)

    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size

    generator = torch.Generator().manual_seed(random_seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f"Dataset split: Train={train_size}, Validation={val_size}, Test={test_size}")
    return train_set, val_set, test_set


def run_one_epoch(model, data_loader, loss_function, optimizer=None, device=None):
    """Run the model over one epoch, training if an optimizer is given.

    Args:
        model (nn.Module): The model.
        data_loader (DataLoader): Batches of (features, labels).
        loss_function: The loss function to use.
        optimizer: If given, the model weights are updated (training mode).
            If None, the model is only evaluated (no weight updates).
        device: The device to use (CPU or GPU).

    Returns:
        tuple[float, float]: Average loss and accuracy for the epoch.
    """
    if device is None:
        device = torch.device("cpu")
    
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch_features, batch_labels in data_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        
        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(batch_features)
            loss = loss_function(outputs, batch_labels)

            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * batch_features.size(0)
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.append(predictions.cpu())
        all_labels.append(batch_labels.cpu())

    average_loss = total_loss / len(data_loader.dataset)
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    accuracy = compute_accuracy(all_predictions, all_labels)

    return average_loss, accuracy


def create_model(feature_type, input_size, hidden_size, num_classes, n_mels=64):
    """Create the appropriate model based on feature type.
    
    Args:
        feature_type (str): "mel" or "hubert".
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden layers.
        num_classes (int): Number of output classes.
        n_mels (int): Number of mel bands (for CNN reshaping).
    
    Returns:
        nn.Module: The appropriate model.
    """
    if feature_type == "mel":
        print("Using CNN for mel-spectrogram features")
        return EmotionCNN(input_size, hidden_size, num_classes, n_mels)
    else:
        print("Using MLP for HuBERT features")
        return EmotionMLP(input_size, hidden_size, num_classes)


def evaluate_on_test_set(model, test_loader, device, class_names, feature_type):
    """Evaluate model on test set and report per-class metrics.
    
    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to use.
        class_names (list[str]): List of class names.
        feature_type (str): "mel" or "hubert".
    
    Returns:
        tuple: (test_loss, test_accuracy, predictions, labels)
    """
    loss_function = torch.nn.CrossEntropyLoss()
    test_loss, test_accuracy = run_one_epoch(model, test_loader, loss_function, None, device)
    
    print(f"\n[{feature_type}] Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
    
    # Collect all predictions and labels for detailed analysis
    all_predictions = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.append(predictions.cpu())
            all_labels.append(batch_labels.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    # Compute and print per-class metrics
    metrics = compute_class_metrics(all_predictions, all_labels, class_names)
    print_class_metrics(metrics, class_names)
    
    return test_loss, test_accuracy, all_predictions, all_labels, metrics


def train_model(config, feature_type):
    """Train the model on the chosen feature type and save the model and plots.

    Args:
        config (dict): The loaded config.yaml settings.
        feature_type (str): "mel" or "hubert".

    Returns:
        dict: Final test loss and test accuracy.
    """
    device = get_device_from_config(config)
    print(f"Device: {device}")
    
    dataset_config = config["dataset"]
    training_config = config["training"]
    mel_config = config["mel_spectrogram"]

    data = load_features(feature_type, dataset_config["processed_data_dir"])
    features, labels = data["features"], data["labels"]
    print(f"Loaded {len(features)} samples with feature dimension {features.shape[1]}")

    train_set, val_set, test_set = split_dataset(
        features,
        labels,
        dataset_config["val_split"],
        dataset_config["test_split"],
        dataset_config["random_seed"],
    )

    batch_size = training_config["batch_size"]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    input_size = features.shape[1]
    num_classes = len(dataset_config["classes"])
    class_names = dataset_config["classes"]
    
    model = create_model(
        feature_type, 
        input_size, 
        training_config["hidden_size"], 
        num_classes,
        mel_config["n_mels"]
    )
    model = model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["learning_rate"])

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Initialize best model tracking
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0

    num_epochs = training_config["num_epochs"]
    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = run_one_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss, val_accuracy = run_one_epoch(model, val_loader, loss_function, None, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(
            f"[{feature_type}] Epoch {epoch}/{num_epochs} - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_accuracy:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_accuracy:.4f}"
        )
        
        # Save best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            print(f"  -> New best model at epoch {best_epoch} with validation accuracy: {best_val_accuracy:.4f}")

    # Save final model
    checkpoint_dir = training_config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save best model
    best_checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{feature_type}.pt")
    torch.save(best_model_state, best_checkpoint_path)
    print(f"\nSaved best model (epoch {best_epoch}, val_acc: {best_val_accuracy:.4f}) to '{best_checkpoint_path}'")
    
    # Also save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, f"model_{feature_type}.pt")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Saved final model to '{final_checkpoint_path}'")

    # Load best model for evaluation
    model.load_state_dict(best_model_state)
    print(f"\nLoaded best model from epoch {best_epoch} for evaluation")

    # Plot training curves
    results_dir = training_config["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    
    plot_training_curves(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        os.path.join(results_dir, f"training_curves_{feature_type}.png"),
        feature_type
    )

    # Evaluate best model on test set
    print("\n" + "=" * 60)
    print(f"EVALUATING BEST MODEL ON TEST SET ({feature_type})")
    print("=" * 60)
    
    test_loss, test_accuracy, all_predictions, all_labels, metrics = evaluate_on_test_set(
        model, test_loader, device, class_names, feature_type
    )

    # Plot confusion matrix
    confusion = compute_confusion_matrix(all_predictions, all_labels, num_classes)
    plot_confusion_matrix(
        confusion,
        class_names,
        os.path.join(results_dir, f"confusion_matrix_{feature_type}.png"),
        feature_type
    )

    # Return results including per-class metrics
    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "per_class_metrics": metrics["per_class"],
        "confusion_matrix": confusion,
    }


if __name__ == "__main__":
    config = load_config()
    feature_type = sys.argv[1] if len(sys.argv) > 1 else config["training"]["feature_type"]
    train_model(config, feature_type)