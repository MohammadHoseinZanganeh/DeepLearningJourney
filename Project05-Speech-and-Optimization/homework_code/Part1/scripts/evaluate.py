"""Load a saved model checkpoint and evaluate it on the test set.

Usage:
    python scripts/evaluate.py mel
    python scripts/evaluate.py hubert
"""

import os
import sys

import torch
from torch.utils.data import DataLoader

from models.mlp import EmotionMLP
from models.cnn import EmotionCNN
from scripts.train import load_config, load_features, run_one_epoch, split_dataset
from utils.metrics import compute_confusion_matrix
from utils.plotting import plot_confusion_matrix
from utils.device import get_device_from_config


def load_model(feature_type, input_size, hidden_size, num_classes, checkpoint_dir, device, n_mels=64):
    """Load the appropriate model based on feature type."""
    if feature_type == "mel":
        model = EmotionCNN(input_size, hidden_size, num_classes, n_mels)
    else:
        model = EmotionMLP(input_size, hidden_size, num_classes)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"model_{feature_type}.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(feature_type):
    """Load the saved checkpoint for feature_type and report test metrics.

    Args:
        feature_type (str): "mel" or "hubert".
    """
    config = load_config()
    dataset_config = config["dataset"]
    training_config = config["training"]
    mel_config = config["mel_spectrogram"]
    
    device = get_device_from_config(config)
    print(f"Device: {device}")

    data = load_features(feature_type, dataset_config["processed_data_dir"])
    features, labels = data["features"], data["labels"]

    _, _, test_set = split_dataset(
        features,
        labels,
        dataset_config["val_split"],
        dataset_config["test_split"],
        dataset_config["random_seed"],
    )
    test_loader = DataLoader(test_set, batch_size=training_config["batch_size"], shuffle=False)

    input_size = features.shape[1]
    num_classes = len(dataset_config["classes"])
    
    model = load_model(
        feature_type,
        input_size,
        training_config["hidden_size"],
        num_classes,
        training_config["checkpoint_dir"],
        device,
        mel_config["n_mels"]
    )

    loss_function = torch.nn.CrossEntropyLoss()
    test_loss, test_accuracy = run_one_epoch(model, test_loader, loss_function, None, device)
    print(f"\n[{feature_type}] Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

    all_predictions = []
    all_labels = []
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

    confusion = compute_confusion_matrix(all_predictions, all_labels, num_classes)
    results_dir = training_config["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    plot_confusion_matrix(
        confusion,
        dataset_config["classes"],
        os.path.join(results_dir, f"confusion_matrix_{feature_type}_eval.png"),
        feature_type
    )


if __name__ == "__main__":
    feature_type = sys.argv[1] if len(sys.argv) > 1 else "mel"
    evaluate_model(feature_type)