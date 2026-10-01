"""Main script that runs the full Speech Emotion Recognition pipeline.

Steps:
    1. Download the CREMA-D dataset (skipped if it is already downloaded).
    2. Extract log-mel and HuBERT features (skipped if already extracted).
    3. Train CNN classifier on the log-mel features.
    4. Train MLP classifier on the HuBERT features.
    5. Print comparison between the two models.

Run this from the project root (the Part1 folder):
    python scripts/main.py
"""

import os
import sys

# Get the absolute path of the project root (Part1 folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root to Python path so imports work
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml

# Import modules
from data.data_loader import download_crema_d, get_dataset_info
from scripts.extract_features import extract_all_features
from scripts.train import train_model


def load_config():
    """Load the YAML configuration file from the config directory."""
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def update_config_paths(config):
    """Update config paths to be absolute paths relative to PROJECT_ROOT."""
    config["dataset"]["raw_data_dir"] = os.path.join(PROJECT_ROOT, config["dataset"]["raw_data_dir"])
    config["dataset"]["processed_data_dir"] = os.path.join(PROJECT_ROOT, config["dataset"]["processed_data_dir"])
    config["training"]["checkpoint_dir"] = os.path.join(PROJECT_ROOT, config["training"]["checkpoint_dir"])
    config["training"]["results_dir"] = os.path.join(PROJECT_ROOT, config["training"]["results_dir"])
    return config


def print_detailed_comparison(mel_results, hubert_results, class_names):
    """Print detailed comparison between the two models."""
    print("\n" + "=" * 70)
    print("DETAILED COMPARISON: Mel-Spectrogram vs HuBERT")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'Mel-Spectrogram + CNN':<25} {'HuBERT + MLP':<25}")
    print("-" * 75)
    print(f"{'Test Accuracy':<25} {mel_results['test_accuracy']:.4f}{' ' * 18} {hubert_results['test_accuracy']:.4f}")
    print(f"{'Test Loss':<25} {mel_results['test_loss']:.4f}{' ' * 18} {hubert_results['test_loss']:.4f}")
    print(f"{'Best Validation Accuracy':<25} {mel_results['best_val_accuracy']:.4f}{' ' * 18} {hubert_results['best_val_accuracy']:.4f}")
    print(f"{'Best Epoch':<25} {mel_results['best_epoch']}{' ' * 18} {hubert_results['best_epoch']}")
    
    print("\n" + "-" * 75)
    print("PER-CLASS ACCURACY COMPARISON")
    print("-" * 75)
    print(f"{'Class':<15} {'Mel-Spectrogram':<25} {'HuBERT':<25}")
    print("-" * 75)
    
    for class_name in class_names:
        mel_acc = mel_results["per_class_metrics"][class_name]["recall"]
        hubert_acc = hubert_results["per_class_metrics"][class_name]["recall"]
        diff = hubert_acc - mel_acc
        print(f"{class_name:<15} {mel_acc:.4f}{' ' * 18} {hubert_acc:.4f} ({'+' if diff > 0 else ''}{diff:.4f})")
    
    print("-" * 75)
    
    improvement = hubert_results['test_accuracy'] - mel_results['test_accuracy']
    print(f"\nHuBERT improvement over Mel-Spectrogram: {improvement:.4f} ({improvement*100:.2f}%)")
    
    print("\n" + "=" * 70)


def main():
    """Run the full pipeline from downloading data to comparing both models."""
    print(f"Project root: {PROJECT_ROOT}")
    
    config = load_config()
    config = update_config_paths(config)
    
    dataset_config = config["dataset"]
    class_names = dataset_config["classes"]
    processed_data_dir = dataset_config["processed_data_dir"]

    print("=" * 60)
    print("STEP 1: Downloading dataset")
    print("=" * 60)
    download_crema_d(dataset_config["raw_data_dir"], dataset_config["kaggle_dataset_id"])
    
    # Show dataset info with speaker_range passed correctly
    dataset_info = get_dataset_info(
        dataset_config["raw_data_dir"], 
        dataset_config["classes"],
        dataset_config.get("speaker_range")
    )
    print(f"\nDataset information:")
    print(f"  Total samples: {dataset_info['total_samples']}")
    print(f"  Class distribution:")
    for cls, count in dataset_info["class_counts"].items():
        print(f"    {cls}: {count}")

    print("\n" + "=" * 60)
    print("STEP 2: Extracting features")
    print("=" * 60)
    
    mel_features_path = os.path.join(processed_data_dir, "mel_features.pt")
    hubert_features_path = os.path.join(processed_data_dir, "hubert_features.pt")
    
    # Check if speaker_range has changed to avoid using old cached features
    cache_info_path = os.path.join(processed_data_dir, "cache_info.txt")
    current_speaker_range = dataset_config.get("speaker_range")
    
    if os.path.exists(cache_info_path):
        with open(cache_info_path, "r") as f:
            cached_range_str = f.read().strip()
        # Convert stored string to tuple for comparison
        try:
            cached_range = eval(cached_range_str)
        except:
            cached_range = None
            
        if cached_range != current_speaker_range:
            print(f"Speaker range changed from {cached_range} to {current_speaker_range}. Re-extracting features...")
            if os.path.exists(mel_features_path):
                os.remove(mel_features_path)
                print(f"  Removed old: {mel_features_path}")
            if os.path.exists(hubert_features_path):
                os.remove(hubert_features_path)
                print(f"  Removed old: {hubert_features_path}")
    
    if os.path.exists(mel_features_path) and os.path.exists(hubert_features_path):
        print("Features already extracted, skipping extraction step.")
    else:
        # Save current speaker_range for next time
        os.makedirs(processed_data_dir, exist_ok=True)
        with open(cache_info_path, "w") as f:
            f.write(str(current_speaker_range))
        print(f"Current speaker_range ({current_speaker_range}) will be cached for future runs.")
        extract_all_features(config)

    print("\n" + "=" * 60)
    print("STEP 3: Training CNN on Mel-Spectrogram features")
    print("=" * 60)
    mel_results = train_model(config, "mel")

    print("\n" + "=" * 60)
    print("STEP 4: Training MLP on HuBERT features")
    print("=" * 60)
    hubert_results = train_model(config, "hubert")

    print("\n" + "=" * 60)
    print("STEP 5: Final Comparison")
    print("=" * 60)
    print_detailed_comparison(mel_results, hubert_results, class_names)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Results saved in: {config['training']['results_dir']}")
    print(f"Checkpoints saved in: {config['training']['checkpoint_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()