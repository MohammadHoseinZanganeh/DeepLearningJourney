"""Extract log-mel and HuBERT features for every CREMA-D audio file.

Run this once before training. Feature extraction (especially HuBERT) is
slow, so we compute the features a single time and save them to disk.
The training script then just loads these cached feature files.
"""

import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from data.data_loader import CremaDDataset
from utils.features import (
    extract_hubert_feature,
    extract_mel_spectrogram_feature,
    load_audio,
    get_device_from_config,
)


def load_config(config_path="config/config.yaml"):
    """Load the YAML configuration file."""
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def extract_all_features(config):
    """Extract mel and HuBERT features for the whole dataset and save them.

    Args:
        config (dict): The loaded config.yaml settings.
    """
    dataset_config = config["dataset"]
    audio_config = config["audio"]
    mel_config = config["mel_spectrogram"]
    hubert_config = config["hubert"]
    max_duration = dataset_config.get("max_duration", 3.0)
    speaker_range = dataset_config.get("speaker_range", (1001, 1021))

    device = get_device_from_config()
    print(f"Extracting features using device: {device}")
    print(f"Using max duration: {max_duration} seconds")
    print(f"Using speakers: {speaker_range[0]} to {speaker_range[1]}")

    dataset = CremaDDataset(
        raw_data_dir=dataset_config["raw_data_dir"],
        classes=dataset_config["classes"],
        speaker_range=speaker_range,
    )
    
    print(f"Total samples found: {len(dataset)}")

    mel_feature_list = []
    hubert_feature_list = []
    label_list = []

    for file_path, label in tqdm(dataset, desc="Extracting features"):
        waveform = load_audio(file_path, audio_config["sampling_rate"], max_duration)

        mel_vector = extract_mel_spectrogram_feature(
            waveform,
            audio_config["sampling_rate"],
            mel_config["n_mels"],
            mel_config["n_fft"],
            mel_config["hop_length"],
        )
        hubert_vector = extract_hubert_feature(
            waveform, audio_config["sampling_rate"], hubert_config["model_name"]
        )

        mel_feature_list.append(mel_vector)
        hubert_feature_list.append(hubert_vector)
        label_list.append(label)

    processed_dir = dataset_config["processed_data_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    torch.save(
        {
            "features": torch.tensor(np.array(mel_feature_list), dtype=torch.float32),
            "labels": torch.tensor(label_list, dtype=torch.long),
        },
        os.path.join(processed_dir, "mel_features.pt"),
    )
    torch.save(
        {
            "features": torch.tensor(np.array(hubert_feature_list), dtype=torch.float32),
            "labels": torch.tensor(label_list, dtype=torch.long),
        },
        os.path.join(processed_dir, "hubert_features.pt"),
    )
    print(f"Saved extracted features to '{processed_dir}'.")


if __name__ == "__main__":
    config = load_config()
    extract_all_features(config)