"""Data loader for CREMA-D dataset.

This module handles downloading the dataset from Kaggle, loading audio files,
and extracting emotion labels. All data-related functionality is centralized here.
"""

import os
import glob
import shutil
import re
import numpy as np
import torch
from torch.utils.data import Dataset

# Maps the 3-letter CREMA-D emotion code to a readable name.
EMOTION_CODE_TO_NAME = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fearful",
    "DIS": "disgust",
}


def get_emotion_from_filename(file_path):
    """Read the emotion name encoded in a CREMA-D file name.

    Args:
        file_path (str): Path to a CREMA-D wav file.

    Returns:
        str or None: The emotion name, or None if it could not be parsed.
    """
    file_name = os.path.basename(file_path)
    name_parts = file_name.split("_")
    if len(name_parts) < 3:
        return None
    emotion_code = name_parts[2]
    return EMOTION_CODE_TO_NAME.get(emotion_code)


def get_speaker_id_from_filename(file_path):
    """Read the speaker ID encoded in a CREMA-D file name.

    Args:
        file_path (str): Path to a CREMA-D wav file.

    Returns:
        int or None: The speaker ID, or None if it could not be parsed.
    """
    file_name = os.path.basename(file_path)
    name_parts = file_name.split("_")
    if len(name_parts) < 1:
        return None
    try:
        return int(name_parts[0])
    except ValueError:
        return None


def download_crema_d(raw_data_dir, kaggle_dataset_id):
    """Download and unpack the CREMA-D dataset if it does not already exist.

    Args:
        raw_data_dir (str): Folder where the raw wav files should live.
        kaggle_dataset_id (str): Kaggle dataset identifier, e.g. "ejlok1/cremad".

    Returns:
        None
    """
    # If the folder already has files in it, we assume the dataset was
    # downloaded before and we skip downloading it again.
    if os.path.isdir(raw_data_dir) and len(os.listdir(raw_data_dir)) > 0:
        print(f"Dataset already found at '{raw_data_dir}', skipping download.")
        return

    print(f"Dataset not found at '{raw_data_dir}'. Downloading from Kaggle...")
    os.makedirs(raw_data_dir, exist_ok=True)

    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub is required to download the dataset. "
            "Install it with: pip install kagglehub"
        )

    downloaded_path = kagglehub.dataset_download(kaggle_dataset_id)
    print(f"Kaggle saved the dataset to: {downloaded_path}")

    # Copy everything into our expected raw_data_dir
    for item_name in os.listdir(downloaded_path):
        source_path = os.path.join(downloaded_path, item_name)
        destination_path = os.path.join(raw_data_dir, item_name)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination_path)

    print(f"Dataset is ready at '{raw_data_dir}'.")


class CremaDDataset(Dataset):
    """A PyTorch Dataset that lists CREMA-D wav files for the wanted classes and speakers.

    This dataset only returns the file path and integer label for each
    sample. Turning the audio into features happens later.
    """

    def __init__(self, raw_data_dir, classes, speaker_range=None):
        """Scan raw_data_dir and keep only files whose emotion is in classes.

        Args:
            raw_data_dir (str): Folder containing the CREMA-D wav files.
            classes (list[str]): Emotion names to keep.
            speaker_range (tuple): Range of speaker IDs to keep, e.g., (1001, 1021).
        """
        self.classes = classes
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        
        # Set default speaker range if not provided
        if speaker_range is None:
            speaker_range = (1001, 1021)  # Default according to the assignment
        self.speaker_range = speaker_range

        all_wav_paths = glob.glob(os.path.join(raw_data_dir, "**", "*.wav"), recursive=True)

        self.file_paths = []
        self.filtered_count = {"total": 0, "by_emotion": 0, "by_speaker": 0}
        
        for wav_path in all_wav_paths:
            emotion_name = get_emotion_from_filename(wav_path)
            speaker_id = get_speaker_id_from_filename(wav_path)
            
            # Filter by emotion
            if emotion_name not in self.classes:
                continue
            self.filtered_count["by_emotion"] += 1
            
            # Filter by speaker range
            if speaker_id is None:
                continue
            if not (speaker_range[0] <= speaker_id <= speaker_range[1]):
                continue
            self.filtered_count["by_speaker"] += 1
            
            self.file_paths.append(wav_path)

        self.filtered_count["total"] = len(self.file_paths)
        
        if len(self.file_paths) == 0:
            print(
                f"Warning: no wav files found in '{raw_data_dir}' for classes {classes} "
                f"and speakers {speaker_range}. "
                "Did you download the dataset first?"
            )
        else:
            print(f"Filtered dataset: {len(self.file_paths)} samples kept "
                  f"(speakers {speaker_range[0]}-{speaker_range[1]}, classes: {classes})")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        """Return (file_path, label) for the sample at this index."""
        file_path = self.file_paths[index]
        emotion_name = get_emotion_from_filename(file_path)
        label = self.class_to_index[emotion_name]
        return file_path, label


def get_dataset_info(raw_data_dir, classes, speaker_range=None):
    """Get information about the dataset.

    Args:
        raw_data_dir (str): Folder containing the CREMA-D wav files.
        classes (list[str]): Emotion names to keep.
        speaker_range (tuple): Range of speaker IDs to keep.

    Returns:
        dict: Information about the dataset.
    """
    dataset = CremaDDataset(raw_data_dir, classes, speaker_range)
    
    # Count samples per class
    class_counts = {cls: 0 for cls in classes}
    speaker_counts = {}
    
    for file_path, _ in dataset:
        emotion_name = get_emotion_from_filename(file_path)
        speaker_id = get_speaker_id_from_filename(file_path)
        
        if emotion_name in class_counts:
            class_counts[emotion_name] += 1
        
        if speaker_id is not None:
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
    
    return {
        "total_samples": len(dataset),
        "class_counts": class_counts,
        "classes": classes,
        "speaker_range": speaker_range,
        "speaker_counts": speaker_counts,
    }