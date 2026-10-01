"""Data module for loading and managing CREMA-D dataset."""

from data.data_loader import (
    CremaDDataset,
    download_crema_d,
    get_emotion_from_filename,
    get_dataset_info,
)

__all__ = [
    "CremaDDataset",
    "download_crema_d",
    "get_emotion_from_filename",
    "get_dataset_info",
]