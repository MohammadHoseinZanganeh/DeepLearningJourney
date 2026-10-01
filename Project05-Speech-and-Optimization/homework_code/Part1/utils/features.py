"""Functions for turning a raw audio waveform into a fixed-size feature vector.

Two feature types are supported:
    1. Log-mel spectrogram (a classic hand-crafted audio feature).
    2. HuBERT embeddings (a self-supervised deep-learning feature).

Both feature vectors are mean-pooled across time, so every audio file
(no matter how long) produces one fixed-size vector that we can feed
into the classifier.
"""

import os
import librosa
import numpy as np
import torch
import yaml

_hubert_feature_extractor = None
_hubert_model = None
_hubert_device = None
_hubert_model_name = None


def get_project_root():
    """Get the project root directory (Part1 folder)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_device_from_config(config_path=None):
    """Get the device from config file."""
    if config_path is None:
        project_root = get_project_root()
        config_path = os.path.join(project_root, "config", "config.yaml")
    
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    device_config = config.get("device", {})
    use_cuda = device_config.get("use_cuda", False)
    cuda_device = device_config.get("cuda_device", 0)
    
    try:
        if use_cuda and torch.cuda.is_available():
            if cuda_device < torch.cuda.device_count():
                return torch.device(f"cuda:{cuda_device}")
            else:
                print(f"Warning: CUDA device {cuda_device} not found. Falling back to CPU.")
                return torch.device("cpu")
        else:
            if use_cuda and not torch.cuda.is_available():
                print("CUDA requested but not available. Using CPU for HuBERT.")
            return torch.device("cpu")
    except Exception as e:
        print(f"Error setting up device for HuBERT: {e}. Falling back to CPU.")
        return torch.device("cpu")


def load_audio(file_path, sampling_rate, max_duration=3.0):
    """Load a wav file, resample it, and pad/truncate to fixed duration.

    Args:
        file_path (str): Path to the wav file.
        sampling_rate (int): Target sampling rate in Hz.
        max_duration (float): Maximum duration in seconds.

    Returns:
        numpy.ndarray: 1D audio waveform with fixed length.
    """
    waveform, _ = librosa.load(file_path, sr=sampling_rate)
    
    # Calculate target length
    target_length = int(max_duration * sampling_rate)
    
    # Pad or truncate to fixed length
    if len(waveform) < target_length:
        # Pad with zeros
        padding = target_length - len(waveform)
        waveform = np.pad(waveform, (0, padding), mode='constant')
    elif len(waveform) > target_length:
        # Truncate
        waveform = waveform[:target_length]
    
    return waveform


def extract_mel_spectrogram_feature(waveform, sampling_rate, n_mels, n_fft, hop_length):
    """Compute a mean-pooled log-mel spectrogram feature vector.

    Args:
        waveform (numpy.ndarray): 1D audio waveform.
        sampling_rate (int): Sampling rate of the waveform.
        n_mels (int): Number of mel frequency bands.
        n_fft (int): FFT window size.
        hop_length (int): Hop length between FFT windows.

    Returns:
        numpy.ndarray: 1D feature vector of size n_mels.
    """
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform, sr=sampling_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    feature_vector = np.mean(log_mel_spectrogram, axis=1)
    return feature_vector


def _get_hubert_model(model_name):
    """Load the HuBERT feature extractor and model once, then reuse them."""
    global _hubert_feature_extractor, _hubert_model, _hubert_device, _hubert_model_name

    if _hubert_model is None or _hubert_model_name != model_name:
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        _hubert_device = get_device_from_config()
        print(f"Loading HuBERT model '{model_name}' on {_hubert_device}...")
        print("(This only happens once and may take a moment)")
        
        _hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        _hubert_model = HubertModel.from_pretrained(
            model_name,
            use_safetensors=True  # Only download safetensors format to avoid double download
        )
        _hubert_model = _hubert_model.to(_hubert_device)
        _hubert_model.eval()
        _hubert_model_name = model_name
        print("HuBERT model loaded successfully.")

    return _hubert_feature_extractor, _hubert_model


def extract_hubert_feature(waveform, sampling_rate, model_name):
    """Compute a mean-pooled HuBERT embedding feature vector.

    Args:
        waveform (numpy.ndarray): 1D audio waveform, must be 16 kHz.
        sampling_rate (int): Sampling rate of the waveform.
        model_name (str): HuggingFace model name.

    Returns:
        numpy.ndarray: 1D feature vector (768 values for the base model).
    """
    feature_extractor, model = _get_hubert_model(model_name)
    device = next(model.parameters()).device

    inputs = feature_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    feature_vector = last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return feature_vector