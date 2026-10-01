"""Device management utilities for GPU/CPU control."""

import torch


def get_device_from_config(config):
    """Get the appropriate device based on config settings.
    
    Args:
        config (dict): The loaded config.yaml settings.
        
    Returns:
        torch.device: The device to use (CPU or CUDA).
    """
    device_config = config.get("device", {})
    use_cuda = device_config.get("use_cuda", False)
    cuda_device = device_config.get("cuda_device", 0)
    
    try:
        if use_cuda and torch.cuda.is_available():
            if cuda_device < torch.cuda.device_count():
                device = torch.device(f"cuda:{cuda_device}")
                return device
            else:
                print(f"Warning: CUDA device {cuda_device} not found. "
                      f"Only {torch.cuda.device_count()} GPU(s) available. "
                      f"Falling back to CPU.")
                return torch.device("cpu")
        else:
            if use_cuda and not torch.cuda.is_available():
                print("CUDA requested but not available. Using CPU.")
            return torch.device("cpu")
    except Exception as e:
        print(f"Error setting up device: {e}. Falling back to CPU.")
        return torch.device("cpu")