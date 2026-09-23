"""
utils.config_loader built to read and write the config/config.yaml file
"""


import yaml
from typing import Dict, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Dictionary containing all config parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model parameters from config.
    
    Args:
        config: Full config dictionary
        
    Returns:
        Dictionary with model-specific parameters:
        - hidden_layers: list of hidden layer sizes
        - weight_mean: mean for weight initialization
        - weight_std: std for weight initialization  
        - bias_init: initial bias value
    """
    model_cfg = config['model']
    return {
        'hidden_layers': model_cfg['hidden_layers'],
        'weight_mean': model_cfg['weight_init']['mean'],
        'weight_std': model_cfg['weight_init']['std'],
        'bias_init': model_cfg['bias_init']['value'],
        'momentum': model_cfg['momentum']

    }


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training parameters from config.
    
    Args:
        config: Full config dictionary
        
    Returns:
        Dictionary with training parameters
    """
    return config['training']


def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data parameters from config.
    
    Args:
        config: Full config dictionary
        
    Returns:
        Dictionary with data parameters
    """
    return config['data']


def get_paths_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract path parameters from config.
    
    Args:
        config: Full config dictionary
        
    Returns:
        Dictionary with path parameters
    """
    return config['paths']
