"""
Utility Functions
=================

Provides logging setup, random seed configuration, and configuration management.
"""

import logging
import random
import numpy as np
import tensorflow as tf
import yaml
from pathlib import Path
from typing import Dict, Any


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        config: Configuration dictionary with logging settings
        
    Returns:
        Configured logger instance
    """
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = log_config.get("file", "ml_futures_prediction.log")
    
    # Create logger
    logger = logging.getLogger("FuturesPrediction")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    return logger


def set_random_seeds(config: Dict[str, Any]) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        config: Configuration dictionary with seed values
    """
    seed = config.get("random_seed", 42)
    numpy_seed = config.get("numpy_seed", 42)
    tf_seed = config.get("tensorflow_seed", 42)
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(numpy_seed)
    
    # TensorFlow
    tf.random.set_seed(tf_seed)
    
    # Set for hash-based operations
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logging.info(f"Random seeds set: Python={seed}, NumPy={numpy_seed}, TensorFlow={tf_seed}")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_directory(directory: str) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
        
    Returns:
        Path object of created directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy - percentage of correct direction predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Directional accuracy (0-1)
    """
    return np.mean((np.sign(y_true) == np.sign(y_pred)).astype(int))


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary for pretty printing.
    
    Args:
        metrics: Dictionary of metric names and values
        
    Returns:
        Formatted string
    """
    lines = ["Metrics:"]
    for name, value in metrics.items():
        lines.append(f"  {name}: {value:.4f}")
    return "\n".join(lines)
