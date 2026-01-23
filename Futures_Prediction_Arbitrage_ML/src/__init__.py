"""
ML-Based Futures Price Prediction Package
==========================================

A comprehensive machine learning system for predicting short-term futures prices
using order book data with proper validation, evaluation, and production deployment.

Modules:
    - data_processor: Data loading, validation, and preprocessing
    - feature_engine: Feature engineering from order book data
    - model_trainer: Model training with cross-validation
    - backtester: Realistic backtesting with transaction costs
    - utils: Utility functions and helpers
"""

__version__ = "2.0.0"
__author__ = "Futures Prediction Team"
__date__ = "2026-01-19"

from src.data_processor import DataProcessor
from src.feature_engine import FeatureEngine
from src.model_trainer import ModelTrainer
from src.backtester import Backtester
from src.utils import setup_logging, set_random_seeds, load_config

__all__ = [
    "DataProcessor",
    "FeatureEngine",
    "ModelTrainer",
    "Backtester",
    "setup_logging",
    "set_random_seeds",
    "load_config",
]
