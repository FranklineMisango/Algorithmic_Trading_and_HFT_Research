"""
Feature Engineering for DRL Portfolio

Creates state representations for RL agent.
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, List


class FeatureEngineer:
    """Prepares state features for RL environment."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.lookback = self.config['data']['lookback_window']
    
    def calculate_rolling_returns(
        self,
        prices: pd.DataFrame,
        window: int
    ) -> pd.DataFrame:
        """
        Calculate rolling returns.
        
        Args:
            prices: Price DataFrame
            window: Lookback window
            
        Returns:
            Rolling returns
        """
        returns = prices.pct_change(window)
        return returns
    
    def calculate_rolling_volatility(
        self,
        returns: pd.DataFrame,
        window: int
    ) -> pd.DataFrame:
        """
        Calculate rolling volatility (annualized).
        
        Args:
            returns: Returns DataFrame
            window: Lookback window
            
        Returns:
            Rolling volatility
        """
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return volatility
    
    def calculate_momentum(
        self,
        prices: pd.DataFrame,
        window: int
    ) -> pd.DataFrame:
        """
        Calculate price momentum.
        
        Args:
            prices: Price DataFrame
            window: Lookback window
            
        Returns:
            Momentum indicators
        """
        momentum = prices / prices.shift(window) - 1
        return momentum
    
    def calculate_correlation_matrix(
        self,
        returns: pd.DataFrame,
        window: int
    ) -> np.ndarray:
        """
        Calculate rolling correlation matrix.
        
        Args:
            returns: Returns DataFrame
            window: Lookback window
            
        Returns:
            Correlation matrix (n_assets x n_assets)
        """
        corr_matrix = returns.rolling(window=window).corr().iloc[-len(returns.columns):]
        return corr_matrix.values
    
    def normalize_features(
        self,
        features: pd.DataFrame,
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Normalize features.
        
        Args:
            features: Feature DataFrame
            method: 'zscore' or 'minmax'
            
        Returns:
            Normalized features
        """
        if method == 'zscore':
            # Z-score normalization
            mean = features.mean()
            std = features.std()
            normalized = (features - mean) / (std + 1e-8)
        
        elif method == 'minmax':
            # Min-max normalization
            min_val = features.min()
            max_val = features.max()
            normalized = (features - min_val) / (max_val - min_val + 1e-8)
        
        else:
            normalized = features
        
        return normalized
    
    def create_state_features(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        current_weights: np.ndarray
    ) -> np.ndarray:
        """
        Create full state vector for RL agent.
        
        Args:
            prices: Price DataFrame
            returns: Returns DataFrame
            current_weights: Current portfolio weights
            
        Returns:
            State vector (flattened)
        """
        # Rolling returns
        rolling_returns = self.calculate_rolling_returns(prices, self.lookback)
        
        # Rolling volatility
        rolling_vol = self.calculate_rolling_volatility(returns, self.lookback)
        
        # Momentum
        momentum = self.calculate_momentum(prices, self.lookback)
        
        # Get latest values
        latest_returns = rolling_returns.iloc[-1].values
        latest_vol = rolling_vol.iloc[-1].values
        latest_momentum = momentum.iloc[-1].values
        
        # Correlation matrix (flattened upper triangle)
        corr_matrix = self.calculate_correlation_matrix(returns, self.lookback)
        corr_flat = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        # Combine all features
        state = np.concatenate([
            current_weights,      # Current allocation
            latest_returns,       # Recent returns
            latest_vol,           # Volatility
            latest_momentum,      # Momentum
            corr_flat             # Correlations
        ])
        
        # Normalize
        state = (state - np.mean(state)) / (np.std(state) + 1e-8)
        
        return state


# Test code
if __name__ == "__main__":
    from data_acquisition import DataAcquisition
    
    data_acq = DataAcquisition('config.yaml')
    dataset = data_acq.fetch_full_dataset()
    
    engineer = FeatureEngineer('config.yaml')
    
    prices = dataset['train']['prices']
    returns = dataset['train']['returns']
    
    # Test feature creation
    current_weights = np.array([0.25, 0.25, 0.25, 0.25])
    state = engineer.create_state_features(prices, returns, current_weights)
    
    print(f"State shape: {state.shape}")
    print(f"State vector (first 10): {state[:10]}")
