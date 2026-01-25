"""
Feature Engineering for RL Agent State Space

Creates features from price spreads, momentum, and volume for RL decision-making.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import yaml

from emrt_calculator import EMRTCalculator


class FeatureEngineer:
    """Generate features for RL agent state representation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.emrt_calc = EMRTCalculator(config_path)
        self.state_features = self.config['rl_agent']['state_features']
        
    def calculate_spread_features(self,
                                   price1: pd.Series,
                                   price2: pd.Series,
                                   window: int = 20) -> pd.DataFrame:
        """
        Calculate spread-based features.
        
        Args:
            price1, price2: Price series for pair
            window: Rolling window for statistics
            
        Returns:
            DataFrame with spread features
        """
        # Calculate log spread
        spread = self.emrt_calc.calculate_spread(price1, price2)
        
        # Z-score (normalized spread)
        zscore = self.emrt_calc.calculate_zscore(spread, window)
        
        # Spread momentum (rate of change)
        spread_momentum = spread.diff(5)  # 5-day momentum
        
        # Spread velocity (acceleration)
        spread_velocity = spread_momentum.diff(5)
        
        # Rolling volatility
        spread_volatility = spread.rolling(window).std()
        
        features = pd.DataFrame({
            'price_spread': spread,
            'spread_zscore': zscore,
            'spread_momentum': spread_momentum,
            'spread_velocity': spread_velocity,
            'spread_volatility': spread_volatility
        }, index=price1.index)
        
        return features
    
    def calculate_volume_features(self,
                                   volume1: pd.Series,
                                   volume2: pd.Series,
                                   window: int = 20) -> pd.DataFrame:
        """
        Calculate volume-based features.
        
        Args:
            volume1, volume2: Volume series for pair
            window: Rolling window
            
        Returns:
            DataFrame with volume features
        """
        # Volume ratio
        volume_ratio = volume1 / (volume2 + 1e-8)  # Avoid division by zero
        
        # Volume imbalance
        total_volume = volume1 + volume2
        volume_imbalance = (volume1 - volume2) / (total_volume + 1e-8)
        
        # Relative volume (vs average)
        vol1_rel = volume1 / volume1.rolling(window).mean()
        vol2_rel = volume2 / volume2.rolling(window).mean()
        
        features = pd.DataFrame({
            'volume_ratio': volume_ratio,
            'volume_imbalance': volume_imbalance,
            'volume1_relative': vol1_rel,
            'volume2_relative': vol2_rel
        }, index=volume1.index)
        
        return features
    
    def calculate_momentum_features(self,
                                     price1: pd.Series,
                                     price2: pd.Series,
                                     windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Calculate momentum features for both stocks.
        
        Args:
            price1, price2: Price series
            windows: List of lookback windows
            
        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=price1.index)
        
        for window in windows:
            # Individual stock returns
            ret1 = price1.pct_change(window)
            ret2 = price2.pct_change(window)
            
            # Relative momentum
            rel_momentum = ret1 - ret2
            
            features[f'return1_{window}d'] = ret1
            features[f'return2_{window}d'] = ret2
            features[f'rel_momentum_{window}d'] = rel_momentum
        
        return features
    
    def calculate_technical_indicators(self,
                                        price1: pd.Series,
                                        price2: pd.Series) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            price1, price2: Price series
            
        Returns:
            DataFrame with technical indicators
        """
        features = pd.DataFrame(index=price1.index)
        
        # RSI-like indicator for spread
        spread = self.emrt_calc.calculate_spread(price1, price2)
        delta = spread.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        features['spread_rsi'] = rsi
        
        # Bollinger Band position
        spread_ma = spread.rolling(20).mean()
        spread_std = spread.rolling(20).std()
        
        bb_position = (spread - spread_ma) / (2 * spread_std + 1e-8)
        features['bb_position'] = bb_position
        
        return features
    
    def create_state_vector(self,
                            price1: pd.Series,
                            price2: pd.Series,
                            volume1: pd.Series = None,
                            volume2: pd.Series = None) -> pd.DataFrame:
        """
        Create complete state vector for RL agent.
        
        Args:
            price1, price2: Price series for pair
            volume1, volume2: Optional volume series
            
        Returns:
            DataFrame with all state features
        """
        # Spread features (core)
        spread_features = self.calculate_spread_features(price1, price2)
        
        # Momentum features
        momentum_features = self.calculate_momentum_features(price1, price2)
        
        # Technical indicators
        technical_features = self.calculate_technical_indicators(price1, price2)
        
        # Combine all features
        state = pd.concat([
            spread_features,
            momentum_features,
            technical_features
        ], axis=1)
        
        # Add volume features if available
        if volume1 is not None and volume2 is not None:
            volume_features = self.calculate_volume_features(volume1, volume2)
            state = pd.concat([state, volume_features], axis=1)
        
        # Forward fill and drop NaNs
        state = state.fillna(method='ffill').dropna()
        
        return state
    
    def normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features for stable RL training.
        
        Args:
            features: Raw feature DataFrame
            
        Returns:
            Normalized features
        """
        normalized = features.copy()
        
        # Z-score normalization for each feature
        for col in features.columns:
            mean = features[col].mean()
            std = features[col].std()
            
            if std > 0:
                normalized[col] = (features[col] - mean) / std
            else:
                normalized[col] = 0
        
        # Clip outliers
        normalized = normalized.clip(-5, 5)
        
        return normalized


if __name__ == "__main__":
    # Test feature engineering
    from data_acquisition import DataAcquisition
    
    # Fetch data
    data_acq = DataAcquisition()
    dataset = data_acq.fetch_full_dataset()
    train_prices, _ = data_acq.split_train_test(dataset['prices'])
    
    # Test with MSFT-GOOGL pair
    if 'MSFT' in train_prices.columns and 'GOOGL' in train_prices.columns:
        feature_eng = FeatureEngineer()
        
        state_features = feature_eng.create_state_vector(
            train_prices['MSFT'],
            train_prices['GOOGL']
        )
        
        print("=== State Features ===")
        print(f"Shape: {state_features.shape}")
        print(f"\nColumns:\n{state_features.columns.tolist()}")
        print(f"\nSample (last 5 rows):\n{state_features.tail()}")
        
        # Normalize
        normalized = feature_eng.normalize_features(state_features)
        print(f"\n=== Normalized Features ===")
        print(f"Mean: {normalized.mean().mean():.4f}")
        print(f"Std: {normalized.std().mean():.4f}")
