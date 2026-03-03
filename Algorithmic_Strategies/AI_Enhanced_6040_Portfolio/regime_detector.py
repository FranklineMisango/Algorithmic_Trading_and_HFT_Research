"""
Regime Detection Module for AI-Enhanced 60/40 Portfolio

Detects market regimes (bull/bear/sideways) to adjust risk parameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.cluster import KMeans


class RegimeDetector:
    """Detect market regimes using multiple methods."""
    
    def __init__(self, config: Dict):
        """
        Initialize regime detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
    def detect_volatility_regime(self, vix: pd.Series) -> pd.Series:
        """
        Detect volatility regime based on VIX levels.
        
        Args:
            vix: VIX series
            
        Returns:
            Series with regime labels (0=low, 1=medium, 2=high)
        """
        vix_25 = vix.quantile(0.33)
        vix_75 = vix.quantile(0.67)
        
        regime = pd.Series(1, index=vix.index)  # Default: medium
        regime[vix < vix_25] = 0  # Low volatility
        regime[vix > vix_75] = 2  # High volatility
        
        return regime
    
    def detect_trend_regime(self, prices: pd.Series, lookback: int = 60) -> pd.Series:
        """
        Detect trend regime using moving averages.
        
        Args:
            prices: Price series
            lookback: Lookback period for moving average
            
        Returns:
            Series with regime labels (0=bear, 1=sideways, 2=bull)
        """
        ma = prices.rolling(lookback).mean()
        price_vs_ma = (prices - ma) / ma
        
        regime = pd.Series(1, index=prices.index)  # Default: sideways
        regime[price_vs_ma > 0.05] = 2  # Bull: price > 5% above MA
        regime[price_vs_ma < -0.05] = 0  # Bear: price > 5% below MA
        
        return regime
    
    def detect_yield_curve_regime(self, yield_spread: pd.Series) -> pd.Series:
        """
        Detect yield curve regime.
        
        Args:
            yield_spread: Yield spread series (10Y - 3M)
            
        Returns:
            Series with regime labels (0=inverted, 1=flat, 2=steep)
        """
        regime = pd.Series(1, index=yield_spread.index)  # Default: flat
        regime[yield_spread < 0] = 0  # Inverted
        regime[yield_spread > 1.5] = 2  # Steep
        
        return regime
    
    def detect_combined_regime(self, 
                               vix: pd.Series,
                               prices: pd.Series,
                               yield_spread: pd.Series) -> pd.Series:
        """
        Detect combined market regime using multiple indicators.
        
        Args:
            vix: VIX series
            prices: Price series (e.g., SPY)
            yield_spread: Yield spread series
            
        Returns:
            Series with regime labels (0=defensive, 1=neutral, 2=aggressive)
        """
        vol_regime = self.detect_volatility_regime(vix)
        trend_regime = self.detect_trend_regime(prices)
        curve_regime = self.detect_yield_curve_regime(yield_spread)
        
        # Combine regimes: average and threshold
        combined = (vol_regime + trend_regime + curve_regime) / 3
        
        regime = pd.Series(1, index=vix.index)  # Default: neutral
        regime[combined < 0.8] = 0  # Defensive
        regime[combined > 1.5] = 2  # Aggressive
        
        return regime
    
    def get_risk_aversion_parameter(self, regime: pd.Series) -> pd.Series:
        """
        Convert regime to risk aversion parameter.
        
        Args:
            regime: Regime series (0=defensive, 1=neutral, 2=aggressive)
            
        Returns:
            Series with risk aversion values
        """
        risk_aversion = regime.copy().astype(float)
        risk_aversion[regime == 0] = 0.5  # Defensive: high risk aversion
        risk_aversion[regime == 1] = 0.2  # Neutral: moderate
        risk_aversion[regime == 2] = 0.1  # Aggressive: low risk aversion
        
        return risk_aversion
    
    def detect_regime_with_clustering(self, 
                                     features: pd.DataFrame,
                                     n_regimes: int = 3) -> pd.Series:
        """
        Detect regimes using K-means clustering on multiple features.
        
        Args:
            features: DataFrame with market features
            n_regimes: Number of regimes to detect
            
        Returns:
            Series with regime labels
        """
        # Normalize features
        features_norm = (features - features.mean()) / features.std()
        features_norm = features_norm.fillna(0)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        regimes = kmeans.fit_predict(features_norm)
        
        regime_series = pd.Series(regimes, index=features.index)
        
        return regime_series


if __name__ == "__main__":
    # Test the module
    import yaml
    from data_acquisition import DataAcquisition
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Fetch data
    data_acq = DataAcquisition(config)
    prices, returns, indicators = data_acq.get_full_dataset()
    
    # Create regime detector
    detector = RegimeDetector(config)
    
    # Detect regimes
    if 'VIX' in indicators.columns and 'Yield_Spread' in indicators.columns:
        regime = detector.detect_combined_regime(
            indicators['VIX'],
            prices['SPY'],
            indicators['Yield_Spread']
        )
        
        print("\n" + "="*50)
        print("Regime Detection Complete!")
        print("="*50)
        print(f"\nRegime distribution:")
        print(regime.value_counts())
        print(f"\nCurrent regime: {regime.iloc[-1]}")
