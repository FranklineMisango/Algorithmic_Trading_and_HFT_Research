"""
Feature Engineering Module for AI-Enhanced 60/40 Portfolio

This module processes economic indicators and creates features
for the machine learning model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Engineer features from raw economic indicators."""
    
    def __init__(self, config: Dict):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scaler = StandardScaler()
        
    def create_lagged_features(self, 
                               indicators: pd.DataFrame, 
                               lags: List[int] = [1, 3, 6]) -> pd.DataFrame:
        """
        Create lagged features from indicators.
        
        Args:
            indicators: DataFrame with economic indicators
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        lagged_features = indicators.copy()
        
        for col in indicators.columns:
            for lag in lags:
                lagged_features[f'{col}_lag_{lag}'] = indicators[col].shift(lag)
        
        return lagged_features
    
    def create_rolling_features(self, 
                                indicators: pd.DataFrame,
                                windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Args:
            indicators: DataFrame with economic indicators
            windows: List of rolling window sizes
            
        Returns:
            DataFrame with rolling features
        """
        rolling_features = indicators.copy()
        
        for col in indicators.columns:
            for window in windows:
                # Rolling mean
                rolling_features[f'{col}_ma_{window}'] = \
                    indicators[col].rolling(window=window).mean()
                
                # Rolling standard deviation
                rolling_features[f'{col}_std_{window}'] = \
                    indicators[col].rolling(window=window).std()
                
                # Rolling min/max
                rolling_features[f'{col}_min_{window}'] = \
                    indicators[col].rolling(window=window).min()
                rolling_features[f'{col}_max_{window}'] = \
                    indicators[col].rolling(window=window).max()
        
        return rolling_features
    
    def create_change_features(self, indicators: pd.DataFrame) -> pd.DataFrame:
        """
        Create change and momentum features.
        
        Args:
            indicators: DataFrame with economic indicators
            
        Returns:
            DataFrame with change features
        """
        change_features = indicators.copy()
        
        for col in indicators.columns:
            # Absolute change
            change_features[f'{col}_change_1m'] = indicators[col].diff(1)
            change_features[f'{col}_change_3m'] = indicators[col].diff(3)
            change_features[f'{col}_change_6m'] = indicators[col].diff(6)
            
            # Percentage change
            change_features[f'{col}_pct_change_1m'] = indicators[col].pct_change(1)
            change_features[f'{col}_pct_change_3m'] = indicators[col].pct_change(3)
            change_features[f'{col}_pct_change_6m'] = indicators[col].pct_change(6)
        
        return change_features
    
    def create_interaction_features(self, indicators: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between indicators.
        
        Args:
            indicators: DataFrame with economic indicators
            
        Returns:
            DataFrame with interaction features
        """
        interaction_features = indicators.copy()
        
        # VIX and Yield Spread interaction (fear + economic outlook)
        if 'VIX' in indicators.columns and 'Yield_Spread' in indicators.columns:
            interaction_features['VIX_x_Spread'] = \
                indicators['VIX'] * indicators['Yield_Spread']
            interaction_features['VIX_div_Spread'] = \
                indicators['VIX'] / (indicators['Yield_Spread'].abs() + 1e-6)
        
        # Interest Rate and VIX interaction (monetary policy + risk)
        if 'Interest_Rate' in indicators.columns and 'VIX' in indicators.columns:
            interaction_features['Rate_x_VIX'] = \
                indicators['Interest_Rate'] * indicators['VIX']
        
        # Interest Rate and Yield Spread (yield curve positioning)
        if 'Interest_Rate' in indicators.columns and 'Yield_Spread' in indicators.columns:
            interaction_features['Rate_x_Spread'] = \
                indicators['Interest_Rate'] * indicators['Yield_Spread']
        
        return interaction_features
    
    def create_regime_features(self, indicators: pd.DataFrame) -> pd.DataFrame:
        """
        Create regime-based features (high/low volatility, inverted yield curve, etc.).
        
        Args:
            indicators: DataFrame with economic indicators
            
        Returns:
            DataFrame with regime features
        """
        regime_features = indicators.copy()
        
        # VIX regimes
        if 'VIX' in indicators.columns:
            vix_median = indicators['VIX'].median()
            regime_features['high_volatility'] = (indicators['VIX'] > vix_median).astype(int)
            regime_features['vix_percentile'] = indicators['VIX'].rank(pct=True)
        
        # Yield curve regimes
        if 'Yield_Spread' in indicators.columns:
            regime_features['inverted_curve'] = (indicators['Yield_Spread'] < 0).astype(int)
            regime_features['flat_curve'] = (indicators['Yield_Spread'].abs() < 0.5).astype(int)
            regime_features['steep_curve'] = (indicators['Yield_Spread'] > 1.5).astype(int)
        
        # Interest rate regimes
        if 'Interest_Rate' in indicators.columns:
            rate_median = indicators['Interest_Rate'].median()
            regime_features['high_rate'] = (indicators['Interest_Rate'] > rate_median).astype(int)
        
        return regime_features
    
    def engineer_all_features(self, 
                              indicators: pd.DataFrame,
                              include_lagged: bool = True,
                              include_rolling: bool = True,
                              include_changes: bool = True,
                              include_interactions: bool = True,
                              include_regimes: bool = True) -> pd.DataFrame:
        """
        Create all engineered features.
        
        Args:
            indicators: DataFrame with economic indicators
            include_lagged: Whether to include lagged features
            include_rolling: Whether to include rolling features
            include_changes: Whether to include change features
            include_interactions: Whether to include interaction features
            include_regimes: Whether to include regime features
            
        Returns:
            DataFrame with all engineered features
        """
        features = indicators.copy()
        
        if include_lagged:
            features = self.create_lagged_features(features)
        
        if include_rolling:
            features = self.create_rolling_features(indicators)
        
        if include_changes:
            change_feats = self.create_change_features(indicators)
            features = pd.concat([features, change_feats], axis=1)
        
        if include_interactions:
            interaction_feats = self.create_interaction_features(indicators)
            features = pd.concat([features, interaction_feats], axis=1)
        
        if include_regimes:
            regime_feats = self.create_regime_features(indicators)
            features = pd.concat([features, regime_feats], axis=1)
        
        # Remove duplicate columns
        features = features.loc[:, ~features.columns.duplicated()]
        
        return features
    
    def prepare_features_for_training(self, 
                                     features: pd.DataFrame,
                                     dropna: bool = True) -> pd.DataFrame:
        """
        Prepare features for model training.
        
        Args:
            features: DataFrame with engineered features
            dropna: Whether to drop rows with NaN values
            
        Returns:
            DataFrame ready for training
        """
        prepared = features.copy()
        
        if dropna:
            # Drop rows with NaN (created by lagging/rolling)
            prepared = prepared.dropna()
        
        # Replace inf values
        prepared = prepared.replace([np.inf, -np.inf], np.nan)
        prepared = prepared.fillna(0)
        
        return prepared
    
    def scale_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using StandardScaler.
        
        Args:
            features: DataFrame with features
            fit: Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            DataFrame with scaled features
        """
        feature_names = features.columns
        
        if fit:
            scaled_values = self.scaler.fit_transform(features)
        else:
            scaled_values = self.scaler.transform(features)
        
        scaled_features = pd.DataFrame(
            scaled_values,
            index=features.index,
            columns=feature_names
        )
        
        return scaled_features
    
    def get_feature_importance_names(self, features: pd.DataFrame) -> List[str]:
        """
        Get list of feature names for importance analysis.
        
        Args:
            features: DataFrame with features
            
        Returns:
            List of feature names
        """
        return features.columns.tolist()


if __name__ == "__main__":
    # Test the module
    import yaml
    from data_acquisition import DataAcquisition
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Fetch data
    data_acq = DataAcquisition(config)
    _, _, indicators = data_acq.get_full_dataset()
    
    # Engineer features
    feature_eng = FeatureEngineer(config)
    features = feature_eng.engineer_all_features(indicators)
    features_prepared = feature_eng.prepare_features_for_training(features)
    
    print("\n" + "="*50)
    print("Feature Engineering Complete!")
    print("="*50)
    print(f"\nOriginal indicators shape: {indicators.shape}")
    print(f"Engineered features shape: {features.shape}")
    print(f"Prepared features shape: {features_prepared.shape}")
    print(f"\nFeature columns ({len(features_prepared.columns)}):")
    for i, col in enumerate(features_prepared.columns):
        print(f"  {i+1}. {col}")
