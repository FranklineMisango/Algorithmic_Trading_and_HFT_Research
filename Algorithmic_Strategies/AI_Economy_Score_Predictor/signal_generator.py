"""
Signal Generator for AI Economy Score Predictor

Generates trading signals from macro predictions.
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, List


class SignalGenerator:
    """Generates trading signals from predictions."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.signals_config = self.config['strategy']['signals']
    
    def generate_national_signal(
        self,
        predictions: pd.DataFrame,
        spf_consensus: pd.DataFrame,
        threshold_std: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate long/short signals for national strategy.
        
        Args:
            predictions: Model predictions
            spf_consensus: SPF consensus forecasts
            threshold_std: Threshold in standard deviations
            
        Returns:
            DataFrame with signals (-1, 0, 1)
        """
        merged = predictions.merge(spf_consensus, on='date', how='inner')
        
        # Calculate prediction differential
        merged['differential'] = merged['prediction'] - merged['spf_forecast']
        
        # Normalize by historical std
        diff_std = merged['differential'].std()
        merged['differential_z'] = merged['differential'] / diff_std
        
        # Generate signals
        merged['signal'] = 0
        merged.loc[merged['differential_z'] > threshold_std, 'signal'] = 1   # Long
        merged.loc[merged['differential_z'] < -threshold_std, 'signal'] = -1  # Short
        
        return merged[['date', 'prediction', 'spf_forecast', 'differential', 'signal']]
    
    def generate_industry_signals(
        self,
        industry_predictions: pd.DataFrame,
        top_k: int = 3
    ) -> pd.DataFrame:
        """
        Generate long/short signals for multi-industry strategy.
        
        Args:
            industry_predictions: Predictions by industry
            top_k: Number of top/bottom industries to trade
            
        Returns:
            DataFrame with industry signals
        """
        # Rank industries by prediction
        industry_predictions['rank'] = industry_predictions.groupby('date')['prediction'].rank(ascending=False)
        
        # Top k = long, bottom k = short
        total_industries = industry_predictions['gics_sector'].nunique()
        
        industry_predictions['signal'] = 0
        industry_predictions.loc[industry_predictions['rank'] <= top_k, 'signal'] = 1
        industry_predictions.loc[industry_predictions['rank'] > (total_industries - top_k), 'signal'] = -1
        
        return industry_predictions[['date', 'gics_sector', 'prediction', 'rank', 'signal']]


if __name__ == "__main__":
    sg = SignalGenerator('config.yaml')
    
    # Test national signal
    dates = pd.date_range('2020-01-01', periods=12, freq='Q')
    predictions = pd.DataFrame({
        'date': dates,
        'prediction': np.random.normal(2.5, 0.8, 12)
    })
    spf = pd.DataFrame({
        'date': dates,
        'spf_forecast': np.random.normal(2.0, 0.5, 12)
    })
    
    signals = sg.generate_national_signal(predictions, spf)
    print("National signals:")
    print(signals)
