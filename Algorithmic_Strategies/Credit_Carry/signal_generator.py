"""
Credit Carry Strategy - Signal Generation Module
"""

import pandas as pd
import numpy as np
import yaml


class CreditSignalGenerator:
    """Generate credit carry signals"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.zscore_window = self.config['signals']['zscore_window']
        self.entry_threshold = self.config['signals']['entry_threshold']
        self.exit_threshold = self.config['signals']['exit_threshold']
    
    def calculate_carry(self, spreads_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate carry from credit spreads"""
        print("Calculating credit carry...")
        
        # Carry approximation: current spread level
        # In practice: spread - expected default loss
        carry = spreads_df.copy()
        
        return carry
    
    def calculate_zscore(self, carry_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate z-scores"""
        rolling_mean = carry_df.rolling(window=self.zscore_window, min_periods=60).mean()
        rolling_std = carry_df.rolling(window=self.zscore_window, min_periods=60).std()
        
        zscore = (carry_df - rolling_mean) / rolling_std.replace(0, np.nan)
        return zscore
    
    def generate_signals(self, zscore_df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals with hysteresis"""
        signals = pd.DataFrame(0, index=zscore_df.index, columns=zscore_df.columns)
        
        for col in signals.columns:
            position = 0
            for i in range(len(signals)):
                z = zscore_df.iloc[i][col]
                
                if pd.isna(z):
                    signals.iloc[i][col] = position
                    continue
                
                if position == 0:
                    if z > self.entry_threshold:
                        position = -1  # High spread = sell protection
                    elif z < -self.entry_threshold:
                        position = 1  # Low spread = buy protection
                elif position == -1:
                    if z < self.exit_threshold:
                        position = 0
                elif position == 1:
                    if z > -self.exit_threshold:
                        position = 0
                
                signals.iloc[i][col] = position
        
        return signals
    
    def calculate_returns(self, spreads_df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns from credit positions"""
        # Approximate: ETF price returns
        returns = spreads_df.pct_change()
        strategy_returns = signals.shift(1) * returns
        
        return strategy_returns
