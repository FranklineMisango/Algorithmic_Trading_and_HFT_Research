"""
Rates Carry Strategy - Signal Generation Module
Generates signals based on roll-down yield
"""

import pandas as pd
import numpy as np
import yaml


class RatesSignalGenerator:
    """Generate carry signals from roll-down yield"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.zscore_window = self.config['signals']['zscore_window']
        self.entry_threshold = self.config['signals']['entry_threshold']
        self.exit_threshold = self.config['signals']['exit_threshold']
        
    def calculate_zscore(self, rolldown_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling z-score of roll-down yield"""
        print(f"Calculating {self.zscore_window}-day rolling z-scores...")
        
        rolling_mean = rolldown_df.rolling(window=self.zscore_window, min_periods=60).mean()
        rolling_std = rolldown_df.rolling(window=self.zscore_window, min_periods=60).std()
        rolling_std = rolling_std.replace(0, np.nan)
        
        zscore = (rolldown_df - rolling_mean) / rolling_std
        
        print(f"Z-score shape: {zscore.shape}")
        return zscore
    
    def generate_signals(self, zscore_df: pd.DataFrame) -> pd.DataFrame:
        """Generate entry/exit signals with hysteresis"""
        print(f"\nGenerating signals with entry={self.entry_threshold}, exit={self.exit_threshold}...")
        
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
                        position = 1
                    elif z < -self.entry_threshold:
                        position = -1
                elif position == 1:
                    if z < self.exit_threshold:
                        position = 0
                elif position == -1:
                    if z > -self.exit_threshold:
                        position = 0
                
                signals.iloc[i][col] = position
        
        long_pct = (signals == 1).sum().sum() / (signals != 0).sum().sum() * 100 if (signals != 0).sum().sum() > 0 else 0
        print(f"Signal distribution: {long_pct:.1f}% long, {100-long_pct:.1f}% short")
        
        return signals
    
    def calculate_returns(self, yields_df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns from bond positions"""
        print("\nCalculating bond returns...")
        
        # Approximate duration for each maturity
        durations = {'2Y': 1.9, '5Y': 4.5, '7Y': 6.3, '10Y': 9.0, '30Y': 20.0}
        
        # Bond return ≈ -duration × Δyield + carry
        yield_changes = yields_df.diff()
        returns = pd.DataFrame(index=yields_df.index, columns=yields_df.columns)
        
        for col in yields_df.columns:
            mat = col.split('_')[-1]
            duration = durations.get(mat, 5.0)
            
            # Price return from yield change
            price_return = -duration * yield_changes[col] / 100  # Yields in percentage points
            
            # Carry (yield/252)
            carry = yields_df[col] / 252 / 100
            
            returns[col] = price_return + carry
        
        # Apply signals
        strategy_returns = signals.shift(1) * returns
        
        print(f"Mean daily return: {strategy_returns.mean().mean()*10000:.2f} bps")
        return strategy_returns


if __name__ == "__main__":
    from data_acquisition import RatesDataAcquisition
    
    rates_data = RatesDataAcquisition()
    yields, rolldown = rates_data.load_data()
    
    signal_gen = RatesSignalGenerator()
    zscores = signal_gen.calculate_zscore(rolldown)
    signals = signal_gen.generate_signals(zscores)
    returns = signal_gen.calculate_returns(yields, signals)
