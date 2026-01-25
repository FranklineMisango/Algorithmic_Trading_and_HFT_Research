"""
FX Carry Strategy - Signal Generation Module

Generates carry trade signals based on:
1. Interest rate differentials (carry)
2. Z-score normalization
3. Entry/exit thresholds
"""

import pandas as pd
import numpy as np
import yaml
from typing import Tuple


class CarrySignalGenerator:
    """Generate carry signals with z-score normalization"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.zscore_window = self.config['signals']['zscore_window']
        self.entry_threshold = self.config['signals']['entry_threshold']
        self.exit_threshold = self.config['signals']['exit_threshold']
        
    def calculate_zscore(self, carry_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling z-score of carry
        
        Args:
            carry_df: Interest rate differentials for each pair
            
        Returns:
            Z-scores for each pair
        """
        print(f"Calculating {self.zscore_window}-day rolling z-scores...")
        
        rolling_mean = carry_df.rolling(window=self.zscore_window, min_periods=60).mean()
        rolling_std = carry_df.rolling(window=self.zscore_window, min_periods=60).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        zscore = (carry_df - rolling_mean) / rolling_std
        
        print(f"Z-score shape: {zscore.shape}")
        print(f"Mean absolute z-score: {zscore.abs().mean().mean():.2f}")
        
        return zscore
    
    def generate_signals(self, zscore_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry/exit signals based on z-score thresholds
        
        Signal values:
            1: Long (carry > entry threshold)
           -1: Short (carry < -entry threshold)
            0: Flat (within exit threshold)
            
        Args:
            zscore_df: Z-scores of carry
            
        Returns:
            Signal dataframe with values {-1, 0, 1}
        """
        print(f"\nGenerating signals with entry={self.entry_threshold}, exit={self.exit_threshold}...")
        
        signals = pd.DataFrame(0, index=zscore_df.index, columns=zscore_df.columns)
        
        # Entry signals
        signals[zscore_df > self.entry_threshold] = 1   # Long high carry
        signals[zscore_df < -self.entry_threshold] = -1  # Short negative carry
        
        # Apply hysteresis: only exit when crosses exit threshold
        for col in signals.columns:
            position = 0
            for i in range(len(signals)):
                z = zscore_df.iloc[i][col]
                
                # If no position, check for entry
                if position == 0:
                    if z > self.entry_threshold:
                        position = 1
                    elif z < -self.entry_threshold:
                        position = -1
                # If long, check for exit
                elif position == 1:
                    if z < self.exit_threshold:
                        position = 0
                # If short, check for exit
                elif position == -1:
                    if z > -self.exit_threshold:
                        position = 0
                
                signals.iloc[i][col] = position
        
        # Summary statistics
        long_pct = (signals == 1).sum().sum() / (signals != 0).sum().sum() * 100
        short_pct = (signals == -1).sum().sum() / (signals != 0).sum().sum() * 100
        
        print(f"Signal distribution: {long_pct:.1f}% long, {short_pct:.1f}% short")
        print(f"Average number of positions: {(signals != 0).sum(axis=1).mean():.1f} pairs")
        
        return signals
    
    def calculate_returns(self, spot_df: pd.DataFrame, signals: pd.DataFrame, 
                         carry_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns from FX positions
        
        Returns = spot price change + carry accrual
        
        Args:
            spot_df: Spot FX rates
            signals: Position signals {-1, 0, 1}
            carry_df: Interest rate differentials
            
        Returns:
            Daily returns for each pair
        """
        print("\nCalculating FX returns...")
        
        # Spot returns (percentage change)
        spot_returns = spot_df.pct_change()
        
        # Carry contribution (annualized rate / 252 trading days)
        daily_carry = carry_df / 252
        
        # Total return = spot return + carry
        total_returns = spot_returns + daily_carry
        
        # Apply signals (lagged by 1 day to avoid look-ahead bias)
        strategy_returns = signals.shift(1) * total_returns
        
        print(f"Returns shape: {strategy_returns.shape}")
        print(f"Mean daily return: {strategy_returns.mean().mean()*10000:.2f} bps")
        
        return strategy_returns
    
    def run_signal_generation(self, carry_df: pd.DataFrame, 
                             spot_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete signal generation pipeline
        
        Returns:
            zscore_df, signals_df, returns_df
        """
        print("\n" + "="*60)
        print("SIGNAL GENERATION")
        print("="*60)
        
        # Step 1: Calculate z-scores
        zscore_df = self.calculate_zscore(carry_df)
        
        # Step 2: Generate signals
        signals_df = self.generate_signals(zscore_df)
        
        # Step 3: Calculate returns
        returns_df = self.calculate_returns(spot_df, signals_df, carry_df)
        
        print("\n✓ Signal generation complete")
        
        return zscore_df, signals_df, returns_df


if __name__ == "__main__":
    # Example usage
    from data_acquisition import FXDataAcquisition
    
    # Load data
    fx_data = FXDataAcquisition()
    spots, rates, carry, factors = fx_data.load_data()
    
    # Generate signals
    signal_gen = CarrySignalGenerator()
    zscores, signals, returns = signal_gen.run_signal_generation(carry, spots)
    
    # Display summary
    print(f"\n{signals.describe()}")
