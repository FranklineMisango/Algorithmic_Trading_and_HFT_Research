"""
Noise Area Calculation Module

This module calculates the "Noise Area" - a volatility-based band around recent
intraday price action. The noise area defines the range of normal market fluctuation.

Breakouts above/below this area signal potential momentum opportunities.

Research Parameters:
- Lookback: 90 days (optimized from original 14 days)
- Method: Statistical boundaries based on intraday range
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class NoiseAreaCalculator:
    """
    Calculates noise area boundaries for breakout detection.
    """
    
    def __init__(self, config: dict):
        """
        Initialize noise area calculator.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.lookback = config['strategy']['noise_area']['lookback_days']
        self.method = config['strategy']['noise_area']['method']
        
        if self.method == 'percentile':
            self.upper_pct = config['strategy']['noise_area']['upper_percentile']
            self.lower_pct = config['strategy']['noise_area']['lower_percentile']
        elif self.method == 'std_dev':
            self.std_mult = config['strategy']['noise_area']['std_multiplier']
    
    def calculate_intraday_range(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate intraday range (High - Low) for each bar.
        
        Parameters
        ----------
        data : pd.DataFrame
            Intraday OHLCV data
            
        Returns
        -------
        pd.Series
            Intraday range series
        """
        return data['High'] - data['Low']
    
    def calculate_noise_area_percentile(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate noise area using percentile method.
        
        Upper Boundary = Price + (Upper Percentile of Range)
        Lower Boundary = Price - (Lower Percentile of Range)
        
        Parameters
        ----------
        data : pd.DataFrame
            Intraday OHLCV data
            
        Returns
        -------
        pd.DataFrame
            Data with upper_boundary and lower_boundary columns
        """
        print(f"Calculating noise area using percentile method ({self.lookback} days)...")
        
        # Calculate intraday range
        data['range'] = self.calculate_intraday_range(data)
        
        # Get trading days for proper lookback
        # Group by date to get daily data
        data['date'] = pd.to_datetime(data.index.date)
        
        # Calculate rolling percentiles of intraday range
        # Use the maximum range per day, then roll over days
        daily_max_range = data.groupby('date')['range'].max()
        
        # Expand back to intraday frequency
        data['daily_max_range'] = data['date'].map(daily_max_range)
        
        # Calculate rolling statistics over lookback period
        data['upper_range'] = data['daily_max_range'].rolling(
            window=self.lookback, min_periods=self.lookback//2
        ).quantile(self.upper_pct / 100)
        
        data['lower_range'] = data['daily_max_range'].rolling(
            window=self.lookback, min_periods=self.lookback//2
        ).quantile(self.lower_pct / 100)
        
        # Calculate boundaries
        # Upper boundary: Current price + upper range threshold
        # Lower boundary: Current price - lower range threshold
        data['upper_boundary'] = data['Close'] + data['upper_range']
        data['lower_boundary'] = data['Close'] - data['lower_range']
        
        print(f"  Calculated noise area for {len(data)} bars")
        print(f"  Avg upper range: {data['upper_range'].mean():.2f}")
        print(f"  Avg lower range: {data['lower_range'].mean():.2f}")
        
        return data
    
    def calculate_noise_area_std(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate noise area using standard deviation method.
        
        Upper Boundary = Mean + (Std * Multiplier)
        Lower Boundary = Mean - (Std * Multiplier)
        
        Parameters
        ----------
        data : pd.DataFrame
            Intraday OHLCV data
            
        Returns
        -------
        pd.DataFrame
            Data with upper_boundary and lower_boundary columns
        """
        print(f"Calculating noise area using std dev method ({self.lookback} days)...")
        
        # Calculate rolling mean and std of close prices
        data['rolling_mean'] = data['Close'].rolling(
            window=self.lookback, min_periods=self.lookback//2
        ).mean()
        
        data['rolling_std'] = data['Close'].rolling(
            window=self.lookback, min_periods=self.lookback//2
        ).std()
        
        # Calculate boundaries
        data['upper_boundary'] = data['rolling_mean'] + (data['rolling_std'] * self.std_mult)
        data['lower_boundary'] = data['rolling_mean'] - (data['rolling_std'] * self.std_mult)
        
        print(f"  Calculated noise area for {len(data)} bars")
        print(f"  Avg std: {data['rolling_std'].mean():.2f}")
        
        return data
    
    def calculate_noise_area_atr(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate noise area using Average True Range (ATR) method.
        
        Parameters
        ----------
        data : pd.DataFrame
            Intraday OHLCV data
            
        Returns
        -------
        pd.DataFrame
            Data with upper_boundary and lower_boundary columns
        """
        print(f"Calculating noise area using ATR method ({self.lookback} days)...")
        
        # Calculate True Range
        data['prev_close'] = data['Close'].shift(1)
        data['tr1'] = data['High'] - data['Low']
        data['tr2'] = abs(data['High'] - data['prev_close'])
        data['tr3'] = abs(data['Low'] - data['prev_close'])
        data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        data['atr'] = data['true_range'].rolling(
            window=self.lookback, min_periods=self.lookback//2
        ).mean()
        
        # Calculate boundaries
        multiplier = 2.0  # Standard ATR multiplier
        data['upper_boundary'] = data['Close'] + (data['atr'] * multiplier)
        data['lower_boundary'] = data['Close'] - (data['atr'] * multiplier)
        
        print(f"  Calculated noise area for {len(data)} bars")
        print(f"  Avg ATR: {data['atr'].mean():.2f}")
        
        return data
    
    def calculate_noise_area(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate noise area boundaries using configured method.
        
        Parameters
        ----------
        data : pd.DataFrame
            Intraday OHLCV data
            
        Returns
        -------
        pd.DataFrame
            Data with upper_boundary and lower_boundary columns
        """
        print("="*60)
        print("NOISE AREA CALCULATION")
        print("="*60)
        
        if self.method == 'percentile':
            data = self.calculate_noise_area_percentile(data)
        elif self.method == 'std_dev':
            data = self.calculate_noise_area_std(data)
        elif self.method == 'atr':
            data = self.calculate_noise_area_atr(data)
        else:
            raise ValueError(f"Unknown noise area method: {self.method}")
        
        # Validate boundaries
        valid_boundaries = data[['upper_boundary', 'lower_boundary']].notna().all(axis=1)
        pct_valid = valid_boundaries.mean() * 100
        
        print(f"\nValidation:")
        print(f"  Valid boundaries: {pct_valid:.1f}%")
        print(f"  Avg boundary width: {(data['upper_boundary'] - data['lower_boundary']).mean():.2f}")
        
        print("="*60)
        
        return data
    
    def identify_breakouts(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify when price breaks out of noise area.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with noise area boundaries
            
        Returns
        -------
        pd.DataFrame
            Data with breakout signals
        """
        # Breakout conditions
        data['break_above'] = data['Close'] > data['upper_boundary']
        data['break_below'] = data['Close'] < data['lower_boundary']
        data['inside_noise'] = ~(data['break_above'] | data['break_below'])
        
        # Track when price re-enters noise area (momentum failure)
        data['momentum_failure'] = (
            (data['inside_noise']) & 
            ((data['break_above'].shift(1)) | (data['break_below'].shift(1)))
        )
        
        return data


def visualize_noise_area(data: pd.DataFrame, symbol: str, start_idx: int = 0, end_idx: int = 500):
    """
    Visualize noise area with price action.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with noise area boundaries
    symbol : str
        Instrument symbol
    start_idx : int
        Start index for visualization
    end_idx : int
        End index for visualization
    """
    import matplotlib.pyplot as plt
    
    # Slice data for visualization
    plot_data = data.iloc[start_idx:end_idx]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot price
    ax.plot(plot_data.index, plot_data['Close'], 
            label='Close Price', color='black', linewidth=1.5, zorder=3)
    
    # Plot noise area boundaries
    ax.plot(plot_data.index, plot_data['upper_boundary'], 
            label='Upper Boundary', color='red', linestyle='--', alpha=0.7, zorder=2)
    ax.plot(plot_data.index, plot_data['lower_boundary'], 
            label='Lower Boundary', color='green', linestyle='--', alpha=0.7, zorder=2)
    
    # Fill noise area
    ax.fill_between(plot_data.index, 
                     plot_data['lower_boundary'], 
                     plot_data['upper_boundary'],
                     alpha=0.1, color='gray', label='Noise Area')
    
    # Mark breakouts
    break_above = plot_data[plot_data['break_above']]
    break_below = plot_data[plot_data['break_below']]
    
    if len(break_above) > 0:
        ax.scatter(break_above.index, break_above['Close'], 
                   color='green', marker='^', s=100, 
                   label='Break Above', zorder=4, alpha=0.7)
    
    if len(break_below) > 0:
        ax.scatter(break_below.index, break_below['Close'], 
                   color='red', marker='v', s=100, 
                   label='Break Below', zorder=4, alpha=0.7)
    
    ax.set_title(f'{symbol} - Noise Area & Breakouts', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{symbol}_noise_area.png', dpi=150, bbox_inches='tight')
    print(f"\nNoise area visualization saved to results/{symbol}_noise_area.png")
    plt.show()


def main():
    """
    Test noise area calculator.
    """
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate sample data (in real implementation, load actual futures data)
    print("Generating sample intraday data...")
    dates = pd.date_range('2023-01-01 09:30', '2023-12-31 16:00', freq='5min')
    n = len(dates)
    
    # Simulate price with noise and trends
    np.random.seed(42)
    price = 4500 + np.cumsum(np.random.randn(n) * 2) + np.random.randn(n) * 10
    
    data = pd.DataFrame({
        'Open': price + np.random.randn(n) * 2,
        'High': price + abs(np.random.randn(n) * 5),
        'Low': price - abs(np.random.randn(n) * 5),
        'Close': price,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    print(f"Sample data: {len(data)} bars")
    
    # Calculate noise area
    calculator = NoiseAreaCalculator(config)
    data = calculator.calculate_noise_area(data)
    data = calculator.identify_breakouts(data)
    
    # Statistics
    print(f"\nBreakout Statistics:")
    print(f"  Break above: {data['break_above'].sum()} bars ({data['break_above'].mean()*100:.1f}%)")
    print(f"  Break below: {data['break_below'].sum()} bars ({data['break_below'].mean()*100:.1f}%)")
    print(f"  Inside noise: {data['inside_noise'].sum()} bars ({data['inside_noise'].mean()*100:.1f}%)")
    print(f"  Momentum failures: {data['momentum_failure'].sum()}")
    
    # Visualize
    visualize_noise_area(data, 'ES', 0, 1000)
    
    # Save
    data.to_csv('results/noise_area_sample.csv')
    print("\nSample data saved to results/noise_area_sample.csv")


if __name__ == "__main__":
    main()
