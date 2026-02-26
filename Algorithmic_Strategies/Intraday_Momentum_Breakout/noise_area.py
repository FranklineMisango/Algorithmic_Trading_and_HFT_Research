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
        
        # Calculate daily max range
        daily_max_range = data.groupby('date')['range'].max()
        
        # Filter outliers using IQR method to handle corrupted data
        q1 = daily_max_range.quantile(0.25)
        q3 = daily_max_range.quantile(0.75)
        iqr = q3 - q1
        upper_fence = q3 + (3.0 * iqr)  # 3x IQR for futures volatility
        
        # Cap extreme values
        daily_max_range_clean = daily_max_range.clip(upper=upper_fence)
        
        outliers_pct = ((daily_max_range > upper_fence).sum() / len(daily_max_range)) * 100
        print(f"  Filtered {outliers_pct:.1f}% outlier days (range > {upper_fence:.2f})")
        
        # Calculate rolling percentiles on DAILY data (not intraday)
        # min_periods=1 so the very first trading day already has a boundary
        # (uses however many days of history are available, growing to lookback)
        daily_upper_range = daily_max_range_clean.rolling(
            window=self.lookback, min_periods=1
        ).quantile(self.upper_pct / 100)
        
        daily_lower_range = daily_max_range_clean.rolling(
            window=self.lookback, min_periods=1
        ).quantile(self.lower_pct / 100)
        
        # Expand back to intraday frequency (forward-fill within each day)
        data['upper_range'] = data['date'].map(daily_upper_range)
        data['lower_range'] = data['date'].map(daily_lower_range)
        
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
        # Breakout conditions - use High/Low for intraday breakout detection
        # This checks if price ACTION breached boundaries during the bar
        data['break_above'] = data['High'] > data['upper_boundary']
        data['break_below'] = data['Low'] < data['lower_boundary']
        data['inside_noise'] = ~(data['break_above'] | data['break_below'])
        
        # Track when price re-enters noise area (momentum failure)
        data['momentum_failure'] = (
            (data['inside_noise']) & 
            ((data['break_above'].shift(1)) | (data['break_below'].shift(1)))
        )
        
        return data


def visualize_noise_area(data: pd.DataFrame, symbol: str, start_idx: int = 0, end_idx: int = 500, config: dict = None):
    """
    Visualize noise area with price action.

    Shows daily-fixed noise area boundaries: at the first bar of each trading
    day the session-open price ± daily range threshold is used as flat
    horizontal levels for the whole day, so you can see price breaking out
    of the zone visually.

    Parameters
    ----------
    data : pd.DataFrame
        Data with OHLCV columns (will calculate boundaries if not present)
    symbol : str
        Instrument symbol
    start_idx : int
        Start index for visualization (within the valid / post-warmup rows)
    end_idx : int
        End index for visualization
    config : dict, optional
        Configuration dictionary for boundary calculation
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # ------------------------------------------------------------------
    # 1. Ensure noise-area columns are present
    # ------------------------------------------------------------------
    if 'upper_boundary' not in data.columns:
        if config is None:
            try:
                import yaml
                with open('config.yaml', 'r') as f:
                    config = yaml.safe_load(f)
            except FileNotFoundError:
                print(f"Warning: config.yaml not found. Cannot calculate boundaries for {symbol}.")
                return
        calculator = NoiseAreaCalculator(config)
        data = calculator.calculate_noise_area(data.copy())
        data = calculator.identify_breakouts(data)

    # ------------------------------------------------------------------
    # 2. Drop NaN warmup rows so start_idx=0 is the first *valid* bar
    # ------------------------------------------------------------------
    if 'upper_boundary' in data.columns:
        valid_data = data.dropna(subset=['upper_boundary', 'lower_boundary']).copy()
        if len(valid_data) == 0:
            print(f"Warning: No valid boundary data for {symbol} — all NaN.")
            return
        skipped = len(data) - len(valid_data)
        if skipped:
            print(f"  Skipped {skipped:,} warmup bars (NaN boundaries) for {symbol}")
    else:
        valid_data = data.copy()

    # ------------------------------------------------------------------
    # 3. Slice the requested window
    # ------------------------------------------------------------------
    plot_data = valid_data.iloc[start_idx:end_idx].copy()

    # ------------------------------------------------------------------
    # 4. Build SESSION-FIXED boundaries
    #    At the first bar of each trading day, pin the noise zone to the
    #    session-open price ± that day's range thresholds.  Within the day
    #    the band stays flat so breakouts are clearly visible.
    # ------------------------------------------------------------------
    plot_data['_date'] = plot_data.index.normalize()   # tz-aware midnight

    # First bar of each day → session open and that day's range offsets
    daily_first = plot_data.groupby('_date').first()[['Open', 'upper_range', 'lower_range']]
    daily_first.columns = ['_sess_open', '_u_range', '_l_range']

    plot_data = plot_data.join(daily_first, on='_date')
    plot_data['daily_upper'] = plot_data['_sess_open'] + plot_data['_u_range']
    plot_data['daily_lower'] = plot_data['_sess_open'] - plot_data['_l_range']
    plot_data.drop(columns=['_date', '_sess_open', '_u_range', '_l_range'], inplace=True)

    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 8))

    # Candlestick-style shading: High–Low range per bar (light background)
    ax.fill_between(plot_data.index, plot_data['Low'], plot_data['High'],
                    alpha=0.08, color='steelblue', label='Bar Range (H–L)')

    # Close price spine
    ax.plot(plot_data.index, plot_data['Close'],
            label='Close', color='black', linewidth=1.2, zorder=3)

    # Daily-fixed noise area boundaries
    ax.plot(plot_data.index, plot_data['daily_upper'],
            label='Upper Boundary (daily fixed)', color='red',
            linestyle='--', linewidth=1.4, alpha=0.85, zorder=4)
    ax.plot(plot_data.index, plot_data['daily_lower'],
            label='Lower Boundary (daily fixed)', color='green',
            linestyle='--', linewidth=1.4, alpha=0.85, zorder=4)

    # Noise area fill (between daily-fixed levels)
    ax.fill_between(plot_data.index,
                    plot_data['daily_lower'], plot_data['daily_upper'],
                    alpha=0.12, color='gold', label='Noise Area')

    # Breakout markers
    if 'break_above' in plot_data.columns:
        breaks_up = plot_data[plot_data['break_above']]
        if len(breaks_up):
            ax.scatter(breaks_up.index, breaks_up['High'],
                       color='limegreen', marker='^', s=80,
                       label=f'Breakout Up ({len(breaks_up)})', zorder=5, alpha=0.9)

    if 'break_below' in plot_data.columns:
        breaks_dn = plot_data[plot_data['break_below']]
        if len(breaks_dn):
            ax.scatter(breaks_dn.index, breaks_dn['Low'],
                       color='crimson', marker='v', s=80,
                       label=f'Breakout Down ({len(breaks_dn)})', zorder=5, alpha=0.9)

    ax.set_title(f'{symbol} — Noise Area & Breakouts  '
                 f'({plot_data.index[0].strftime("%Y-%m-%d")} → '
                 f'{plot_data.index[-1].strftime("%Y-%m-%d")})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.autofmt_xdate()

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
    
    # Load real ES data
    print("Loading ES data...")
    data = pd.read_csv(
        'Data/ES_5min_RTH.csv',
        index_col='ts_event',
        parse_dates=True
    )
    
    print(f"Loaded {len(data)} bars")
    
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
    data.to_csv('results/noise_area_es.csv')
    print("\nNoise area data saved to results/noise_area_es.csv")


if __name__ == "__main__":
    main()
