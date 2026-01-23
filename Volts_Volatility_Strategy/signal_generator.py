"""
Trading Signal Generator Module

Generates buy/sell signals based on Granger causality relationships
and trend-following indicators on predictor stock's volatility.

If Stock X's volatility Granger-causes Stock Y's volatility:
- Positive trend in X's volatility -> BUY Y
- Negative trend in X's volatility -> SELL Y (or SHORT Y)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum


class Signal(Enum):
    """Trading signal types."""
    BUY = 1
    SELL = -1
    HOLD = 0


class TrendMethod(Enum):
    """Trend detection methods."""
    SMA_CROSSOVER = "sma_crossover"
    LINEAR_REGRESSION = "linear_regression"
    MACD = "macd"
    RATE_OF_CHANGE = "rate_of_change"


class SignalGenerator:
    """
    Generate trading signals based on volatility trends.
    """
    
    def __init__(
        self,
        trend_method: str = 'sma_crossover',
        trend_params: Optional[Dict] = None
    ):
        """
        Initialize signal generator.
        
        Parameters:
        -----------
        trend_method : str
            Method for trend detection
        trend_params : Dict, optional
            Parameters for trend method
        """
        self.trend_method = TrendMethod(trend_method)
        self.trend_params = trend_params or {}
        
        # Default parameters for each method
        self._default_params = {
            TrendMethod.SMA_CROSSOVER: {'fast_period': 5, 'slow_period': 20},
            TrendMethod.LINEAR_REGRESSION: {'window': 20},
            TrendMethod.MACD: {'fast': 12, 'slow': 26, 'signal': 9},
            TrendMethod.RATE_OF_CHANGE: {'window': 10}
        }
        
        # Merge with defaults
        default = self._default_params.get(self.trend_method, {})
        self.params = {**default, **self.trend_params}
    
    def detect_trend_sma_crossover(
        self, 
        series: pd.Series
    ) -> pd.Series:
        """
        Detect trend using Simple Moving Average crossover.
        
        Trend is positive when fast SMA > slow SMA, negative otherwise.
        
        Parameters:
        -----------
        series : pd.Series
            Time series data (e.g., volatility)
            
        Returns:
        --------
        pd.Series : Trend signal (+1 for uptrend, -1 for downtrend, 0 for neutral)
        """
        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']
        
        fast_sma = series.rolling(window=fast_period).mean()
        slow_sma = series.rolling(window=slow_period).mean()
        
        trend = pd.Series(0, index=series.index)
        trend[fast_sma > slow_sma] = 1  # Uptrend
        trend[fast_sma < slow_sma] = -1  # Downtrend
        
        return trend
    
    def detect_trend_linear_regression(
        self,
        series: pd.Series
    ) -> pd.Series:
        """
        Detect trend using rolling linear regression slope.
        
        Positive slope indicates uptrend, negative slope indicates downtrend.
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
            
        Returns:
        --------
        pd.Series : Trend signal
        """
        window = self.params['window']
        
        def calculate_slope(window_data):
            if len(window_data) < 2:
                return 0
            x = np.arange(len(window_data))
            y = window_data.values
            # Simple linear regression: slope = cov(x,y) / var(x)
            slope = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else 0
            return slope
        
        slopes = series.rolling(window=window).apply(calculate_slope, raw=False)
        
        trend = pd.Series(0, index=series.index)
        trend[slopes > 0] = 1
        trend[slopes < 0] = -1
        
        return trend
    
    def detect_trend_macd(
        self,
        series: pd.Series
    ) -> pd.Series:
        """
        Detect trend using MACD (Moving Average Convergence Divergence).
        
        Trend is positive when MACD line > signal line.
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
            
        Returns:
        --------
        pd.Series : Trend signal
        """
        fast = self.params['fast']
        slow = self.params['slow']
        signal_period = self.params['signal']
        
        # Calculate MACD
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        trend = pd.Series(0, index=series.index)
        trend[macd_line > signal_line] = 1
        trend[macd_line < signal_line] = -1
        
        return trend
    
    def detect_trend_rate_of_change(
        self,
        series: pd.Series
    ) -> pd.Series:
        """
        Detect trend using Rate of Change (ROC).
        
        Positive ROC indicates uptrend.
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
            
        Returns:
        --------
        pd.Series : Trend signal
        """
        window = self.params['window']
        
        roc = series.pct_change(periods=window)
        
        trend = pd.Series(0, index=series.index)
        trend[roc > 0] = 1
        trend[roc < 0] = -1
        
        return trend
    
    def detect_trend(
        self,
        series: pd.Series
    ) -> pd.Series:
        """
        Detect trend using the configured method.
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
            
        Returns:
        --------
        pd.Series : Trend signal
        """
        method_map = {
            TrendMethod.SMA_CROSSOVER: self.detect_trend_sma_crossover,
            TrendMethod.LINEAR_REGRESSION: self.detect_trend_linear_regression,
            TrendMethod.MACD: self.detect_trend_macd,
            TrendMethod.RATE_OF_CHANGE: self.detect_trend_rate_of_change
        }
        
        return method_map[self.trend_method](series)
    
    def generate_signals_for_pair(
        self,
        predictor_volatility: pd.Series,
        target_ticker: str
    ) -> pd.DataFrame:
        """
        Generate trading signals for a single pair.
        
        Parameters:
        -----------
        predictor_volatility : pd.Series
            Volatility time series of predictor stock
        target_ticker : str
            Ticker symbol of target stock
            
        Returns:
        --------
        pd.DataFrame : DataFrame with dates, trend, and signals
        """
        # Detect trend in predictor's volatility
        trend = self.detect_trend(predictor_volatility)
        
        # Generate signals based on trend
        signals = pd.Series(Signal.HOLD.value, index=trend.index)
        signals[trend == 1] = Signal.BUY.value
        signals[trend == -1] = Signal.SELL.value
        
        result = pd.DataFrame({
            'date': trend.index,
            'predictor_volatility': predictor_volatility.values,
            'trend': trend.values,
            'signal': signals.values,
            'target': target_ticker
        })
        
        result.set_index('date', inplace=True)
        
        return result
    
    def generate_signals_for_all_pairs(
        self,
        volatility_df: pd.DataFrame,
        trading_pairs: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate signals for all trading pairs.
        
        Parameters:
        -----------
        volatility_df : pd.DataFrame
            Volatility time series for all assets
        trading_pairs : pd.DataFrame
            DataFrame with predictor->target relationships
            
        Returns:
        --------
        Dict[str, pd.DataFrame] : Dictionary of {pair_name: signals_df}
        """
        all_signals = {}
        
        for _, row in trading_pairs.iterrows():
            predictor = row['predictor']
            target = row['target']
            pair_name = f"{predictor}->{target}"
            
            signals = self.generate_signals_for_pair(
                volatility_df[predictor],
                target
            )
            
            all_signals[pair_name] = signals
        
        return all_signals
    
    def get_active_signals(
        self,
        signals_dict: Dict[str, pd.DataFrame],
        date: pd.Timestamp
    ) -> Dict[str, int]:
        """
        Get active signals for all pairs on a specific date.
        
        Parameters:
        -----------
        signals_dict : Dict[str, pd.DataFrame]
            Dictionary of signals for all pairs
        date : pd.Timestamp
            Date to query
            
        Returns:
        --------
        Dict[str, int] : Dictionary of {pair_name: signal_value}
        """
        active_signals = {}
        
        for pair_name, signals_df in signals_dict.items():
            if date in signals_df.index:
                signal = signals_df.loc[date, 'signal']
                if signal != Signal.HOLD.value:
                    active_signals[pair_name] = signal
        
        return active_signals


class SignalAnalyzer:
    """
    Analyze trading signals for quality and characteristics.
    """
    
    @staticmethod
    def count_signals(signals_df: pd.DataFrame) -> Dict[str, int]:
        """
        Count different signal types.
        
        Parameters:
        -----------
        signals_df : pd.DataFrame
            Signals dataframe
            
        Returns:
        --------
        Dict[str, int] : Count of each signal type
        """
        return {
            'buy': (signals_df['signal'] == Signal.BUY.value).sum(),
            'sell': (signals_df['signal'] == Signal.SELL.value).sum(),
            'hold': (signals_df['signal'] == Signal.HOLD.value).sum(),
            'total': len(signals_df)
        }
    
    @staticmethod
    def calculate_signal_changes(signals_df: pd.DataFrame) -> pd.Series:
        """
        Calculate when signals change (trade triggers).
        
        Parameters:
        -----------
        signals_df : pd.DataFrame
            Signals dataframe
            
        Returns:
        --------
        pd.Series : Boolean series indicating signal changes
        """
        signal_changes = signals_df['signal'].diff() != 0
        signal_changes.iloc[0] = True  # First signal is always a change
        return signal_changes
    
    @staticmethod
    def get_signal_statistics(
        signals_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Get statistics for all signal sets.
        
        Parameters:
        -----------
        signals_dict : Dict[str, pd.DataFrame]
            Dictionary of signals for all pairs
            
        Returns:
        --------
        pd.DataFrame : Statistics for each pair
        """
        stats = []
        
        for pair_name, signals_df in signals_dict.items():
            counts = SignalAnalyzer.count_signals(signals_df)
            changes = SignalAnalyzer.calculate_signal_changes(signals_df)
            
            stats.append({
                'pair': pair_name,
                'n_buy': counts['buy'],
                'n_sell': counts['sell'],
                'n_hold': counts['hold'],
                'total_days': counts['total'],
                'n_trades': changes.sum() - 1,  # Subtract initial signal
                'pct_active': (counts['buy'] + counts['sell']) / counts['total'] * 100
            })
        
        return pd.DataFrame(stats)
    
    @staticmethod
    def plot_signals(
        signals_df: pd.DataFrame,
        price_data: Optional[pd.Series] = None,
        title: str = "Trading Signals",
        save_path: str = None
    ) -> None:
        """
        Plot trading signals along with price data.
        
        Parameters:
        -----------
        signals_df : pd.DataFrame
            Signals dataframe
        price_data : pd.Series, optional
            Price data to overlay
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: Predictor volatility
        axes[0].plot(signals_df.index, signals_df['predictor_volatility'], 
                     label='Predictor Volatility', color='blue', linewidth=1.5)
        axes[0].set_ylabel('Volatility')
        axes[0].set_title(f'{title} - Predictor Volatility')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Trend
        axes[1].plot(signals_df.index, signals_df['trend'], 
                     label='Trend', color='green', linewidth=1.5)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].fill_between(signals_df.index, 0, signals_df['trend'], 
                             where=signals_df['trend'] > 0, alpha=0.3, color='green', label='Uptrend')
        axes[1].fill_between(signals_df.index, 0, signals_df['trend'], 
                             where=signals_df['trend'] < 0, alpha=0.3, color='red', label='Downtrend')
        axes[1].set_ylabel('Trend')
        axes[1].set_title('Trend Detection')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Signals
        buy_signals = signals_df[signals_df['signal'] == Signal.BUY.value]
        sell_signals = signals_df[signals_df['signal'] == Signal.SELL.value]
        
        if price_data is not None:
            # Align price data with signals
            aligned_price = price_data.reindex(signals_df.index, method='ffill')
            axes[2].plot(aligned_price.index, aligned_price.values, 
                        label='Price', color='black', linewidth=1.5, alpha=0.7)
            
            # Mark buy/sell signals on price chart
            axes[2].scatter(buy_signals.index, aligned_price.loc[buy_signals.index], 
                           color='green', marker='^', s=100, label='BUY', zorder=5)
            axes[2].scatter(sell_signals.index, aligned_price.loc[sell_signals.index], 
                           color='red', marker='v', s=100, label='SELL', zorder=5)
            axes[2].set_ylabel('Price')
        else:
            # Just plot signals
            axes[2].plot(signals_df.index, signals_df['signal'], 
                        label='Signal', color='purple', linewidth=1.5)
            axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        axes[2].set_xlabel('Date')
        axes[2].set_title('Trading Signals')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    from volatility_estimators import calculate_volatility_for_assets
    from volatility_clustering import cluster_assets_by_volatility
    from granger_causality import identify_trading_pairs
    
    # Download data
    tickers = ['MSFT', 'GOOGL', 'NVDA', 'AMZN', 'META', 'QCOM', 'IBM', 'INTC', 'MU']
    
    print("Downloading data...")
    data_dict = {}
    for ticker in tickers:
        df = yf.download(ticker, start="2020-05-01", end="2023-05-31", progress=False)
        data_dict[ticker] = df
    
    # Calculate volatility
    print("\nCalculating volatility...")
    volatility_df = calculate_volatility_for_assets(data_dict, estimator='yang_zhang')
    
    # Cluster and identify pairs
    print("\nClustering and identifying pairs...")
    clustering, mid_cluster = cluster_assets_by_volatility(volatility_df, n_clusters=3)
    trading_pairs, _ = identify_trading_pairs(volatility_df, mid_cluster, target_lag=5)
    
    if len(trading_pairs) == 0:
        print("No trading pairs found.")
    else:
        # Generate signals
        print("\n" + "="*60)
        print("GENERATING TRADING SIGNALS")
        print("="*60)
        
        signal_gen = SignalGenerator(
            trend_method='sma_crossover',
            trend_params={'fast_period': 5, 'slow_period': 20}
        )
        
        signals = signal_gen.generate_signals_for_all_pairs(volatility_df, trading_pairs)
        
        # Analyze signals
        stats = SignalAnalyzer.get_signal_statistics(signals)
        print("\nSignal Statistics:")
        print(stats.to_string(index=False))
        
        # Plot first pair
        first_pair = list(signals.keys())[0]
        print(f"\nPlotting signals for {first_pair}...")
        
        target_ticker = trading_pairs.iloc[0]['target']
        target_price = data_dict[target_ticker]['Close']
        
        SignalAnalyzer.plot_signals(
            signals[first_pair],
            price_data=target_price,
            title=first_pair
        )
