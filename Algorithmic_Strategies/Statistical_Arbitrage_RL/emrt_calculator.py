"""
EMRT (Empirical Mean Reversion Time) Calculator

Custom metric to measure actual historical time for price spread reversion.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import yaml


class EMRTCalculator:
    """Calculate Empirical Mean Reversion Time for stock pairs."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.emrt_config = self.config['emrt']
        self.window = self.emrt_config['calculation_window']
        self.deviation_threshold = self.emrt_config['deviation_threshold']
        self.max_reversion_days = self.emrt_config['max_reversion_days']
    
    def calculate_spread(self, 
                         price1: pd.Series, 
                         price2: pd.Series) -> pd.Series:
        """
        Calculate log price spread between two stocks.
        
        Args:
            price1, price2: Price series for two stocks
            
        Returns:
            Log price ratio (spread)
        """
        spread = np.log(price1 / price2)
        return spread
    
    def calculate_zscore(self, 
                         spread: pd.Series, 
                         window: int = None) -> pd.Series:
        """
        Calculate rolling z-score of spread.
        
        Args:
            spread: Price spread series
            window: Rolling window (default from config)
            
        Returns:
            Z-score normalized spread
        """
        if window is None:
            window = self.window
        
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        
        zscore = (spread - rolling_mean) / rolling_std
        return zscore
    
    def identify_deviation_events(self, 
                                   zscore: pd.Series) -> pd.DataFrame:
        """
        Identify when spread deviates beyond threshold.
        
        Args:
            zscore: Z-score normalized spread
            
        Returns:
            DataFrame with deviation events (start_date, direction)
        """
        threshold = self.deviation_threshold
        
        # Find crossings
        above_threshold = zscore > threshold
        below_threshold = zscore < -threshold
        
        # Detect changes (new deviation events)
        events = []
        
        # Positive deviations (spread too high)
        above_start = above_threshold & ~above_threshold.shift(1).fillna(False)
        for date in zscore[above_start].index:
            events.append({
                'start_date': date,
                'direction': 'positive',
                'zscore': zscore.loc[date]
            })
        
        # Negative deviations (spread too low)
        below_start = below_threshold & ~below_threshold.shift(1).fillna(False)
        for date in zscore[below_start].index:
            events.append({
                'start_date': date,
                'direction': 'negative',
                'zscore': zscore.loc[date]
            })
        
        events_df = pd.DataFrame(events)
        return events_df
    
    def calculate_reversion_time(self, 
                                  zscore: pd.Series, 
                                  event_start: pd.Timestamp,
                                  direction: str) -> int:
        """
        Calculate time for spread to revert to mean after deviation.
        
        Args:
            zscore: Z-score series
            event_start: Start of deviation event
            direction: 'positive' or 'negative'
            
        Returns:
            Number of days until reversion (or max if didn't revert)
        """
        # Get data after event start
        future_zscore = zscore.loc[event_start:]
        
        # Define reversion: zscore crosses back through zero
        if direction == 'positive':
            # Was above threshold, looking for cross below 0
            reverted = future_zscore < 0
        else:
            # Was below -threshold, looking for cross above 0
            reverted = future_zscore > 0
        
        # Find first reversion
        reversion_dates = future_zscore[reverted].index
        
        if len(reversion_dates) > 0:
            reversion_date = reversion_dates[0]
            days_to_revert = (reversion_date - event_start).days
            
            # Cap at max reversion days
            days_to_revert = min(days_to_revert, self.max_reversion_days)
        else:
            # Didn't revert within available data
            days_to_revert = self.max_reversion_days
        
        return days_to_revert
    
    def calculate_emrt(self, 
                       price1: pd.Series, 
                       price2: pd.Series) -> Tuple[float, dict]:
        """
        Calculate Empirical Mean Reversion Time for a stock pair.
        
        Args:
            price1, price2: Price series for the two stocks
            
        Returns:
            (emrt_value, details_dict)
            - emrt_value: Average days to mean reversion
            - details_dict: Metadata about calculation
        """
        # Calculate spread and z-score
        spread = self.calculate_spread(price1, price2)
        zscore = self.calculate_zscore(spread)
        
        # Identify deviation events
        events = self.identify_deviation_events(zscore)
        
        if len(events) == 0:
            # No deviation events found
            return np.inf, {'num_events': 0, 'reason': 'no_deviations'}
        
        # Calculate reversion time for each event
        reversion_times = []
        for _, event in events.iterrows():
            days_to_revert = self.calculate_reversion_time(
                zscore,
                event['start_date'],
                event['direction']
            )
            reversion_times.append(days_to_revert)
        
        # Calculate EMRT as mean reversion time
        emrt = np.mean(reversion_times)
        
        details = {
            'num_events': len(events),
            'reversion_times': reversion_times,
            'mean_reversion_time': emrt,
            'std_reversion_time': np.std(reversion_times),
            'median_reversion_time': np.median(reversion_times)
        }
        
        return emrt, details
    
    def calculate_emrt_batch(self, 
                             prices: pd.DataFrame,
                             pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Calculate EMRT for multiple stock pairs.
        
        Args:
            prices: DataFrame with stock prices (columns = tickers)
            pairs: List of (ticker1, ticker2) tuples
            
        Returns:
            DataFrame with EMRT values and metadata for each pair
        """
        results = []
        
        for ticker1, ticker2 in pairs:
            if ticker1 not in prices.columns or ticker2 not in prices.columns:
                continue
            
            emrt, details = self.calculate_emrt(
                prices[ticker1],
                prices[ticker2]
            )
            
            results.append({
                'ticker1': ticker1,
                'ticker2': ticker2,
                'emrt': emrt,
                'num_events': details['num_events'],
                'std_reversion_time': details.get('std_reversion_time', np.nan)
            })
        
        results_df = pd.DataFrame(results)
        return results_df


if __name__ == "__main__":
    # Test EMRT calculation
    from data_acquisition import DataAcquisition
    
    # Fetch data
    data_acq = DataAcquisition()
    dataset = data_acq.fetch_full_dataset()
    train_prices, _ = data_acq.split_train_test(dataset['prices'])
    
    # Test with known pair (MSFT, GOOGL)
    emrt_calc = EMRTCalculator()
    
    if 'MSFT' in train_prices.columns and 'GOOGL' in train_prices.columns:
        emrt, details = emrt_calc.calculate_emrt(
            train_prices['MSFT'],
            train_prices['GOOGL']
        )
        
        print("=== MSFT-GOOGL Pair ===")
        print(f"EMRT: {emrt:.2f} days")
        print(f"Number of deviation events: {details['num_events']}")
        print(f"Std reversion time: {details.get('std_reversion_time', 'N/A')}")
