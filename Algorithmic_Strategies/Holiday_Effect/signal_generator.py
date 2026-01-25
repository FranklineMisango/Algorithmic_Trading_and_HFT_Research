"""
Signal Generator for Holiday Effect Strategy

Detects Black Friday and Prime Day events, generates calendar-based trading signals.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import yaml


class SignalGenerator:
    """Generate calendar-based trading signals for holiday events."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.events_config = self.config['events']
        
    def get_black_friday_date(self, year: int) -> datetime:
        """
        Calculate Black Friday date for a given year.
        
        Black Friday = Friday after 4th Thursday in November.
        
        Args:
            year: Calendar year
            
        Returns:
            Black Friday date
        """
        # November 1st
        nov_first = datetime(year, 11, 1)
        
        # Find first Thursday
        days_until_thursday = (3 - nov_first.weekday()) % 7
        first_thursday = nov_first + timedelta(days=days_until_thursday)
        
        # 4th Thursday (Thanksgiving)
        thanksgiving = first_thursday + timedelta(weeks=3)
        
        # Black Friday (day after)
        black_friday = thanksgiving + timedelta(days=1)
        
        return black_friday
    
    def get_prime_day_date(self, year: int) -> datetime:
        """
        Get Prime Day date for a given year.
        
        Uses historical dates from config. Prime Day typically mid-July.
        
        Args:
            year: Calendar year
            
        Returns:
            Prime Day date (or None if before 2015)
        """
        prime_dates = self.events_config['prime_day']['dates']
        
        if year < 2015:
            # Prime Day started in 2015
            return None
        
        if year in prime_dates:
            return pd.to_datetime(prime_dates[year])
        else:
            # Estimate as mid-July if not in config
            return datetime(year, 7, 15)
    
    def get_event_windows(self, 
                          start_year: int,
                          end_year: int,
                          trading_calendar: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Generate all event windows for date range.
        
        Args:
            start_year: Start year
            end_year: End year
            trading_calendar: Valid trading days
            
        Returns:
            DataFrame with event windows (entry_date, exit_date, event_date, event_type)
        """
        windows = []
        
        for year in range(start_year, end_year + 1):
            # Black Friday
            bf_date = self.get_black_friday_date(year)
            bf_entry, bf_exit = self._calculate_window_dates(
                bf_date,
                self.events_config['black_friday']['lookback_days'],
                trading_calendar
            )
            
            if bf_entry and bf_exit:
                windows.append({
                    'year': year,
                    'event_type': 'black_friday',
                    'event_date': bf_date,
                    'entry_date': bf_entry,
                    'exit_date': bf_exit,
                    'holding_days': len(pd.bdate_range(bf_entry, bf_exit))
                })
            
            # Prime Day
            pd_date = self.get_prime_day_date(year)
            if pd_date:
                pd_entry, pd_exit = self._calculate_window_dates(
                    pd_date,
                    self.events_config['prime_day']['lookback_days'],
                    trading_calendar
                )
                
                if pd_entry and pd_exit:
                    windows.append({
                        'year': year,
                        'event_type': 'prime_day',
                        'event_date': pd_date,
                        'entry_date': pd_entry,
                        'exit_date': pd_exit,
                        'holding_days': len(pd.bdate_range(pd_entry, pd_exit))
                    })
        
        return pd.DataFrame(windows)
    
    def _calculate_window_dates(self,
                                 event_date: datetime,
                                 lookback_days: int,
                                 trading_calendar: pd.DatetimeIndex) -> Tuple:
        """
        Calculate entry and exit dates based on event date and lookback.
        
        Args:
            event_date: Event date
            lookback_days: Number of trading days to look back
            trading_calendar: Valid trading days
            
        Returns:
            (entry_date, exit_date) or (None, None) if dates not in calendar
        """
        # Find event date in trading calendar
        event_date_normalized = pd.Timestamp(event_date).normalize()
        
        # Get closest trading day before or on event date
        valid_event_dates = trading_calendar[trading_calendar <= event_date_normalized]
        
        if len(valid_event_dates) == 0:
            return None, None
        
        # Exit is day before event (last trading day)
        exit_date = valid_event_dates[-1]
        
        # Entry is lookback_days before exit
        exit_idx = trading_calendar.get_loc(exit_date)
        entry_idx = max(0, exit_idx - lookback_days)
        
        entry_date = trading_calendar[entry_idx]
        
        return entry_date, exit_date
    
    def generate_signal_series(self,
                                trading_calendar: pd.DatetimeIndex,
                                start_year: int = None,
                                end_year: int = None) -> pd.DataFrame:
        """
        Generate daily signal series.
        
        Args:
            trading_calendar: Trading days index
            start_year, end_year: Year range
            
        Returns:
            DataFrame with daily signals (1 = in event window, 0 = out)
        """
        if start_year is None:
            start_year = trading_calendar[0].year
        if end_year is None:
            end_year = trading_calendar[-1].year
        
        # Get event windows
        windows = self.get_event_windows(start_year, end_year, trading_calendar)
        
        # Create signal series
        signals = pd.DataFrame(index=trading_calendar)
        signals['in_window'] = 0
        signals['event_type'] = ''
        
        # Mark event windows
        for _, window in windows.iterrows():
            mask = (signals.index >= window['entry_date']) & (signals.index <= window['exit_date'])
            signals.loc[mask, 'in_window'] = 1
            signals.loc[mask, 'event_type'] = window['event_type']
        
        return signals, windows
    
    def apply_market_filters(self,
                             signals: pd.DataFrame,
                             spy_prices: pd.Series,
                             vix: pd.Series) -> pd.DataFrame:
        """
        Apply market regime filters to signals.
        
        Only trade when:
        - SPY above 200-day MA
        - VIX below threshold
        
        Args:
            signals: Base signals
            spy_prices: SPY adjusted close
            vix: VIX index
            
        Returns:
            Filtered signals
        """
        risk_config = self.config['risk_management']
        
        # Calculate 200-day MA
        spy_ma200 = spy_prices.rolling(window=200).mean()
        
        # Market filters
        above_ma = spy_prices > spy_ma200
        vix_low = vix < risk_config['vix_threshold']
        
        # Align with signals index
        above_ma_aligned = above_ma.reindex(signals.index).fillna(False)
        vix_low_aligned = vix_low.reindex(signals.index).fillna(False)
        
        # Apply filters
        filtered_signals = signals.copy()
        
        if risk_config['market_filter'] == 'ma_200':
            filtered_signals.loc[~above_ma_aligned, 'in_window'] = 0
        
        if 'vix_threshold' in risk_config:
            filtered_signals.loc[~vix_low_aligned, 'in_window'] = 0
        
        # Track filter statistics
        filtered_signals['above_ma200'] = above_ma_aligned
        filtered_signals['vix_ok'] = vix_low_aligned
        
        return filtered_signals


if __name__ == "__main__":
    # Test signal generation
    from data_acquisition import DataAcquisition
    
    # Fetch data
    data_acq = DataAcquisition()
    dataset = data_acq.fetch_full_dataset()
    
    # Generate signals
    signal_gen = SignalGenerator()
    
    trading_calendar = dataset['amzn_prices'].index
    signals, windows = signal_gen.generate_signal_series(trading_calendar)
    
    print("=== Event Windows ===")
    print(windows[['year', 'event_type', 'entry_date', 'exit_date', 'holding_days']])
    
    # Apply filters
    filtered_signals = signal_gen.apply_market_filters(
        signals,
        dataset['spy_prices']['Adj Close'],
        dataset['vix']
    )
    
    print(f"\n=== Signal Statistics ===")
    print(f"Total trading days: {len(signals)}")
    print(f"Days in event windows: {signals['in_window'].sum()}")
    print(f"Days after filtering: {filtered_signals['in_window'].sum()}")
    print(f"Black Friday events: {len(windows[windows['event_type'] == 'black_friday'])}")
    print(f"Prime Day events: {len(windows[windows['event_type'] == 'prime_day'])}")
