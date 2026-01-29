"""
Data Acquisition Module for Holiday Effect Strategy

Fetches Amazon (AMZN) and S&P 500 (SPY) historical data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple
import yaml


class DataAcquisition:
    """Fetch and preprocess stock data for Holiday Effect strategy."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.primary_ticker = self.data_config['primary_ticker']
        self.benchmark_ticker = self.data_config['benchmark_ticker']
        
    def fetch_price_data(self, 
                         ticker: str,
                         start_date: str = None,
                         end_date: str = None) -> pd.DataFrame:
        """
        Download historical price data.
        
        Args:
            ticker: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = self.data_config['start_date']
        if end_date is None:
            end_date = self.data_config['end_date']
        
        print(f"Fetching {ticker} data from {start_date} to {end_date}...")
        
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        return data
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.Series:
        """Calculate daily returns from adjusted close prices."""
        if 'Adj Close' in prices.columns:
            return prices['Adj Close'].pct_change()
        else:
            return prices['Close'].pct_change()
    
    def fetch_vix_data(self, start_date: str = None, end_date: str = None) -> pd.Series:
        """
        Fetch VIX (volatility index) for market filter.
        
        Args:
            start_date, end_date: Date range
            
        Returns:
            VIX closing prices
        """
        if start_date is None:
            start_date = self.data_config['start_date']
        if end_date is None:
            end_date = self.data_config['end_date']
        
        print(f"Fetching VIX data...")
        
        vix_data = yf.download(
            '^VIX',
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )
        
        # Flatten MultiIndex columns if present
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = vix_data.columns.get_level_values(0)
        
        return vix_data['Close']
    
    def fetch_full_dataset(self) -> Dict:
        """
        Fetch complete dataset for strategy.
        
        Returns:
            Dictionary with AMZN, SPY, and VIX data
        """
        # Fetch primary security (AMZN)
        amzn_data = self.fetch_price_data(self.primary_ticker)
        
        # Fetch benchmark (SPY)
        spy_data = self.fetch_price_data(self.benchmark_ticker)
        
        # Fetch VIX for market filter
        vix_data = self.fetch_vix_data()
        
        # Align all data on common dates
        common_dates = amzn_data.index.intersection(spy_data.index)
        
        amzn_aligned = amzn_data.loc[common_dates]
        spy_aligned = spy_data.loc[common_dates]
        vix_aligned = vix_data.reindex(common_dates).ffill()
        
        # Calculate returns
        amzn_returns = self.calculate_returns(amzn_aligned)
        spy_returns = self.calculate_returns(spy_aligned)
        
        # Calculate 200-day MA for SPY (market filter)
        price_col = 'Adj Close' if 'Adj Close' in spy_aligned.columns else 'Close'
        spy_ma200 = spy_aligned[price_col].rolling(window=200).mean()
        
        dataset = {
            'amzn_prices': amzn_aligned,
            'spy_prices': spy_aligned,
            'vix': vix_aligned,
            'amzn_returns': amzn_returns,
            'spy_returns': spy_returns,
            'spy_ma200': spy_ma200,
            'start_date': common_dates[0],
            'end_date': common_dates[-1],
            'trading_days': len(common_dates)
        }
        
        return dataset
    
    def split_data(self, 
                   data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into discovery, validation, and holdout periods.
        
        Args:
            data: Full dataset
            
        Returns:
            (discovery_data, validation_data, holdout_data)
        """
        oos_year = self.config['backtest']['oos_split_year']
        validation_year = self.config['backtest']['validation_start']
        holdout_year = self.config['backtest']['holdout_start']
        
        discovery_end = pd.to_datetime(f'{oos_year}-12-31')
        validation_end = pd.to_datetime(f'{holdout_year-1}-12-31')
        
        discovery = data.loc[:discovery_end]
        validation = data.loc[f'{validation_year}-01-01':validation_end]
        holdout = data.loc[f'{holdout_year}-01-01':]
        
        return discovery, validation, holdout


if __name__ == "__main__":
    # Test data acquisition
    data_acq = DataAcquisition()
    
    print("Fetching full dataset...")
    dataset = data_acq.fetch_full_dataset()
    
    print("\n=== Dataset Summary ===")
    print(f"Date range: {dataset['start_date']} to {dataset['end_date']}")
    print(f"Trading days: {dataset['trading_days']}")
    print(f"\nAMZN price range: ${dataset['amzn_prices']['Adj Close' if 'Adj Close' in dataset['amzn_prices'].columns else 'Close'].min():.2f} - ${dataset['amzn_prices']['Adj Close' if 'Adj Close' in dataset['amzn_prices'].columns else 'Close'].max():.2f}")
    print(f"SPY price range: ${dataset['spy_prices']['Adj Close' if 'Adj Close' in dataset['spy_prices'].columns else 'Close'].min():.2f} - ${dataset['spy_prices']['Adj Close' if 'Adj Close' in dataset['spy_prices'].columns else 'Close'].max():.2f}")
    
    # Split data
    discovery, validation, holdout = data_acq.split_data(dataset['amzn_prices'])
    print(f"\nDiscovery period: {len(discovery)} days")
    print(f"Validation period: {len(validation)} days")
    print(f"Holdout period: {len(holdout)} days")
