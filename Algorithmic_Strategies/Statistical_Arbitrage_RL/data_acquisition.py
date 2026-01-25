"""
Data Acquisition Module for Statistical Arbitrage RL Strategy

Fetches S&P 500 constituent data with sector classifications.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import yaml
from pathlib import Path


class DataAcquisition:
    """Fetch and preprocess S&P 500 stock data for pairs trading."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.pair_config = self.config['pair_selection']
        
    def get_sp500_tickers(self) -> pd.DataFrame:
        """
        Fetch current S&P 500 constituents with sector information.
        
        Returns:
            DataFrame with columns: ticker, sector, industry
        """
        # Fetch S&P 500 list from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        
        # Clean and format
        sp500_df = sp500_table[['Symbol', 'GICS Sector', 'GICS Sub-Industry']].copy()
        sp500_df.columns = ['ticker', 'sector', 'industry']
        
        # Filter for target sectors
        target_sectors = self.pair_config['sectors']
        sp500_df = sp500_df[sp500_df['sector'].isin(target_sectors)]
        
        return sp500_df
    
    def fetch_price_data(self, 
                         tickers: List[str], 
                         start_date: str, 
                         end_date: str) -> pd.DataFrame:
        """
        Download historical price data for list of tickers.
        
        Args:
            tickers: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with MultiIndex columns (ticker, price_field)
        """
        print(f"Fetching price data for {len(tickers)} tickers...")
        
        # Download data with progress bar
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval=self.data_config['frequency'],
            progress=True,
            group_by='ticker'
        )
        
        return data
    
    def get_adjusted_close(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract adjusted close prices and handle missing data.
        
        Args:
            price_data: Raw price data from yfinance
            
        Returns:
            DataFrame with tickers as columns, dates as index
        """
        # Handle both single and multiple tickers
        if isinstance(price_data.columns, pd.MultiIndex):
            adj_close = price_data.xs('Adj Close', axis=1, level=1)
        else:
            adj_close = price_data[['Adj Close']].copy()
            adj_close.columns = [price_data.columns.name] if price_data.columns.name else ['Ticker']
        
        # Forward fill up to 5 days, then drop remaining NaNs
        adj_close = adj_close.fillna(method='ffill', limit=5)
        
        # Remove tickers with >20% missing data
        missing_pct = adj_close.isna().sum() / len(adj_close)
        valid_tickers = missing_pct[missing_pct < 0.2].index.tolist()
        adj_close = adj_close[valid_tickers]
        
        # Drop any remaining rows with NaN
        adj_close = adj_close.dropna()
        
        return adj_close
    
    def get_volume_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract volume data."""
        if isinstance(price_data.columns, pd.MultiIndex):
            volume = price_data.xs('Volume', axis=1, level=1)
        else:
            volume = price_data[['Volume']].copy()
            volume.columns = [price_data.columns.name] if price_data.columns.name else ['Ticker']
        
        volume = volume.fillna(method='ffill', limit=5).dropna()
        return volume
    
    def fetch_full_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch complete dataset for strategy.
        
        Returns:
            Dictionary containing:
                - 'constituents': S&P 500 constituents with sectors
                - 'prices': Adjusted close prices
                - 'volumes': Trading volumes
                - 'metadata': Data quality metrics
        """
        # Get S&P 500 constituents
        constituents = self.get_sp500_tickers()
        print(f"Found {len(constituents)} stocks in target sectors")
        
        # Fetch price data
        tickers = constituents['ticker'].tolist()
        price_data = self.fetch_price_data(
            tickers,
            self.data_config['start_date'],
            self.data_config['test_end']
        )
        
        # Extract adjusted close and volume
        prices = self.get_adjusted_close(price_data)
        volumes = self.get_volume_data(price_data)
        
        # Align constituents with available data
        available_tickers = prices.columns.tolist()
        constituents = constituents[constituents['ticker'].isin(available_tickers)]
        
        # Calculate metadata
        metadata = {
            'total_tickers': len(constituents),
            'date_range': (prices.index[0], prices.index[-1]),
            'trading_days': len(prices),
            'sectors': constituents['sector'].value_counts().to_dict(),
            'missing_data_pct': (prices.isna().sum() / len(prices)).to_dict()
        }
        
        return {
            'constituents': constituents,
            'prices': prices,
            'volumes': volumes,
            'metadata': metadata
        }
    
    def split_train_test(self, 
                         prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing periods.
        
        Args:
            prices: Full price DataFrame
            
        Returns:
            (train_prices, test_prices)
        """
        train_start = pd.to_datetime(self.data_config['train_start'])
        train_end = pd.to_datetime(self.data_config['train_end'])
        test_start = pd.to_datetime(self.data_config['test_start'])
        test_end = pd.to_datetime(self.data_config['test_end'])
        
        train_prices = prices.loc[train_start:train_end]
        test_prices = prices.loc[test_start:test_end]
        
        return train_prices, test_prices


if __name__ == "__main__":
    # Test data acquisition
    data_acq = DataAcquisition()
    
    print("Fetching S&P 500 data...")
    dataset = data_acq.fetch_full_dataset()
    
    print("\n=== Data Summary ===")
    print(f"Total stocks: {dataset['metadata']['total_tickers']}")
    print(f"Date range: {dataset['metadata']['date_range']}")
    print(f"Trading days: {dataset['metadata']['trading_days']}")
    print(f"\nSector distribution:")
    for sector, count in dataset['metadata']['sectors'].items():
        print(f"  {sector}: {count}")
    
    # Split train/test
    train_prices, test_prices = data_acq.split_train_test(dataset['prices'])
    print(f"\nTrain period: {len(train_prices)} days")
    print(f"Test period: {len(test_prices)} days")
