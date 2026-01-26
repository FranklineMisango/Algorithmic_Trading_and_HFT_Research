"""
Data acquisition module for Foreign Market Lead-Lag ML Strategy.
Downloads S&P 500 constituents and 47 foreign market indices.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataAcquisition:
    """Handles data download for target universe and foreign markets."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.start_date = config['data']['start_date']
        self.end_date = config['data']['end_date']
        self.foreign_markets = config['data']['foreign_markets']
        self.target_universe = config['data']['target_universe']
        
    def get_sp500_constituents(self) -> List[str]:
        """Get S&P 500 constituent tickers."""
        logger.info("Fetching S&P 500 constituents...")
        
        # Download S&P 500 constituents table
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        
        # Clean tickers (replace dots with dashes for Yahoo Finance)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        logger.info(f"Found {len(tickers)} S&P 500 constituents")
        return tickers
    
    def download_daily_prices(self, tickers: List[str]) -> pd.DataFrame:
        """Download daily adjusted close prices for given tickers."""
        logger.info(f"Downloading daily prices for {len(tickers)} tickers...")
        
        data = yf.download(
            tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            group_by='ticker',
            auto_adjust=True
        )
        
        # Extract close prices
        if len(tickers) == 1:
            prices = data['Close'].to_frame(tickers[0])
        else:
            prices = data.xs('Close', level=1, axis=1)
        
        logger.info(f"Downloaded {len(prices)} days of data")
        return prices
    
    def download_foreign_markets(self) -> pd.DataFrame:
        """Download daily prices for foreign market ETFs."""
        logger.info(f"Downloading {len(self.foreign_markets)} foreign market ETFs...")
        
        foreign_prices = self.download_daily_prices(self.foreign_markets)
        
        # Handle missing data
        foreign_prices = foreign_prices.fillna(method='ffill').fillna(method='bfill')
        
        return foreign_prices
    
    def resample_to_weekly(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        """Resample daily prices to weekly (Friday close)."""
        weekly_freq = self.config['data']['weekly_frequency']
        weekly_prices = daily_prices.resample(weekly_freq).last()
        
        logger.info(f"Resampled to {len(weekly_prices)} weekly observations")
        return weekly_prices
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate simple returns."""
        returns = prices.pct_change()
        return returns
    
    def get_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Download all required data.
        
        Returns:
            Tuple of (sp500_daily_prices, sp500_daily_returns, foreign_weekly_returns)
        """
        # Get S&P 500 constituents
        sp500_tickers = self.get_sp500_constituents()
        
        # Download S&P 500 daily prices
        sp500_daily_prices = self.download_daily_prices(sp500_tickers)
        
        # Calculate S&P 500 daily returns
        sp500_daily_returns = self.calculate_returns(sp500_daily_prices)
        
        # Download foreign market prices
        foreign_daily_prices = self.download_foreign_markets()
        
        # Resample to weekly
        foreign_weekly_prices = self.resample_to_weekly(foreign_daily_prices)
        
        # Calculate weekly returns
        foreign_weekly_returns = self.calculate_returns(foreign_weekly_prices)
        
        logger.info("Data acquisition complete")
        
        return sp500_daily_prices, sp500_daily_returns, foreign_weekly_returns
    
    def save_data(self, sp500_prices: pd.DataFrame, sp500_returns: pd.DataFrame, 
                  foreign_returns: pd.DataFrame, output_dir: str = 'data'):
        """Save downloaded data to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        sp500_prices.to_csv(f'{output_dir}/sp500_daily_prices.csv')
        sp500_returns.to_csv(f'{output_dir}/sp500_daily_returns.csv')
        foreign_returns.to_csv(f'{output_dir}/foreign_weekly_returns.csv')
        
        logger.info(f"Data saved to {output_dir}/")


if __name__ == "__main__":
    import yaml
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Download data
    data_acq = DataAcquisition(config)
    sp500_prices, sp500_returns, foreign_returns = data_acq.get_all_data()
    
    # Save data
    data_acq.save_data(sp500_prices, sp500_returns, foreign_returns)
    
    print("\nData Summary:")
    print(f"S&P 500 daily prices: {sp500_prices.shape}")
    print(f"S&P 500 daily returns: {sp500_returns.shape}")
    print(f"Foreign weekly returns: {foreign_returns.shape}")
