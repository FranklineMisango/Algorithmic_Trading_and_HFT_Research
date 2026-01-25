"""
Data Acquisition Module for AI-Enhanced 60/40 Portfolio

This module handles fetching market data and economic indicators
for the AI-driven portfolio allocation strategy.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataAcquisition:
    """Fetch and process market data and economic indicators."""
    
    def __init__(self, config: Dict):
        """
        Initialize data acquisition.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.start_date = config['data']['start_date']
        self.end_date = config['data']['end_date']
        
    def fetch_asset_prices(self) -> pd.DataFrame:
        """
        Fetch historical prices for all assets.
        
        Returns:
            DataFrame with adjusted close prices for all assets
        """
        print("Fetching asset prices...")
        
        # Collect all tickers
        tickers = []
        for asset in self.config['assets']['traditional']:
            tickers.append(asset['ticker'])
        for asset in self.config['assets']['alternative']:
            tickers.append(asset['ticker'])
        
        # Fetch data using yfinance
        data = yf.download(
            tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False
        )
        
        # Extract adjusted close prices
        if len(tickers) == 1:
            prices = pd.DataFrame(data['Adj Close'])
            prices.columns = tickers
        else:
            prices = data['Adj Close']
        
        # Fill missing values (forward fill then backward fill)
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Fetched prices for {len(tickers)} assets from {prices.index[0]} to {prices.index[-1]}")
        
        return prices
    
    def fetch_vix(self) -> pd.Series:
        """
        Fetch VIX (CBOE Volatility Index).
        
        Returns:
            Series with VIX values
        """
        print("Fetching VIX data...")
        
        vix_ticker = self.config['indicators']['vix']['ticker']
        vix_data = yf.download(
            vix_ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False
        )
        
        vix = vix_data['Adj Close']
        vix.name = 'VIX'
        
        return vix
    
    def fetch_yield_spread(self) -> pd.Series:
        """
        Fetch and calculate yield spread (10Y - 3M Treasury).
        
        Returns:
            Series with yield spread values
        """
        print("Fetching yield spread data...")
        
        long_term = self.config['indicators']['yield_spread']['long_term']
        short_term = self.config['indicators']['yield_spread']['short_term']
        
        # Fetch both yields
        yields_data = yf.download(
            [long_term, short_term],
            start=self.start_date,
            end=self.end_date,
            progress=False
        )
        
        # Calculate spread
        if len([long_term, short_term]) > 1:
            long_yield = yields_data['Adj Close'][long_term]
            short_yield = yields_data['Adj Close'][short_term]
        else:
            long_yield = yields_data['Adj Close']
            short_yield = yields_data['Adj Close']
        
        spread = long_yield - short_yield
        spread.name = 'Yield_Spread'
        
        return spread
    
    def fetch_interest_rate(self) -> pd.Series:
        """
        Fetch Federal Funds Rate from FRED.
        
        Returns:
            Series with interest rate values
        """
        print("Fetching interest rate data...")
        
        try:
            # Try to fetch from FRED
            rate = pdr.DataReader(
                'DFF',
                'fred',
                start=self.start_date,
                end=self.end_date
            )
            rate = rate['DFF']
            rate.name = 'Interest_Rate'
        except Exception as e:
            print(f"Warning: Could not fetch from FRED: {e}")
            print("Using 10Y Treasury as proxy for interest rates...")
            
            # Fallback to 10Y Treasury
            treasury = yf.download(
                '^TNX',
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            rate = treasury['Adj Close']
            rate.name = 'Interest_Rate'
        
        return rate
    
    def fetch_all_indicators(self) -> pd.DataFrame:
        """
        Fetch all economic indicators.
        
        Returns:
            DataFrame with all indicators
        """
        vix = self.fetch_vix()
        yield_spread = self.fetch_yield_spread()
        interest_rate = self.fetch_interest_rate()
        
        # Combine all indicators
        indicators = pd.concat([vix, yield_spread, interest_rate], axis=1)
        
        # Fill missing values
        indicators = indicators.fillna(method='ffill').fillna(method='bfill')
        
        print(f"\nIndicators summary:")
        print(indicators.describe())
        
        return indicators
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns from prices.
        
        Args:
            prices: DataFrame with asset prices
            
        Returns:
            DataFrame with returns
        """
        returns = prices.pct_change().dropna()
        return returns
    
    def resample_to_monthly(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to monthly frequency.
        
        Args:
            data: DataFrame with daily data
            
        Returns:
            DataFrame with monthly data
        """
        # Use last value of each month
        monthly_data = data.resample('M').last()
        return monthly_data
    
    def get_full_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch complete dataset: prices, returns, and indicators.
        
        Returns:
            Tuple of (prices, returns, indicators) DataFrames
        """
        prices = self.fetch_asset_prices()
        indicators = self.fetch_all_indicators()
        
        # Align dates
        common_dates = prices.index.intersection(indicators.index)
        prices = prices.loc[common_dates]
        indicators = indicators.loc[common_dates]
        
        # Calculate returns
        returns = self.calculate_returns(prices)
        
        # Resample to monthly
        prices_monthly = self.resample_to_monthly(prices)
        returns_monthly = self.resample_to_monthly(returns)
        indicators_monthly = self.resample_to_monthly(indicators)
        
        return prices_monthly, returns_monthly, indicators_monthly


if __name__ == "__main__":
    # Test the module
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_acq = DataAcquisition(config)
    prices, returns, indicators = data_acq.get_full_dataset()
    
    print("\n" + "="*50)
    print("Data Acquisition Complete!")
    print("="*50)
    print(f"\nPrices shape: {prices.shape}")
    print(f"Returns shape: {returns.shape}")
    print(f"Indicators shape: {indicators.shape}")
    print(f"\nDate range: {prices.index[0]} to {prices.index[-1]}")
