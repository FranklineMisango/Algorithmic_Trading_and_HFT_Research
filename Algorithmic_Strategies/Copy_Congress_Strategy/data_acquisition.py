"""
Data Acquisition Module for Copy Congress Strategy

This module handles fetching Congressional trade data and market data
for systematic replication of Congressional stock trades.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import os
import requests
warnings.filterwarnings('ignore')


class CongressionalDataAcquisition:
    """Fetch Congressional trade disclosures and market data."""
    
    def __init__(self, config: Dict):
        """
        Initialize data acquisition.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.start_date = config['data']['start_date']
        self.end_date = config['data']['end_date']
        self.api_key = os.getenv(config['data_sources']['congressional_trades'].get('api_key_env', ''))
        
    def fetch_congressional_trades_sample(self) -> pd.DataFrame:
        """
        Fetch sample Congressional trade data.
        
        NOTE: This is a placeholder. In production, integrate with:
        - Quiver Quantitative API
        - Capitol Trades
        - Senate/House disclosure databases
        
        Returns:
            DataFrame with Congressional trades
        """
        print("Fetching Congressional trade data...")
        print("NOTE: Using synthetic sample data. Replace with real API in production.")
        
        # Generate realistic sample data for demonstration
        # In production, replace with actual API calls
        
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        # Sample tickers that Congress frequently trades
        common_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            'JPM', 'BAC', 'GS', 'V', 'MA',
            'JNJ', 'UNH', 'PFE', 'ABBV',
            'XOM', 'CVX', 'COP',
            'DIS', 'NFLX', 'CMCSA'
        ]
        
        # Sample politicians
        politicians = [
            'Rep. A', 'Rep. B', 'Rep. C', 'Sen. X', 'Sen. Y', 'Sen. Z',
            'Rep. D', 'Rep. E', 'Sen. W'
        ]
        
        # Sample committees
        committees = [
            'Finance', 'Healthcare', 'Energy', 'Technology', 
            'Banking', 'Judiciary', 'Defense'
        ]
        
        parties = ['Democrat', 'Republican']
        
        # Generate sample trades
        np.random.seed(42)
        n_trades = 1000
        
        trades = []
        for _ in range(n_trades):
            # Random transaction date
            trans_date = np.random.choice(date_range)
            
            # Filing date is 1-45 days after transaction
            filing_delay = np.random.randint(1, 46)
            filing_date = trans_date + timedelta(days=filing_delay)
            
            # Ensure filing date doesn't exceed end_date
            if filing_date > pd.Timestamp(self.end_date):
                filing_date = pd.Timestamp(self.end_date)
            
            trade = {
                'filing_date': filing_date,
                'transaction_date': trans_date,
                'politician': np.random.choice(politicians),
                'party': np.random.choice(parties),
                'committee': np.random.choice(committees),
                'ticker': np.random.choice(common_tickers),
                'transaction_type': np.random.choice(['Buy', 'Sell'], p=[0.55, 0.45]),
                'amount': np.random.choice([5000, 15000, 50000, 100000, 250000, 500000]),
                'shares': 0  # Will calculate based on price
            }
            
            trades.append(trade)
        
        df = pd.DataFrame(trades)
        
        # Sort by filing date
        df = df.sort_values('filing_date').reset_index(drop=True)
        
        print(f"Fetched {len(df)} Congressional trades")
        print(f"Date range: {df['filing_date'].min()} to {df['filing_date'].max()}")
        
        return df
    
    def fetch_market_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch market data for given tickers.
        
        Args:
            tickers: List of stock tickers
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching market data for {len(tickers)} tickers...")
        
        # Fetch data using yfinance
        data = yf.download(
            tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=True  # Adjust for splits/dividends
        )
        
        # Extract adjusted close prices
        if len(tickers) == 1:
            prices = pd.DataFrame(data['Close'])
            prices.columns = tickers
            volumes = pd.DataFrame(data['Volume'])
            volumes.columns = tickers
        else:
            prices = data['Close']
            volumes = data['Volume']
        
        # Fill missing values
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        volumes = volumes.fillna(0)
        
        print(f"Fetched prices from {prices.index[0]} to {prices.index[-1]}")
        
        return prices, volumes
    
    def calculate_market_caps(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate market capitalization.
        
        Args:
            prices: DataFrame with prices
            volumes: DataFrame with volumes
            
        Returns:
            DataFrame with estimated market caps
        """
        # This is a simplified estimation
        # In production, fetch actual market cap data from financial APIs
        
        print("Calculating market capitalizations...")
        
        market_caps = pd.DataFrame(index=prices.index, columns=prices.columns)
        
        for ticker in prices.columns:
            # Rough estimate: assume shares outstanding is consistent
            # Better: fetch actual shares outstanding data
            avg_volume = volumes[ticker].rolling(window=60).mean()
            market_caps[ticker] = prices[ticker] * avg_volume * 100  # Rough multiplier
        
        return market_caps
    
    def calculate_volatility(self, prices: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """
        Calculate historical volatility.
        
        Args:
            prices: DataFrame with prices
            window: Lookback window in days
            
        Returns:
            DataFrame with annualized volatility
        """
        print(f"Calculating {window}-day historical volatility...")
        
        returns = prices.pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Apply minimum volatility floor
        min_vol = self.config['weighting']['min_volatility']
        volatility = volatility.clip(lower=min_vol)
        
        return volatility
    
    def apply_universe_filters(self, 
                              prices: pd.DataFrame,
                              volumes: pd.DataFrame,
                              market_caps: pd.DataFrame) -> pd.DataFrame:
        """
        Apply universe filters.
        
        Args:
            prices: DataFrame with prices
            volumes: DataFrame with volumes
            market_caps: DataFrame with market caps
            
        Returns:
            Boolean DataFrame indicating eligible stocks
        """
        print("Applying universe filters...")
        
        min_price = self.config['universe']['min_price']
        min_market_cap = self.config['universe']['min_market_cap']
        
        # Create filter mask
        eligible = (
            (prices >= min_price) &  # Minimum price
            (market_caps >= min_market_cap) &  # Minimum market cap
            (volumes > 0)  # Must have volume
        )
        
        return eligible
    
    def clean_congressional_data(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate Congressional trade data.
        
        Args:
            trades_df: Raw Congressional trades DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("Cleaning Congressional trade data...")
        
        df = trades_df.copy()
        
        # Filter minimum transaction size
        min_size = self.config['signal']['min_transaction_size']
        df = df[df['amount'] >= min_size].copy()
        
        # Remove invalid tickers
        df = df[df['ticker'].notna()].copy()
        df = df[df['ticker'].str.len() <= 5].copy()  # Reasonable ticker length
        
        # Standardize transaction types
        df['transaction_type'] = df['transaction_type'].str.lower()
        df['transaction_type'] = df['transaction_type'].replace({
            'purchase': 'buy',
            'sale': 'sell',
            'sold': 'sell',
            'bought': 'buy'
        })
        
        # Ensure dates are datetime
        df['filing_date'] = pd.to_datetime(df['filing_date'])
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        print(f"Cleaned data: {len(df)} trades remaining")
        
        return df
    
    def get_full_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch complete dataset.
        
        Returns:
            Tuple of (congressional_trades, prices, volumes, market_caps, volatility)
        """
        # Fetch Congressional trades
        congressional_trades = self.fetch_congressional_trades_sample()
        congressional_trades = self.clean_congressional_data(congressional_trades)
        
        # Get unique tickers
        tickers = congressional_trades['ticker'].unique().tolist()
        print(f"\nUnique tickers in Congressional trades: {len(tickers)}")
        
        # Fetch market data
        prices, volumes = self.fetch_market_data(tickers)
        
        # Calculate market caps and volatility
        market_caps = self.calculate_market_caps(prices, volumes)
        volatility = self.calculate_volatility(
            prices,
            window=self.config['weighting']['volatility_lookback']
        )
        
        # Apply universe filters
        eligible = self.apply_universe_filters(prices, volumes, market_caps)
        
        # Filter trades to only eligible universe
        valid_tickers = eligible.any(axis=0)
        valid_tickers = valid_tickers[valid_tickers].index.tolist()
        
        congressional_trades = congressional_trades[
            congressional_trades['ticker'].isin(valid_tickers)
        ].copy()
        
        print(f"\nFinal dataset:")
        print(f"  Congressional trades: {len(congressional_trades)}")
        print(f"  Valid tickers: {len(valid_tickers)}")
        print(f"  Price data shape: {prices.shape}")
        
        return congressional_trades, prices, volumes, market_caps, volatility


if __name__ == "__main__":
    # Test the module
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_acq = CongressionalDataAcquisition(config)
    congressional_trades, prices, volumes, market_caps, volatility = data_acq.get_full_dataset()
    
    print("\n" + "="*60)
    print("Data Acquisition Complete!")
    print("="*60)
    print(f"\nCongressional Trades Sample:")
    print(congressional_trades.head(10))
    print(f"\nPrices shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
