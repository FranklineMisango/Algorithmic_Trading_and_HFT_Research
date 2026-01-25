"""
Data Acquisition for DRL Portfolio Allocation

Fetches multi-asset data and prepares features for RL agent.
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    print("Warning: yfinance not installed")


class DataAcquisition:
    """Handles data fetching and preprocessing."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.symbols = self.config['assets']['symbols']
    
    def fetch_prices(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch adjusted close prices for all assets.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dict of price DataFrames by symbol
        """
        prices = {}
        
        for symbol in self.symbols:
            try:
                data = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                if not data.empty:
                    prices[symbol] = data[['Adj Close']].rename(columns={'Adj Close': symbol})
                    print(f"Fetched {symbol}: {len(data)} days")
                
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
        
        return prices
    
    def combine_prices(
        self,
        prices: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Combine individual price series into single DataFrame.
        
        Args:
            prices: Dict of price DataFrames
            
        Returns:
            Combined DataFrame with all assets
        """
        # Merge all on date index
        combined = pd.DataFrame()
        
        for symbol, df in prices.items():
            if combined.empty:
                combined = df
            else:
                combined = combined.join(df, how='outer')
        
        # Forward fill missing values (max 5 days)
        combined = combined.ffill(limit=5)
        
        # Drop rows with any remaining NaN
        combined = combined.dropna()
        
        print(f"\nCombined data: {len(combined)} days, {len(combined.columns)} assets")
        
        return combined
    
    def calculate_returns(
        self,
        prices: pd.DataFrame,
        method: str = 'log'
    ) -> pd.DataFrame:
        """
        Calculate asset returns.
        
        Args:
            prices: Price DataFrame
            method: 'log' or 'simple'
            
        Returns:
            Returns DataFrame
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        
        return returns.dropna()
    
    def split_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets.
        
        Args:
            df: Full dataset
            
        Returns:
            train, validation, test DataFrames
        """
        n = len(df)
        
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['validation_ratio']
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train = df.iloc[:train_end]
        validation = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
        
        print(f"\nData split:")
        print(f"  Train: {len(train)} days ({train.index[0]} to {train.index[-1]})")
        print(f"  Validation: {len(validation)} days ({validation.index[0]} to {validation.index[-1]})")
        print(f"  Test: {len(test)} days ({test.index[0]} to {test.index[-1]})")
        
        return train, validation, test
    
    def fetch_full_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch complete dataset with train/val/test splits.
        
        Returns:
            Dict with 'prices', 'returns', 'train', 'val', 'test'
        """
        # Fetch prices
        start_date = self.config['data']['start_date']
        end_date = self.config['data']['end_date']
        
        print(f"Fetching data from {start_date} to {end_date}...")
        prices_dict = self.fetch_prices(start_date, end_date)
        
        # Combine
        prices = self.combine_prices(prices_dict)
        
        # Calculate returns
        returns = self.calculate_returns(prices)
        
        # Split
        train_prices, val_prices, test_prices = self.split_data(prices)
        train_returns, val_returns, test_returns = self.split_data(returns)
        
        return {
            'prices': prices,
            'returns': returns,
            'train': {
                'prices': train_prices,
                'returns': train_returns
            },
            'val': {
                'prices': val_prices,
                'returns': val_returns
            },
            'test': {
                'prices': test_prices,
                'returns': test_returns
            }
        }


# Test code
if __name__ == "__main__":
    data_acq = DataAcquisition('config.yaml')
    
    dataset = data_acq.fetch_full_dataset()
    
    print(f"\nPrices shape: {dataset['prices'].shape}")
    print(f"\nFirst few rows:")
    print(dataset['prices'].head())
    
    print(f"\nReturns statistics:")
    print(dataset['returns'].describe())
