"""
Data Acquisition Module for Dividend-Price Ratio Strategy

This module handles:
1. Downloading S&P 500 price data
2. Obtaining dividend data
3. Calculating dividend-price (D/P) ratio
4. Data validation and cleaning
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DividendPriceDataFetcher:
    """
    Fetches and processes S&P 500 price and dividend data.
    """
    
    def __init__(self, config: dict):
        """
        Initialize data fetcher.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with data parameters
        """
        self.config = config
        self.ticker = config['data']['ticker']
        self.start_date = config['data']['start_date']
        self.end_date = config['data']['end_date'] or datetime.now().strftime('%Y-%m-%d')
        self.frequency = config['data']['frequency']
        
    def download_sp500_data(self) -> pd.DataFrame:
        """
        Download S&P 500 price data with dividends.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with Date index and columns: Open, High, Low, Close, Volume, Dividends
        """
        print(f"Downloading S&P 500 data from {self.start_date} to {self.end_date}...")
        
        try:
            # Try primary ticker
            data = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                progress=True,
                auto_adjust=False  # Get raw prices and dividends
            )
            
            if data.empty:
                raise ValueError(f"No data retrieved for {self.ticker}")
                
        except Exception as e:
            print(f"Failed to download {self.ticker}: {e}")
            print("Trying fallback tickers...")
            
            # Try fallback tickers
            for fallback in self.config['data'].get('fallback_tickers', []):
                try:
                    print(f"Trying {fallback}...")
                    data = yf.download(
                        fallback,
                        start=self.start_date,
                        end=self.end_date,
                        progress=True,
                        auto_adjust=False
                    )
                    if not data.empty:
                        print(f"Successfully downloaded {fallback}")
                        break
                except Exception as e2:
                    print(f"Failed {fallback}: {e2}")
                    continue
            else:
                raise ValueError("All data sources failed")
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Got: {data.columns.tolist()}")
        
        # Check for Dividends column
        if 'Dividends' not in data.columns:
            print("Warning: No Dividends column found. Creating zero dividends.")
            data['Dividends'] = 0.0
        
        print(f"Downloaded {len(data)} daily observations")
        return data
    
    def calculate_trailing_12m_dividends(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trailing 12-month dividends.
        
        Parameters
        ----------
        data : pd.DataFrame
            Daily data with Dividends column
            
        Returns
        -------
        pd.DataFrame
            Data with trailing_12m_div column
        """
        print("Calculating trailing 12-month dividends...")
        
        # Rolling sum of dividends over past 252 trading days (~1 year)
        data['trailing_12m_div'] = data['Dividends'].rolling(window=252, min_periods=1).sum()
        
        # Alternative: Use actual 365-day window
        # data['trailing_12m_div'] = data['Dividends'].rolling(window='365D').sum()
        
        return data
    
    def calculate_dp_ratio(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate dividend-price (D/P) ratio.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with Close prices and trailing_12m_div
            
        Returns
        -------
        pd.DataFrame
            Data with dp_ratio column
        """
        print("Calculating D/P ratio...")
        
        # D/P = Trailing 12-month dividends / Current Price
        data['dp_ratio'] = data['trailing_12m_div'] / data['Close']
        
        # Handle any infinities or NaNs
        data['dp_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Forward fill small gaps (up to 5 days)
        data['dp_ratio'].fillna(method='ffill', limit=5, inplace=True)
        
        print(f"D/P ratio range: {data['dp_ratio'].min():.4f} to {data['dp_ratio'].max():.4f}")
        
        return data
    
    def resample_to_monthly(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Resample daily data to monthly frequency.
        
        Parameters
        ----------
        data : pd.DataFrame
            Daily data
            
        Returns
        -------
        pd.DataFrame
            Monthly data (month-end)
        """
        print("Resampling to monthly frequency...")
        
        # Month-end resampling
        monthly = data.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Dividends': 'sum',  # Sum dividends within month
            'trailing_12m_div': 'last',  # Take end-of-month value
            'dp_ratio': 'last'  # Take end-of-month value
        })
        
        print(f"Resampled to {len(monthly)} monthly observations")
        
        return monthly
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate monthly returns.
        
        Parameters
        ----------
        data : pd.DataFrame
            Monthly data with Close prices
            
        Returns
        -------
        pd.DataFrame
            Data with monthly_return column
        """
        print("Calculating monthly returns...")
        
        # Simple returns: (P_t - P_t-1) / P_t-1
        data['monthly_return'] = data['Close'].pct_change()
        
        # Log returns: log(P_t / P_t-1)
        data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Total return (including dividends paid during month)
        # Approximation: Price return + Dividend yield
        data['dividend_yield_monthly'] = data['Dividends'] / data['Close'].shift(1)
        data['total_return'] = data['monthly_return'] + data['dividend_yield_monthly']
        
        return data
    
    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to validate
            
        Returns
        -------
        pd.DataFrame
            Cleaned data
        """
        print("Validating data...")
        
        initial_len = len(data)
        
        # Remove rows with NaN in critical columns
        critical_cols = ['Close', 'dp_ratio', 'monthly_return']
        data = data.dropna(subset=critical_cols)
        
        # Remove extreme outliers (returns > 50% or < -50%)
        data = data[(data['monthly_return'] > -0.5) & (data['monthly_return'] < 0.5)]
        
        # Remove rows where D/P ratio is zero or negative
        data = data[data['dp_ratio'] > 0]
        
        final_len = len(data)
        removed = initial_len - final_len
        
        if removed > 0:
            print(f"Removed {removed} invalid rows ({removed/initial_len*100:.1f}%)")
        
        print(f"Final dataset: {final_len} valid monthly observations")
        
        return data
    
    def fetch_and_prepare_data(self) -> pd.DataFrame:
        """
        Main method to fetch and prepare all data.
        
        Returns
        -------
        pd.DataFrame
            Fully prepared monthly data with all features
        """
        print("="*60)
        print("DIVIDEND-PRICE RATIO DATA ACQUISITION")
        print("="*60)
        
        # 1. Download raw data
        data = self.download_sp500_data()
        
        # 2. Calculate trailing dividends
        data = self.calculate_trailing_12m_dividends(data)
        
        # 3. Calculate D/P ratio
        data = self.calculate_dp_ratio(data)
        
        # 4. Resample to monthly
        monthly_data = self.resample_to_monthly(data)
        
        # 5. Calculate returns
        monthly_data = self.calculate_returns(monthly_data)
        
        # 6. Validate and clean
        monthly_data = self.validate_data(monthly_data)
        
        print("\nData summary:")
        print(f"  Date range: {monthly_data.index[0].strftime('%Y-%m-%d')} to {monthly_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"  Total months: {len(monthly_data)}")
        print(f"  Avg monthly return: {monthly_data['monthly_return'].mean()*100:.2f}%")
        print(f"  Avg D/P ratio: {monthly_data['dp_ratio'].mean()*100:.2f}%")
        print(f"  D/P ratio std: {monthly_data['dp_ratio'].std()*100:.2f}%")
        
        print("="*60)
        
        return monthly_data


def main():
    """
    Test data acquisition module.
    """
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Fetch data
    fetcher = DividendPriceDataFetcher(config)
    data = fetcher.fetch_and_prepare_data()
    
    # Display sample
    print("\nFirst 5 rows:")
    print(data.head())
    
    print("\nLast 5 rows:")
    print(data.tail())
    
    print("\nData info:")
    print(data.info())
    
    print("\nDescriptive statistics:")
    print(data[['Close', 'dp_ratio', 'monthly_return', 'total_return']].describe())
    
    # Save to CSV
    output_file = 'results/sp500_monthly_data.csv'
    data.to_csv(output_file)
    print(f"\nData saved to {output_file}")


if __name__ == "__main__":
    main()
