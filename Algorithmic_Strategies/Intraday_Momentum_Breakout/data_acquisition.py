"""
Data Acquisition Module

Downloads historical futures data for backtesting and live trading.

Data Sources:
- Primary: yfinance (free, limited futures data)
- Alternative: Interactive Brokers API, Databento
- Format: 5-minute intraday bars

Instruments:
- ES: E-mini S&P 500 Futures
- NQ: E-mini Nasdaq-100 Futures
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class FuturesDataDownloader:
    """
    Downloads and processes futures data.
    """
    
    def __init__(self, config: dict):
        """
        Initialize data downloader.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.instruments = config['data']['instruments']
        self.start_date = config['data']['start_date']
        self.end_date = config['data'].get('end_date', datetime.now().strftime('%Y-%m-%d'))
        self.timeframe = config['data']['timeframe']
    
    def download_es_futures(self, start: str, end: str) -> pd.DataFrame:
        """
        Download ES futures data.
        
        Note: yfinance has limited futures data. For production, use:
        - Interactive Brokers API
        - Databento
        - CME DataMine
        
        Parameters
        ----------
        start : str
            Start date (YYYY-MM-DD)
        end : str
            End date (YYYY-MM-DD)
            
        Returns
        -------
        pd.DataFrame
            ES futures OHLCV data
        """
        print("Downloading ES futures data...")
        
        # ES futures symbol in yfinance
        # Note: This downloads the continuous contract (ES=F)
        # For specific contracts, use ES{MONTH}{YEAR}.CME (e.g., ESH24.CME)
        symbol = 'ES=F'
        
        try:
            # Download data
            data = yf.download(
                symbol,
                start=start,
                end=end,
                interval=self.timeframe,
                progress=False
            )
            
            # Standardize columns
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Filter trading hours (RTH: 9:30 AM - 4:00 PM ET)
            data = self._filter_trading_hours(data)
            
            print(f"  Downloaded {len(data)} bars")
            print(f"  Date range: {data.index[0]} to {data.index[-1]}")
            
            return data
        
        except Exception as e:
            print(f"Error downloading ES data: {e}")
            raise
    
    def download_nq_futures(self, start: str, end: str) -> pd.DataFrame:
        """
        Download NQ futures data.
        
        Parameters
        ----------
        start : str
            Start date (YYYY-MM-DD)
        end : str
            End date (YYYY-MM-DD)
            
        Returns
        -------
        pd.DataFrame
            NQ futures OHLCV data
        """
        print("Downloading NQ futures data...")
        
        # NQ futures symbol in yfinance
        symbol = 'NQ=F'
        
        try:
            # Download data
            data = yf.download(
                symbol,
                start=start,
                end=end,
                interval=self.timeframe,
                progress=False
            )
            
            # Standardize columns
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Filter trading hours
            data = self._filter_trading_hours(data)
            
            print(f"  Downloaded {len(data)} bars")
            print(f"  Date range: {data.index[0]} to {data.index[-1]}")
            
            return data
        
        except Exception as e:
            print(f"Error downloading NQ data: {e}")
            raise
    
    def _filter_trading_hours(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to regular trading hours (9:30 AM - 4:00 PM ET).
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw data
            
        Returns
        -------
        pd.DataFrame
            Filtered data
        """
        # Convert to ET timezone
        data.index = pd.to_datetime(data.index)
        
        # Filter by time
        data['hour'] = data.index.hour
        data['minute'] = data.index.minute
        
        # RTH: 9:30 AM - 4:00 PM
        rth_data = data[
            ((data['hour'] == 9) & (data['minute'] >= 30)) |
            ((data['hour'] >= 10) & (data['hour'] < 16)) |
            ((data['hour'] == 16) & (data['minute'] == 0))
        ]
        
        # Drop helper columns
        rth_data = rth_data.drop(['hour', 'minute'], axis=1)
        
        return rth_data
    
    def download_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Download data for all instruments.
        
        Returns
        -------
        dict
            Data for each instrument
        """
        print("="*60)
        print("DATA ACQUISITION")
        print("="*60)
        print(f"Date range: {self.start_date} to {self.end_date}")
        print(f"Timeframe: {self.timeframe}")
        print()
        
        data = {}
        
        # Download ES
        data['ES'] = self.download_es_futures(self.start_date, self.end_date)
        
        # Download NQ
        data['NQ'] = self.download_nq_futures(self.start_date, self.end_date)
        
        print("\nData acquisition complete!")
        print("="*60)
        
        return data
    
    def save_data(self, data: Dict[str, pd.DataFrame], output_dir: str = 'data'):
        """
        Save data to CSV files.
        
        Parameters
        ----------
        data : dict
            Data for each instrument
        output_dir : str
            Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol, df in data.items():
            filename = f"{output_dir}/{symbol}_5min.csv"
            df.to_csv(filename)
            print(f"Saved {symbol} data to {filename}")
    
    def load_data(self, input_dir: str = 'data') -> Dict[str, pd.DataFrame]:
        """
        Load data from CSV files.
        
        Parameters
        ----------
        input_dir : str
            Input directory
            
        Returns
        -------
        dict
            Data for each instrument
        """
        import os
        
        data = {}
        
        for instrument in self.instruments:
            symbol = instrument['symbol']
            filename = f"{input_dir}/{symbol}_5min.csv"
            
            if os.path.exists(filename):
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                data[symbol] = df
                print(f"Loaded {symbol} data from {filename} ({len(df)} bars)")
            else:
                print(f"File not found: {filename}")
        
        return data


def main():
    """
    Download futures data.
    """
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Download data
    downloader = FuturesDataDownloader(config)
    data = downloader.download_all_data()
    
    # Save data
    downloader.save_data(data, output_dir='data')
    
    # Summary statistics
    for symbol, df in data.items():
        print(f"\n{symbol} Statistics:")
        print(f"  Bars: {len(df)}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
        print(f"  Avg volume: {df['Volume'].mean():,.0f}")


if __name__ == "__main__":
    main()
