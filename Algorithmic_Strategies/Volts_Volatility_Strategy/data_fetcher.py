"""
Data fetcher with Alpaca and yfinance fallback
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
import os
from typing import Dict, List, Optional

# Try to import Alpaca
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Alpaca not available, will use yfinance only")


def download_data_alpaca(
    tickers: List[str],
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Download data using Alpaca API.
    
    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    api_key : Optional[str]
        Alpaca API key (or set ALPACA_API_KEY env var)
    secret_key : Optional[str]
        Alpaca secret key (or set ALPACA_SECRET_KEY env var)
        
    Returns:
    --------
    Dict[str, pd.DataFrame] : Dictionary with ticker as key and OHLCV DataFrame as value
    """
    if not ALPACA_AVAILABLE:
        raise ImportError("alpaca-trade-api not installed. Install with: pip install alpaca-trade-api")
    
    # Get credentials
    api_key = api_key or os.getenv('ALPACA_API_KEY')
    secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        raise ValueError("Alpaca credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
    
    # Initialize Alpaca API
    api = tradeapi.REST(
        api_key,
        secret_key,
        'https://paper-api.alpaca.markets',  # Use paper trading endpoint for data
        api_version='v2'
    )
    
    data_dict = {}
    
    for ticker in tickers:
        try:
            print(f"  Fetching {ticker} from Alpaca...")
            
            # Get bars from Alpaca
            bars = api.get_bars(
                ticker,
                '1Day',
                start=start_date,
                end=end_date,
                adjustment='raw'
            ).df
            
            if len(bars) > 0:
                # Rename columns to match yfinance format
                bars = bars.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Remove timezone info if present
                if bars.index.tz is not None:
                    bars.index = bars.index.tz_localize(None)
                
                data_dict[ticker] = bars
                print(f"    {ticker}: {len(bars)} days")
            else:
                print(f"    {ticker}: No data available")
                
        except Exception as e:
            print(f"    {ticker}: Error - {e}")
    
    return data_dict


def download_data_yfinance(
    tickers: List[str],
    start_date: str,
    end_date: str,
    max_retries: int = 3
) -> Dict[str, pd.DataFrame]:
    """
    Download data using yfinance with retry logic.
    
    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    max_retries : int
        Maximum number of retry attempts
        
    Returns:
    --------
    Dict[str, pd.DataFrame] : Dictionary with ticker as key and OHLCV DataFrame as value
    """
    data_dict = {}
    
    for ticker in tickers:
        success = False
        
        for attempt in range(max_retries):
            try:
                print(f"  Fetching {ticker} from yfinance (attempt {attempt + 1}/{max_retries})...")
                
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if len(df) > 0:
                    # Fix MultiIndex columns issue
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    
                    data_dict[ticker] = df
                    print(f"    {ticker}: {len(df)} days ✓")
                    success = True
                    break
                else:
                    print(f"    {ticker}: No data available")
                    break
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    {ticker}: Error - {e}. Retrying...")
                    import time
                    time.sleep(2)  # Wait 2 seconds before retry
                else:
                    print(f"    {ticker}: Failed after {max_retries} attempts - {e}")
    
    return data_dict


def download_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    source: str = 'auto',
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Download data with automatic source selection or fallback.
    
    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    source : str
        Data source: 'alpaca', 'yfinance', or 'auto' (tries Alpaca first, falls back to yfinance)
    **kwargs : dict
        Additional arguments for specific downloaders
        
    Returns:
    --------
    Dict[str, pd.DataFrame] : Dictionary with ticker as key and OHLCV DataFrame as value
    """
    print(f"Downloading data for {len(tickers)} tickers...")
    print(f"Period: {start_date} to {end_date}\n")
    
    if source == 'alpaca':
        return download_data_alpaca(tickers, start_date, end_date, **kwargs)
    
    elif source == 'yfinance':
        return download_data_yfinance(tickers, start_date, end_date, **kwargs)
    
    elif source == 'auto':
        # Try Alpaca first if available and credentials are set
        if ALPACA_AVAILABLE:
            api_key = kwargs.get('api_key') or os.getenv('ALPACA_API_KEY')
            secret_key = kwargs.get('secret_key') or os.getenv('ALPACA_SECRET_KEY')
            
            if api_key and secret_key:
                print("Using Alpaca API (faster and more reliable)\n")
                try:
                    return download_data_alpaca(tickers, start_date, end_date, api_key, secret_key)
                except Exception as e:
                    print(f"\nAlpaca failed: {e}")
                    print("Falling back to yfinance...\n")
        
        # Fallback to yfinance
        print("Using yfinance\n")
        return download_data_yfinance(tickers, start_date, end_date, **kwargs)
    
    else:
        raise ValueError(f"Unknown source: {source}. Use 'alpaca', 'yfinance', or 'auto'")
