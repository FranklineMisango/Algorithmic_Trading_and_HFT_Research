"""
Simple Data Fetcher for Deep Learning Options Trading
Downloads underlying prices and options data using Databento and yfinance
"""

import pandas as pd
import numpy as np
import yfinance as yf
import databento as db
import yaml
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_underlying_prices(tickers, start_date, end_date):
    """Fetch underlying stock prices using yfinance"""
    logger.info(f"Fetching underlying prices for {len(tickers)} tickers...")
    
    all_data = []
    for ticker in tqdm(tickers, desc="Downloading underlying prices"):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, 
                             progress=False, auto_adjust=True)
            
            if not data.empty:
                data = data.reset_index()
                data['ticker'] = ticker
                
                # Flatten columns if needed
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
                
                all_data.append(data)
                logger.info(f"   {ticker}: {len(data)} days")
        except Exception as e:
            logger.error(f"   {ticker}: {e}")
    
    if not all_data:
        raise ValueError("No underlying price data fetched")
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Rename columns to standard format
    column_map = {
        'Date': 'date',
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    combined = combined.rename(columns=column_map)
    
    # Calculate returns
    combined = combined.sort_values(['ticker', 'date'])
    combined['return_1d'] = combined.groupby('ticker')['close'].pct_change()
    combined['return_5d'] = combined.groupby('ticker')['close'].pct_change(5)
    
    # Calculate 30-day volatility
    combined['volatility_30d'] = combined.groupby('ticker')['return_1d'].transform(
        lambda x: x.rolling(30, min_periods=10).std() * np.sqrt(252)
    )
    
    return combined


def fetch_options_databento(api_key, tickers, start_date, end_date, max_contracts=50):
    """
    Fetch options data from Databento using batch requests.
    Uses OHLCV-1D schema for clean daily data with NO NaN issues.
    Fetches day-by-day with rate limiting.
    """
    logger.info(f"Fetching options data from Databento (OHLCV-1D batch mode)...")
    
    client = db.Historical(api_key)
    all_options = []
    
    # Convert dates to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate list of dates to fetch
    current_date = start_dt
    date_list = []
    while current_date <= end_dt:
        date_list.append(current_date)
        current_date += timedelta(days=1)
    
    logger.info(f"Will fetch {len(date_list)} days using OHLCV-1D...")
    
    for ticker in tickers:
        logger.info(f"\nProcessing {ticker}...")
        ticker_options = []
        
        # Fetch day-by-day with OHLCV-1D schema
        for fetch_date in tqdm(date_list, desc=f"  {ticker} daily batches"):
            try:
                # Format date for API
                date_str = fetch_date.strftime('%Y-%m-%d')
                next_day = (fetch_date + timedelta(days=1)).strftime('%Y-%m-%d')
                
                parent_symbol = f"{ticker}.OPT"
                
                # Use ohlcv-1d - clean daily OHLC data, no NaN issues
                data = client.timeseries.get_range(
                    dataset='OPRA.PILLAR',
                    symbols=[parent_symbol],
                    schema='ohlcv-1d',
                    start=date_str,
                    end=next_day,
                    stype_in='parent',
                    limit=1000  # 1000 contracts per day
                )
                
                if data is not None:
                    df = data.to_df()
                    
                    if not df.empty:
                        df_reset = df.reset_index()
                        df_reset['ticker'] = ticker
                        
                        # Extract date - OHLCV-1D uses ts_recv as index
                        if 'ts_recv' in df_reset.columns:
                            df_reset['date'] = pd.to_datetime(df_reset['ts_recv'])
                        else:
                            df_reset['date'] = fetch_date
                        
                        # Keep relevant columns (OHLCV-1D has: open, high, low, close, volume)
                        keep_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
                        if 'symbol' in df_reset.columns:
                            keep_cols.append('symbol')
                        
                        df_clean = df_reset[[col for col in keep_cols if col in df_reset.columns]].copy()
                        
                        # Filter for valid prices and liquidity
                        if 'close' in df_clean.columns:
                            df_clean = df_clean[df_clean['close'] > 0]
                        
                        if 'volume' in df_clean.columns:
                            df_clean = df_clean[df_clean['volume'] > 0]
                        
                        if not df_clean.empty:
                            ticker_options.append(df_clean)
                            logger.info(f"     {date_str}: {len(df_clean)} contracts")
                
                # Rate limiting
                time.sleep(0.3)  # 300ms delay between requests
                
            except Exception as e:
                logger.warning(f"     {date_str}: {str(e)[:150]}")
                time.sleep(0.5)  # Longer delay on errors
                continue
        
        if ticker_options:
            ticker_df = pd.concat(ticker_options, ignore_index=True)
            all_options.append(ticker_df)
            logger.info(f"   {ticker}: {len(ticker_df)} total option records across {len(ticker_options)} days")
        else:
            logger.warning(f"   {ticker}: No data retrieved")
    
    if not all_options:
        raise ValueError("No options data fetched from Databento.")
    
    combined = pd.concat(all_options, ignore_index=True)
    logger.info(f"\nTotal options records: {len(combined)} across {combined['date'].nunique()} days")
    return combined


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch data for Deep Learning Options Trading')
    parser.add_argument('--test', action='store_true', help='Test mode with limited data')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--tickers', type=str, nargs='+', help='List of tickers')
    
    args = parser.parse_args()
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    api_key = config['data']['databento_api_key']
    
    # Set parameters
    if args.test:
        tickers = ['AAPL', 'MSFT', 'AMZN']
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # 30 days for batch testing
        end_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        logger.info("TEST MODE: 3 tickers, ~30 days (batch mode)")
    else:
        tickers = args.tickers or ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 
                                   'NVDA', 'JPM', 'JNJ', 'V']
        start_date = args.start_date or config['data']['start_date']
        end_date = args.end_date or config['data']['end_date']
    
    print("\n" + "="*70)
    print("Deep Learning Options Trading - Data Acquisition")
    print("  OPTIMIZED: OHLCV-1D + Batch requests (day-by-day)")
    print("="*70)
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Schema: ohlcv-1d (Daily OHLC - clean data, no NaN)")
    print(f"Method: Day-by-day batches, 1000 contracts/day, 300ms rate limit")
    print("="*70 + "\n")
    
    try:
        # Step 1: Fetch underlying prices
        print("\nStep 1: Fetching Underlying Stock Prices")
        print("-" * 70)
        underlying = fetch_underlying_prices(tickers, start_date, end_date)
        print(f" Downloaded {len(underlying)} price records")
        
        # Step 2: Fetch options data
        print("\nStep 2: Fetching Options Data from Databento")
        print("  Using: OHLCV-1D (Daily OHLC) - clean data")
        print("  Batch: 1000 contracts/day, 300ms rate limit")
        print("-" * 70)
        options = fetch_options_databento(api_key, tickers, start_date, end_date)
        print(f" Downloaded {len(options)} option records")
        print(f"  Date coverage: {options['date'].nunique()} unique days")
        
        # Step 3: Save data
        print("\nStep 3: Saving Data")
        print("-" * 70)
        
        Path('data/underlying_prices').mkdir(parents=True, exist_ok=True)
        Path('data/options_data').mkdir(parents=True, exist_ok=True)
        
        underlying_file = 'data/underlying_prices/underlying_prices.csv'
        options_file = 'data/options_data/options_data.csv'
        
        underlying.to_csv(underlying_file, index=False)
        options.to_csv(options_file, index=False)
        
        print(f" Saved underlying prices: {underlying_file}")
        print(f"  - Shape: {underlying.shape}")
        print(f"  - Date range: {underlying['date'].min()} to {underlying['date'].max()}")
        print(f"  - Tickers: {underlying['ticker'].nunique()}")
        
        print(f"\n Saved options data: {options_file}")
        print(f"  - Shape: {options.shape}")
        print(f"  - Date range: {options['date'].min()} to {options['date'].max()}")
        print(f"  - Tickers: {options['ticker'].nunique()}")
        
        print("\n" + "="*70)
        print(" DATA ACQUISITION COMPLETE")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
