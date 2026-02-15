"""
Adapter to use main data_pipeline for Deep Learning Options Trading Strategy

This script leverages the root data_pipeline infrastructure instead of 
duplicating data fetching logic. It provides a simple interface to fetch
S&P 100 options data for LSTM-based delta-neutral straddle trading.

Usage:
    python fetch_data_main_pipeline.py
    python fetch_data_main_pipeline.py --test  # Test mode with limited data
"""

import sys
import os
import yaml
import pandas as pd
import numpy as np
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    yf = None

# Add root data_pipeline to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "data_pipeline"))

from databento_options_downloader import DatabentoOptionsDownloader
from config import DATA_ROOT
from utils import setup_logging


class DeepLearningOptionsDataFetcher:
    """
    High-level interface to fetch options data using the main data_pipeline.
    Specifically designed for the Deep Learning Options Trading strategy.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with Deep Learning strategy config"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = setup_logging()
        
        # S&P 100 constituents (can be expanded)
        self.sp100_tickers = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ',
            'V', 'PG', 'UNH', 'HD', 'MA', 'PFE', 'KO', 'DIS', 'NFLX', 'ADBE',
            'CRM', 'AMD', 'INTC', 'CSCO', 'VZ', 'T', 'CMCSA', 'PEP', 'ABT', 'COST'
        ]
        
        # Get API key from config and pass to downloader
        databento_key = self.config['data'].get('databento_api_key')
        if not databento_key:
            self.logger.error("No Databento API key found in config.yaml")
            self.logger.error("Set: data.databento_api_key: 'YOUR_KEY_HERE'")
            raise ValueError("Databento API key not configured")
        
        # Debug: Check key length (don't print full key for security)
        self.logger.info(f"Using Databento API key (length: {len(databento_key)}, starts with: {databento_key[:6]}...)")
        
        # Initialize main pipeline downloaders with explicit API key
        self.options_downloader = DatabentoOptionsDownloader(api_key=databento_key)
        
    def fetch_full_dataset(self, 
                          start_date: str = None,
                          end_date: str = None,
                          test_mode: bool = False) -> tuple:
        """
        Fetch complete dataset using main data_pipeline infrastructure.
        
        Args:
            start_date: Start date in YYYY-MM-DD format (default from config)
            end_date: End date in YYYY-MM-DD format (default from config)
            test_mode: If True, use limited symbols and date range for testing
            
        Returns:
            Tuple of (underlying_prices_df, options_df)
        """
        # Parse dates
        start_date = start_date or self.config['data']['start_date']
        end_date = end_date or self.config['data']['end_date']
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Historical data is delayed - ensure end_date is at least 2 days ago
        max_end_date = datetime.now() - timedelta(days=2)
        if end_dt > max_end_date:
            self.logger.warning(f"End date {end_dt.date()} is too recent. Adjusting to {max_end_date.date()}")
            end_dt = max_end_date
        
        # Test mode adjustments
        if test_mode:
            self.sp100_tickers = self.sp100_tickers[:3]  # Only 3 tickers
            start_dt = datetime.now() - timedelta(days=60)  # Last 60 days
            end_dt = datetime.now() - timedelta(days=2)  # Historical data lags 1-2 days
            self.logger.info("Running in TEST MODE with limited data")
        
        self.logger.info(f"Fetching data from {start_dt.date()} to {end_dt.date()}")
        self.logger.info(f"Tickers: {', '.join(self.sp100_tickers)}")
        
        # Step 1: Fetch underlying prices using main pipeline CLI
        self.logger.info("\n=== STEP 1: Fetching Underlying Prices ===")
        underlying_prices = self._fetch_underlying_via_pipeline(
            self.sp100_tickers,
            start_dt,
            end_dt
        )
        
        # Step 2: Fetch options data via Databento
        self.logger.info("\n=== STEP 2: Fetching Options Data ===")
        options_data = self._fetch_options_data(
            self.sp100_tickers,
            start_dt,
            end_dt
        )
        
        # Step 3: Process and combine data
        self.logger.info("\n=== STEP 3: Processing Data ===")
        processed_prices, processed_options = self._process_data(
            underlying_prices,
            options_data
        )
        
        # Step 4: Save to local strategy data directory
        self._save_to_strategy_format(processed_prices, processed_options)
        
        return processed_prices, processed_options
    
    def _fetch_underlying_via_pipeline(self, 
                                       tickers: list,
                                       start_date: datetime,
                                       end_date: datetime) -> pd.DataFrame:
        """
        Fetch underlying stock prices by calling main data_pipeline directly.
        Falls back to manual CSV if pipeline fails.
        """
        try:
            self.logger.info(f"Downloading {len(tickers)} equity symbols via main pipeline...")
            self.logger.info("NOTE: This requires configured API keys in data_pipeline/.env")
            
            # Call main pipeline as subprocess
            pipeline_script = REPO_ROOT / "data_pipeline" / "main.py"
            
            cmd = [
                sys.executable,
                str(pipeline_script),
                "--source", "alpaca",
                "--equity-symbols"] + tickers + [
                "--start-date", start_date.strftime('%Y-%m-%d'),
                "--end-date", end_date.strftime('%Y-%m-%d'),
                "--resolution", "daily"
            ]
            
            self.logger.info(f"Running: {' '.join(cmd)}")
            
            # Run pipeline (this will download to data/ directory)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT / "data_pipeline")
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Pipeline returned error: {result.stderr}")
                self.logger.info("Falling back to manual CSV loading...")
                return self._load_or_create_empty_prices(tickers, start_date, end_date)
            
                # Load downloaded LEAN format data (this needs custom parsing)
            self.logger.info("Loading downloaded data from LEAN format...")
            return self._load_lean_equity_data(tickers)
            
        except Exception as e:
            self.logger.warning(f"Pipeline execution failed: {e}")
            self.logger.info("Falling back to manual CSV loading...")
            return self._load_or_create_empty_prices(tickers, start_date, end_date)
    
    def _load_or_create_empty_prices(self, tickers: list, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fallback: Load from CSV or fetch from yfinance"""
        # Try to load from existing CSV
        csv_path = Path(self.config['data']['underlying_data_path']) / 'underlying_prices.csv'
        
        if csv_path.exists():
            try:
                self.logger.info(f"Loading existing underlying prices from {csv_path}")
                df = pd.read_csv(csv_path)
                df['date'] = pd.to_datetime(df['date'])
                self.logger.info(f"✓ Loaded {len(df)} records from existing CSV")
                return df
            except Exception as e:
                self.logger.warning(f"Failed to load existing CSV: {e}")
        
        # Try to fetch from yfinance
        if yf is not None:
            self.logger.info("Fetching underlying prices from Yahoo Finance...")
            try:
                all_data = []
                for ticker in tickers:
                    self.logger.info(f"Downloading {ticker} from {start_date.date()} to {end_date.date()}")
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not data.empty:
                        data = data.reset_index()
                        data['ticker'] = ticker
                        data['date'] = pd.to_datetime(data['Date'])
                        data = data[['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
                        data.columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
                        all_data.append(data)
                        self.logger.info(f"✓ Downloaded {len(data)} records for {ticker}")
                    else:
                        self.logger.warning(f"No data available for {ticker}")
                
                if all_data:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    self.logger.info(f"✓ Total: {len(combined_df)} price records from Yahoo Finance")
                    return combined_df
            except Exception as e:
                self.logger.warning(f"Failed to fetch from Yahoo Finance: {e}")
        
        # If no data available, create a message for the user
        self.logger.error("="*70)
        self.logger.error("NO UNDERLYING PRICE DATA AVAILABLE")
        self.logger.error("="*70)
        self.logger.error("Options:")
        self.logger.error("1. Set up Alpaca API keys in data_pipeline/.env")
        self.logger.error("2. OR place underlying_prices.csv in data/underlying_prices/")
        self.logger.error("3. OR run the main pipeline manually:")
        self.logger.error(f"   cd data_pipeline")
        self.logger.error(f"   python main.py --source alpaca --equity-symbols {' '.join(tickers[:3])} --resolution daily")
        self.logger.error("4. OR install yfinance: pip install yfinance")
        self.logger.error("="*70)
        
        raise ValueError("No underlying price data available. See options above.")
    
    def _load_lean_equity_data(self, tickers: list) -> pd.DataFrame:
        """Load equity data from LEAN format (if downloaded by pipeline)"""
        all_data = []
        equity_path = Path(DATA_ROOT) / 'equity' / 'usa'
        
        self.logger.info(f"Looking for LEAN data in: {equity_path}")
        
        if not equity_path.exists():
            self.logger.warning("LEAN equity data path doesn't exist")
            return pd.DataFrame()
        
        for ticker in tickers:
            ticker_path = equity_path / ticker.lower()
            
            if not ticker_path.exists():
                continue
            
            # Find CSV files in the ticker directory
            csv_files = list(ticker_path.glob("*.csv"))
            
            if not csv_files:
                continue
            
            for csv_file in csv_files:
                try:
                    # LEAN format varies by resolution
                    # For daily: date_ohlcv.csv with millisecond timestamps
                    df = pd.read_csv(csv_file, header=None)
                    
                    # Attempt to parse LEAN format
                    if len(df.columns) >= 5:
                        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'][:len(df.columns)]
                        df['ticker'] = ticker
                        df['date'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
                        
                        # Convert LEAN deci-cents to dollars
                        for col in ['open', 'high', 'low', 'close']:
                            if col in df.columns:
                                df[col] = df[col] / 10000.0
                        
                        all_data.append(df)
                        self.logger.info(f"✓ Loaded {len(df)} records for {ticker}")
                        break  # Only need one file per ticker
                        
                except Exception as e:
                    self.logger.warning(f"Error parsing {csv_file.name}: {e}")
                    continue
        
        if not all_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"✓ Total: {len(combined_df)} price records for {len(all_data)} tickers")
        
        return combined_df
    
    def _fetch_options_data(self,
                           tickers: list,
                           start_date: datetime,
                           end_date: datetime) -> pd.DataFrame:
        """Fetch options data using Databento via main pipeline"""
        try:
            self.logger.info(f"Downloading options for {len(tickers)} symbols via Databento...")
            
            options_source = self.config['data'].get('options_source', 'databento')
            
            if options_source != 'databento':
                self.logger.warning(f"Options source '{options_source}' not optimal. Using Databento.")
            
            # Download options for each underlying
            for ticker in tickers:
                self.logger.info(f"Fetching options for {ticker}...")
                self.options_downloader.download_options(
                    underlying=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    resolution='daily',
                    schema='ohlcv-1d',
                    limit_contracts=50,  # Limit to 50 most liquid contracts
                    filter_near_money=True  # Only near-the-money options
                )
            
            # Load downloaded options data
            all_options = []
            options_path = Path(DATA_ROOT) / 'option' / 'usa'
            
            for ticker in tickers:
                ticker_path = options_path / ticker.lower()
                
                if not ticker_path.exists():
                    self.logger.warning(f"No options data folder for {ticker}")
                    continue
                
                # Scan for option contract files
                resolution_dir = ticker_path / 'daily'  # Options are saved in daily subdirectory
                if not resolution_dir.exists():
                    self.logger.warning(f"No daily options data folder for {ticker}")
                    continue
                option_files = list(resolution_dir.glob("*.csv"))
                
                for opt_file in option_files[:50]:  # Limit files to process
                    try:
                        df = pd.read_csv(opt_file)
                        df['ticker'] = ticker
                        all_options.append(df)
                    except Exception as e:
                        self.logger.warning(f"Error reading {opt_file.name}: {e}")
                        continue
            
            if not all_options:
                raise ValueError("No options data retrieved from Databento")
            
            combined_options = pd.concat(all_options, ignore_index=True)
            
            # Filter for liquidity
            min_volume = self.config['data'].get('min_volume', 10)
            min_oi = self.config['data'].get('min_open_interest', 50)
            
            if 'volume' in combined_options.columns:
                initial_count = len(combined_options)
                combined_options = combined_options[
                    (combined_options['volume'] >= min_volume)
                ]
                self.logger.info(f"Filtered by volume: {len(combined_options)}/{initial_count} records")
            
            self.logger.info(f"✓ Loaded {len(combined_options)} options records")
            
            return combined_options
            
        except Exception as e:
            self.logger.error(f"Error fetching options data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _process_data(self,
                     prices_df: pd.DataFrame,
                     options_df: pd.DataFrame) -> tuple:
        """Process and standardize data for deep learning model"""
        self.logger.info("Processing and standardizing data...")
        
        # Calculate returns for underlying
        if 'close' in prices_df.columns:
            prices_df = prices_df.sort_values(['ticker', 'date'])
            prices_df['return_1d'] = prices_df.groupby('ticker')['close'].pct_change()
            prices_df['return_5d'] = prices_df.groupby('ticker')['close'].pct_change(5)
            
            # Calculate rolling volatility
            prices_df['volatility_30d'] = (
                prices_df.groupby('ticker')['return_1d']
                .rolling(30, min_periods=10)
                .std()
                .reset_index(0, drop=True)
            )
        
        self.logger.info(f"✓ Processed {len(prices_df)} price records")
        self.logger.info(f"✓ Processed {len(options_df)} options records")
        
        return prices_df, options_df
    
    def _save_to_strategy_format(self,
                                 prices_df: pd.DataFrame,
                                 options_df: pd.DataFrame):
        """Save data in format expected by deep learning strategy"""
        # Create data directories
        underlying_path = Path(self.config['data']['underlying_data_path'])
        options_path = Path(self.config['data']['options_data_path'])
        
        underlying_path.mkdir(parents=True, exist_ok=True)
        options_path.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        prices_file = underlying_path / 'underlying_prices.csv'
        options_file = options_path / 'options_data.csv'
        
        prices_df.to_csv(prices_file, index=False)
        options_df.to_csv(options_file, index=False)
        
        self.logger.info(f"✓ Saved underlying prices to: {prices_file}")
        self.logger.info(f"✓ Saved options data to: {options_file}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fetch options data for Deep Learning strategy using main data_pipeline'
    )
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with limited data')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Initialize fetcher
    fetcher = DeepLearningOptionsDataFetcher()
    
    # Fetch data
    print("\n" + "="*70)
    print("Deep Learning Options Trading - Data Acquisition")
    print("Using Main Data Pipeline Infrastructure")
    print("="*70 + "\n")
    
    try:
        prices, options = fetcher.fetch_full_dataset(
            start_date=args.start_date,
            end_date=args.end_date,
            test_mode=args.test
        )
        
        print("\n" + "="*70)
        print("✓ DATA ACQUISITION COMPLETE")
        print("="*70)
        print(f"Underlying Prices: {len(prices)} records")
        print(f"Options Data: {len(options)} records")
        print("\nData saved to:")
        print(f"  - data/underlying_prices/underlying_prices.csv")
        print(f"  - data/options_data/options_data.csv")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
